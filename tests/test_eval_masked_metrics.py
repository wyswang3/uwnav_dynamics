"""
模块名称：masked eval metrics 测试

模块职责：
验证 PR5 第一阶段的评估主路径能够并行产出 dense 与 masked 指标，
并保持 `pred_samples.npz` 三键 schema 不变。

主要功能：
1. 构造带 `dvl_mask` 的最小评估数据集。
2. 验证 dense / masked horizon CSV 同时落盘。
3. 验证 masked velocity 指标只由 `target_mask` 决定，而不是 forward-fill 数值本身。

数据流：
synthetic features/labels + split/scaler + zero checkpoint
    ↓
evaluate.main()
    ↓
metrics.yaml / dense csv / masked csv / pred_samples.npz
    ↓
artifact assertions

依赖模块：
- numpy
- pytest
- torch
- yaml
- uwnav_dynamics.eval.evaluate
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

from uwnav_dynamics.dataset.normalize import save_scaler
from uwnav_dynamics.dataset.split import save_split_indices
from uwnav_dynamics.eval import evaluate
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.train.config import load_train_config


def _write_train_yaml(tmp_path: Path, data_dir: Path) -> Path:
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "masked_eval_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B5_masked_eval",
                },
                "data": {
                    "data_dir": str(data_dir),
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "split": {"train_ratio": 0.5, "val_ratio": 0.25},
                },
                "model": {
                    "name": "s1_predictor",
                    "din": 25,
                    "dout": 9,
                    "pred_len": 2,
                    "rnn_hidden": 8,
                    "rnn_layers": 1,
                    "dropout": 0.0,
                    "u_in_idx": list(range(0, 8)),
                    "y_in_idx": list(range(8, 17)),
                    "use_thruster_as_replacement": True,
                    "use_hydro_feat": True,
                    "blocks": {
                        "thruster_lag": {
                            "enabled": False,
                            "normalize_input": True,
                            "pwm_center": 7.5,
                            "pwm_half_range": 2.5,
                            "deadzone": 0.05,
                            "sat_gain": 2.0,
                            "learn_tau": True,
                            "tau_init": 0.08,
                            "tau_min": 0.01,
                            "per_channel_tau": True,
                            "dt": 0.01,
                        },
                        "hydro_ssm": {
                            "enabled": False,
                            "hidden_dim": 8,
                            "lambda_init": 0.85,
                            "act": "tanh",
                            "per_dim_lambda": True,
                        },
                        "damping": {
                            "enabled": False,
                            "v_start": 6,
                            "v_dim": 3,
                            "mode": "simple",
                            "d_init": 0.5,
                            "mlp_hidden": 8,
                        },
                        "uncertainty": {
                            "enabled": False,
                            "feat_dim": 8,
                            "hidden": 8,
                            "logvar_min": -10.0,
                            "logvar_max": 6.0,
                        },
                    },
                },
                "rollout": {"y0_source": "x_last_state", "mode": "delta_cumsum"},
                "loss": {"type": "nll_diag", "logvar_clip": [-10.0, 6.0]},
                "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0},
                "train": {"epochs": 1, "eval_every": 1, "save_best": True, "save_last": True, "metric": "val_loss"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return train_yaml


def test_evaluate_main_writes_dense_and_masked_horizon_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    X = np.zeros((4, 4, 25), dtype=np.float32)
    Y = np.zeros((4, 2, 9), dtype=np.float32)
    dvl_mask = np.ones((4, 2), dtype=bool)

    # test split 使用样本 0 和 1：
    # horizon=2 时，样本 0 的 velocity 误差被 mask 掉，样本 1 保持有效且误差为 0。
    dvl_mask[0, 1] = False
    Y[0, 1, 6:] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    Y[1, 1, 6:] = 0.0

    np.savez_compressed(data_dir / "features.npz", X=X)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=Y,
        dvl_mask=dvl_mask,
        target_cols=np.asarray(
            [
                "AccX_body_mps2",
                "AccY_body_mps2",
                "AccZ_body_mps2",
                "GyroX_body_rad_s",
                "GyroY_body_rad_s",
                "GyroZ_body_rad_s",
                "VelX_state_mps",
                "VelY_state_mps",
                "VelZ_state_mps",
            ],
            dtype=object,
        ),
    )

    run_dir = tmp_path / "out" / "B5_masked_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(
        run_dir / "split_indices.npz",
        {
            "train": np.asarray([2], dtype=np.int64),
            "val": np.asarray([3], dtype=np.int64),
            "test": np.asarray([0, 1], dtype=np.int64),
        },
    )
    save_scaler(run_dir / "scalers" / "x_scaler.npz", {"mean": np.zeros(25, dtype=np.float32), "std": np.ones(25, dtype=np.float32)})
    save_scaler(run_dir / "scalers" / "y_scaler.npz", {"mean": np.zeros(9, dtype=np.float32), "std": np.ones(9, dtype=np.float32)})

    train_yaml = _write_train_yaml(tmp_path, data_dir)
    cfg_train = load_train_config(train_yaml)
    model = S1Predictor(cfg_train.model)
    for param in model.parameters():
        param.data.zero_()
    ckpt = run_dir / "best.pth"
    torch.save({"model": model.state_dict()}, ckpt)

    out_dir = tmp_path / "eval_test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.eval.evaluate",
            "-y",
            str(train_yaml),
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
            "--device",
            "cpu",
            "--batch_size",
            "2",
            "--out_dir",
            str(out_dir),
            "--save_samples",
            "2",
        ],
    )

    assert evaluate.main() == 0

    metrics = yaml.safe_load((out_dir / "metrics.yaml").read_text(encoding="utf-8"))
    assert metrics["supervision"]["schema_version"] == "supervision_v1"
    assert metrics["supervision"]["dense_metrics"]["present"] is True
    assert metrics["supervision"]["masked_metrics"]["present"] is True
    assert metrics["supervision"]["masked_metrics"]["mask_name"] == "target_mask"
    assert metrics["supervision"]["masked_metrics"]["raw_mask_source"] == "dvl_mask"
    assert metrics["supervision"]["masked_metrics"]["applies_to_groups"] == ["vel"]
    assert metrics["supervision"]["masked_metrics"]["group_source"] == "canonical_acc_gyro_vel_v1"
    assert metrics["rmse_global_masked"] < metrics["rmse_global"]

    dense_rmse = np.loadtxt(out_dir / "rmse_by_horizon.csv", delimiter=",", skiprows=1)[:, 1:]
    masked_rmse = np.loadtxt(out_dir / "rmse_by_horizon_masked.csv", delimiter=",", skiprows=1)[:, 1:]
    dense_mae = np.loadtxt(out_dir / "mae_by_horizon.csv", delimiter=",", skiprows=1)[:, 1:]
    masked_mae = np.loadtxt(out_dir / "mae_by_horizon_masked.csv", delimiter=",", skiprows=1)[:, 1:]

    assert dense_rmse.shape == (2, 9)
    assert masked_rmse.shape == (2, 9)
    assert dense_mae.shape == (2, 9)
    assert masked_mae.shape == (2, 9)
    assert np.all(dense_rmse[1, 6:] > 0.0)
    assert np.allclose(masked_rmse[1, 6:], 0.0)
    assert np.all(dense_mae[1, 6:] > 0.0)
    assert np.allclose(masked_mae[1, 6:], 0.0)

    with np.load(out_dir / "pred_samples.npz", allow_pickle=False) as pred_npz:
        assert set(pred_npz.files) == {"y_hat", "y_true", "logvar"}
