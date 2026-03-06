"""
模块名称：评估布局 metadata 测试

模块职责：
验证 PR4 引入的 execution / semantic layout metadata
已经由评估主程序稳定落盘，且不改变 `pred_samples.npz` schema。

主要功能：
1. 用非默认 `y_in_idx` 运行一次最小 evaluate 主流程。
2. 验证 `metrics.yaml` 同时记录 execution 与 semantic layout metadata。
3. 验证 `pred_samples.npz` 仍只包含 `y_hat / y_true / logvar`。

数据流：
synthetic dataset + custom train yaml + zeroed checkpoint
    ↓
evaluate.main()
    ↓
metrics.yaml / pred_samples.npz
    ↓
layout metadata assertions

依赖模块：
- numpy
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


def _write_train_yaml(tmp_path: Path, data_dir: Path, y_in_idx: list[int]) -> Path:
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "layout_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B4_layout",
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
                    "din": 20,
                    "dout": 9,
                    "pred_len": 3,
                    "rnn_hidden": 8,
                    "rnn_layers": 1,
                    "dropout": 0.0,
                    "u_in_idx": [0, 4, 8, 12],
                    "y_in_idx": y_in_idx,
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


def test_evaluate_main_writes_layout_metadata_and_keeps_pred_samples_schema(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    y_in_idx = [1, 5, 9, 2, 6, 10, 3, 7, 11]

    base_y0 = np.asarray([0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2], dtype=np.float32)
    X = np.zeros((4, 4, 20), dtype=np.float32)
    for n in range(X.shape[0]):
        X[n, -1, y_in_idx] = base_y0 + float(n)
    Y = np.repeat((base_y0.reshape(1, 1, 9) + np.arange(4, dtype=np.float32).reshape(4, 1, 1)), 3, axis=1)

    np.savez_compressed(data_dir / "features.npz", X=X)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=Y,
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

    run_dir = tmp_path / "out" / "B4_layout"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(
        run_dir / "split_indices.npz",
        {
            "train": np.asarray([0, 1], dtype=np.int64),
            "val": np.asarray([2], dtype=np.int64),
            "test": np.asarray([3], dtype=np.int64),
        },
    )
    scalers_dir = run_dir / "scalers"
    save_scaler(scalers_dir / "x_scaler.npz", {"mean": np.zeros(20, dtype=np.float32), "std": np.ones(20, dtype=np.float32)})
    save_scaler(scalers_dir / "y_scaler.npz", {"mean": np.zeros(9, dtype=np.float32), "std": np.ones(9, dtype=np.float32)})

    train_yaml = _write_train_yaml(tmp_path, data_dir, y_in_idx)
    cfg_train = load_train_config(train_yaml)
    model = S1Predictor(cfg_train.model)
    for param in model.parameters():
        param.data.zero_()
    ckpt = run_dir / "best.pth"
    torch.save({"model": model.state_dict()}, ckpt)

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
            str(tmp_path / "eval_test"),
            "--save_samples",
            "2",
        ],
    )

    ret = evaluate.main()
    assert ret == 0

    metrics = yaml.safe_load((tmp_path / "eval_test" / "metrics.yaml").read_text(encoding="utf-8"))
    assert metrics["rmse_global"] == pytest.approx(0.0)
    assert metrics["layout"]["schema_version"] == "state_layout_v1"
    assert metrics["layout"]["execution"]["source"] == "cfg_model.y_in_idx"
    assert metrics["layout"]["execution"]["y_in_idx"] == y_in_idx
    assert metrics["layout"]["semantic"]["source"] == "canonical_acc_gyro_vel_v1"
    assert metrics["layout"]["semantic"]["validated_against_target_cols"] is True
    assert metrics["layout"]["semantic"]["component_labels"] == [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "vel_x",
        "vel_y",
        "vel_z",
    ]

    with np.load(tmp_path / "eval_test" / "pred_samples.npz", allow_pickle=False) as pred_npz:
        assert set(pred_npz.files) == {"y_hat", "y_true", "logvar"}
