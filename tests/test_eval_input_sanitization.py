"""
模块名称：评估输入 NaN 清洗测试

模块职责：
验证离线评估阶段会在不修改原始 dataset artifact 的前提下，
把经过 scaler 后仍残留在输入 `X` 中的非有限值清洗为 0.0，
避免稀疏辅助输入的 NaN 直接进入模型并导致评估指标变成 NaN。

主要功能：
1. 构造带输入侧 NaN 的最小 features/labels 数据集。
2. 运行 `uwnav_dynamics.eval.evaluate.main()` 覆盖正式评估入口。
3. 断言 `metrics.yaml` 中的全局指标为 finite。
4. 断言 `pred_samples.npz` 中 `y_hat / logvar` 也保持 finite。

数据流：
synthetic features.npz / labels.npz
    ↓
evaluate.main()
    ↓
输入 X 非有限值清洗
    ↓
metrics.yaml / pred_samples.npz

依赖模块：
- numpy
- torch
- yaml
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
- uwnav_dynamics.eval.evaluate
- uwnav_dynamics.models.nets.s1_predictor
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
                    "name": "eval_input_nan_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "cpu_smoke",
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
                    "pred_len": 3,
                    "rnn_hidden": 8,
                    "rnn_layers": 1,
                    "dropout": 0.0,
                    "u_in_idx": list(range(8)),
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


def test_evaluate_main_sanitizes_nonfinite_input_features(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n, hist_len, pred_len, din, dout = 4, 4, 3, 25, 9
    x = np.zeros((n, hist_len, din), dtype=np.float32)
    y = np.zeros((n, pred_len, dout), dtype=np.float32)

    # 模拟稀疏 power 输入在 dataset artifact 中保留 NaN。
    x[..., -1] = np.nan
    x[0, 0, -2] = np.nan

    np.savez_compressed(data_dir / "features.npz", X=x)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=y,
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

    run_dir = tmp_path / "out" / "cpu_smoke"
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
    save_scaler(scalers_dir / "x_scaler.npz", {"mean": np.zeros(din, dtype=np.float32), "std": np.ones(din, dtype=np.float32)})
    save_scaler(scalers_dir / "y_scaler.npz", {"mean": np.zeros(dout, dtype=np.float32), "std": np.ones(dout, dtype=np.float32)})

    train_yaml = _write_train_yaml(tmp_path, data_dir)
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
    assert metrics["mae_global"] == pytest.approx(0.0)

    with np.load(tmp_path / "eval_test" / "pred_samples.npz", allow_pickle=False) as pred_npz:
        assert np.isfinite(pred_npz["y_hat"]).all()
        assert np.isfinite(pred_npz["y_true"]).all()
        assert np.isfinite(pred_npz["logvar"]).all()
