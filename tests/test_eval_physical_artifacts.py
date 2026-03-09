"""
模块名称：物理量纲评估产物测试

模块职责：
验证评估主程序已经把物理量纲指标与样例产物作为主输出落盘，
并并行保留 z-score 空间的辅助 artifact 与控制前诊断摘要，
方便调参与审查同时进行。

主要功能：
1. 构造非单位 `y_scaler` 的最小评估场景。
2. 验证 `metrics.yaml` 中主指标使用物理量纲，且并行记录 zspace 指标。
3. 验证 `control_readiness` 摘要与主评估结果保持一致。
4. 验证 `pred_samples.npz` 为物理量纲，`pred_samples_zspace.npz` 为标准化空间。
5. 验证 `pred_context.npz` 与 `component_metrics*.csv` 一并落盘。

数据流：
synthetic dataset + non-identity scaler + zero checkpoint
    ↓
evaluate.main()
    ↓
metrics.yaml / pred_samples*.npz / component_metrics*.csv
    ↓
physical-vs-zspace assertions

依赖模块：
- numpy
- pytest
- torch
- yaml
- uwnav_dynamics.eval.evaluate
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
                    "name": "physical_eval_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B6_physical_eval",
                },
                "data": {
                    "data_dir": str(data_dir),
                    "batch_size": 1,
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


def test_evaluate_writes_physical_primary_artifacts_and_parallel_zspace_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    X = np.zeros((2, 4, 25), dtype=np.float32)
    X[:, -1, 8:17] = 0.0
    Y = np.full((2, 2, 9), 12.0, dtype=np.float32)
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

    run_dir = tmp_path / "out" / "B6_physical_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(
        run_dir / "split_indices.npz",
        {
            "train": np.asarray([0], dtype=np.int64),
            "val": np.asarray([], dtype=np.int64),
            "test": np.asarray([1], dtype=np.int64),
        },
    )
    save_scaler(run_dir / "scalers" / "x_scaler.npz", {"mean": np.zeros(25, dtype=np.float32), "std": np.ones(25, dtype=np.float32)})
    save_scaler(
        run_dir / "scalers" / "y_scaler.npz",
        {
            "mean": np.full(9, 10.0, dtype=np.float32),
            "std": np.full(9, 2.0, dtype=np.float32),
        },
    )

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
            "1",
            "--out_dir",
            str(out_dir),
            "--save_samples",
            "1",
        ],
    )

    assert evaluate.main() == 0

    metrics = yaml.safe_load((out_dir / "metrics.yaml").read_text(encoding="utf-8"))
    assert metrics["metric_space"]["primary"] == "physical"
    assert metrics["metric_space"]["auxiliary"] == "zspace"
    assert metrics["rmse_global"] == pytest.approx(2.0)
    assert metrics["mae_global"] == pytest.approx(2.0)
    assert metrics["rmse_global_zspace"] == pytest.approx(1.0)
    assert metrics["mae_global_zspace"] == pytest.approx(1.0)
    assert metrics["control_readiness"]["schema_version"] == "control_readiness_v1"
    assert metrics["control_readiness"]["intended_use"] == "offline_screening_for_control"
    assert metrics["control_readiness"]["closed_loop_proof"] is False
    dense_diag = metrics["control_readiness"]["physical"]["dense"]
    assert dense_diag["final_step"]["rmse_global"] == pytest.approx(2.0)
    assert dense_diag["final_step"]["mae_global"] == pytest.approx(2.0)
    assert dense_diag["tail_error"]["abs_p95_global"] == pytest.approx(2.0)
    assert dense_diag["tail_error"]["abs_p99_global"] == pytest.approx(2.0)
    assert dense_diag["tail_error"]["final_step_abs_p95_global"] == pytest.approx(2.0)
    assert dense_diag["rollout_growth"]["rmse_last_over_first"] == pytest.approx(1.0)
    assert dense_diag["rollout_growth"]["mae_last_over_first"] == pytest.approx(1.0)
    assert dense_diag["bias"]["worst_component"] == "acc_x"
    assert dense_diag["bias"]["worst_abs_bias"] == pytest.approx(2.0)
    assert dense_diag["final_step"]["group_rmse"]["acc"] == pytest.approx(2.0)
    assert dense_diag["final_step"]["group_rmse"]["gyro"] == pytest.approx(2.0)
    assert dense_diag["final_step"]["group_rmse"]["vel"] == pytest.approx(2.0)

    with np.load(out_dir / "pred_samples.npz", allow_pickle=False) as pred_npz:
        assert np.allclose(pred_npz["y_hat"], 10.0)
        assert np.allclose(pred_npz["y_true"], 12.0)

    with np.load(out_dir / "pred_samples_zspace.npz", allow_pickle=False) as pred_npz_z:
        assert np.allclose(pred_npz_z["y_hat"], 0.0)
        assert np.allclose(pred_npz_z["y_true"], 1.0)

    with np.load(out_dir / "pred_context.npz", allow_pickle=False) as ctx_npz:
        assert ctx_npz["target_mask"].shape == (1, 2, 9)
        assert ctx_npz["sample_index"].tolist() == [1]

    assert (out_dir / "component_metrics.csv").exists()
    assert (out_dir / "component_metrics_masked.csv").exists()
    assert (out_dir / "component_metrics_zspace.csv").exists()
    assert (out_dir / "component_metrics_masked_zspace.csv").exists()
