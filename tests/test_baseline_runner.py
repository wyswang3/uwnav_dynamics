"""
模块名称：baseline runner 测试

模块职责：
验证 trivial 与 classical baseline 能在与主训练链路一致的
split/scaler/eval artifact 合同下完整跑通。

主要功能：
1. 构造最小 synthetic dataset 与 train yaml。
2. 运行 `run_baseline_suite()` 生成两个 baseline run。
3. 断言 summary.csv、resolved_baseline.yaml、baseline_state.npz 与 metrics.yaml 存在。

数据流：
synthetic features.npz / labels.npz + train yaml
    ↓
run_baseline_suite()
    ↓
baseline run dir / eval artifact / summary.csv

依赖模块：
- csv
- yaml
- uwnav_dynamics.baselines.runner
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from uwnav_dynamics.baselines.runner import run_baseline_suite


def _write_train_yaml(tmp_path: Path, data_dir: Path) -> Path:
    train_yaml = tmp_path / "train.yaml"
    payload = {
        "run": {
            "name": "baseline_demo",
            "seed": 0,
            "device": "cpu",
            "amp": False,
            "out_dir": str(tmp_path / "out"),
            "variant": "B0_demo",
        },
        "data": {
            "data_dir": str(data_dir),
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "split": {
                "train_ratio": 0.5,
                "val_ratio": 0.25,
            },
        },
        "model": {
            "name": "s1_predictor",
            "din": 25,
            "dout": 9,
            "pred_len": 2,
            "rnn_hidden": 16,
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
                    "hidden_dim": 16,
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
                    "mlp_hidden": 16,
                },
                "uncertainty": {
                    "enabled": False,
                    "feat_dim": 32,
                    "hidden": 16,
                    "logvar_min": -10.0,
                    "logvar_max": 6.0,
                },
            },
        },
        "rollout": {"y0_source": "x_last_state", "mode": "delta_cumsum"},
        "loss": {"type": "nll_diag", "logvar_clip": [-10.0, 6.0]},
        "optim": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 1.0e-4, "grad_clip": 1.0},
        "train": {"epochs": 2, "eval_every": 1, "save_best": True, "save_last": True, "metric": "val_loss"},
    }
    train_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return train_yaml


def test_run_baseline_suite_writes_summary_and_eval_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n, hist_len, pred_len, din, dout = 12, 4, 2, 25, 9
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, hist_len, din)).astype(np.float32)
    y_last = x[:, -1, 8:17]
    y = np.repeat(y_last[:, None, :], pred_len, axis=1).astype(np.float32)
    y += 0.05 * rng.normal(size=y.shape).astype(np.float32)
    dvl_mask = np.ones((n, pred_len, 1), dtype=bool)
    np.savez_compressed(data_dir / "features.npz", X=x)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=y,
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

    train_yaml = _write_train_yaml(tmp_path, data_dir)
    summary_path = run_baseline_suite(
        train_yaml=train_yaml,
        baselines=("trivial_last", "classical_ridge"),
        split="test",
        work_dir=tmp_path / "baseline_runs",
        alpha_grid=(1.0e-4, 1.0e-2),
        save_samples=8,
        compare=False,
        dt=0.01,
        x_axis="sec",
        plot_fmt="png",
    )

    assert summary_path.exists()
    rows = list(csv.DictReader(summary_path.read_text(encoding="utf-8").splitlines()))
    assert [row["kind"] for row in rows] == ["trivial_last", "classical_ridge"]
    assert rows[1]["selected_alpha"] != ""

    for row in rows:
        run_dir = summary_path.parent / row["run_dir"]
        eval_dir = summary_path.parent / row["eval_dir"]
        assert (run_dir / "resolved_baseline.yaml").exists()
        assert (run_dir / "baseline_state.npz").exists()
        assert (eval_dir / "metrics.yaml").exists()
        assert (eval_dir / "resolved_eval.yaml").exists()

        metrics = yaml.safe_load((eval_dir / "metrics.yaml").read_text(encoding="utf-8"))
        assert metrics["cfg"]["predictor"]["type"] == "baseline"
