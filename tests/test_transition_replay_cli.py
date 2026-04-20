"""
模块名称：状态求解器 replay CLI 测试

模块职责：
验证“训练后模型 -> 经验型状态求解器 -> 长序列 autoregressive replay”最小链路可运行，
并确认当前升级已经补上长期递推验证入口。

主要功能：
1. 构造最小 constant-state 基表与 processed dataset 索引。
2. 使用零权重 `pred_len=1` 模型构造稳定的一步状态求解器。
3. 调用 `cli.transition_replay`，检查 replay artifact 与核心指标落盘。

数据流：
synthetic base_csv + meta/features + split/scaler + zero ckpt
    ↓
transition_replay.main()
    ↓
metrics.yaml / segment_metrics.csv / pred_samples.npz
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from uwnav_dynamics.dataset.normalize import save_scaler
from uwnav_dynamics.dataset.split import save_split_indices
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.cli import transition_replay


def _write_train_yaml(tmp_path: Path, data_dir: Path) -> Path:
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "transition_replay_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B0_replay",
                },
                "data": {
                    "data_dir": str(data_dir),
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "split": {"train_ratio": 0.7, "val_ratio": 0.15},
                },
                "model": {
                    "name": "s1_predictor",
                    "din": 25,
                    "dout": 9,
                    "pred_len": 1,
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
                "loss": {"type": "transition_balance", "logvar_clip": [-10.0, 6.0], "state_huber_weight": 1.0},
                "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0},
                "train": {"epochs": 1, "eval_every": 1, "save_best": True, "save_last": True, "metric": "val_transition_score"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return train_yaml


def test_transition_replay_cli_writes_autoregressive_replay_artifacts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 12
    hist_len = 4
    input_cols = [f"f{i}" for i in range(25)]
    target_cols = [f"s{i}" for i in range(9)]

    base = np.zeros((n_rows, 25), dtype=np.float32)
    base[:, 8:17] = 5.0
    base_df = {col: base[:, i] for i, col in enumerate(input_cols)}
    base_csv = tmp_path / "train_base.csv"
    import pandas as pd
    pd.DataFrame(base_df).assign(**{col: 5.0 for col in target_cols}).to_csv(base_csv, index=False)

    idx0 = np.arange(0, n_rows - hist_len, dtype=np.int64)
    X = np.zeros((idx0.size, hist_len, 25), dtype=np.float32)
    X[:, :, 8:17] = 5.0
    Y = np.full((idx0.size, 1, 9), 5.0, dtype=np.float32)
    np.savez_compressed(data_dir / "features.npz", X=X, idx0=idx0)
    np.savez_compressed(data_dir / "labels.npz", Y=Y, target_cols=np.asarray(target_cols, dtype=object))
    meta = {
        "name": "synthetic_replay",
        "base_csv": str(base_csv),
        "hist_len": hist_len,
        "pred_len": 1,
        "input_cols": input_cols,
        "target_cols": target_cols,
    }
    (data_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")

    run_dir = tmp_path / "out" / "B0_replay"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(
        run_dir / "split_indices.npz",
        {
            "train": np.asarray([], dtype=np.int64),
            "val": np.asarray([], dtype=np.int64),
            "test": np.asarray(list(range(idx0.size)), dtype=np.int64),
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

    out_dir = tmp_path / "replay_test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.transition_replay",
            "-y",
            str(train_yaml),
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
            "--device",
            "cpu",
            "--out_dir",
            str(out_dir),
            "--min_steps",
            "3",
            "--save_samples",
            "2",
        ],
    )

    assert transition_replay.main() == 0

    metrics = yaml.safe_load((out_dir / "metrics.yaml").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "transition_replay_v1"
    assert metrics["solver_step_semantics"] == "single_step"
    assert metrics["segment_count"] == 1
    assert metrics["total_steps"] == idx0.size
    assert metrics["rmse_global"] == 0.0
    assert metrics["mae_global"] == 0.0
    assert metrics["closed_loop_proof"] is False
    assert metrics["robustness"]["finite_pass_rate"] == 1.0
    assert metrics["segment_stats"]["n_steps_mean"] == idx0.size
    assert metrics["final_step"]["abs_p95_global"] == 0.0
    assert metrics["long_horizon"]["thresholds"]["rmse"] == 0.05
    assert metrics["long_horizon"]["time_to_threshold"]["rmse"]["failure_rate"] == 0.0
    assert not Path(str(metrics["cfg"]["train_yaml"])).is_absolute()
    assert not Path(str(metrics["cfg"]["ckpt"])).is_absolute()
    assert not Path(str(metrics["cfg"]["data_dir"])).is_absolute()

    resolved = yaml.safe_load((out_dir / "resolved_replay.yaml").read_text(encoding="utf-8"))
    assert not Path(str(resolved["cfg"]["split_indices_path"])).is_absolute()

    assert (out_dir / "segment_metrics.csv").exists()
    assert (out_dir / "component_metrics.csv").exists()
    assert (out_dir / "step_metrics.csv").exists()
    with np.load(out_dir / "pred_samples.npz", allow_pickle=False) as pred_npz:
        assert pred_npz["y_hat"].shape[0] == 1
        assert pred_npz["valid_mask"].all()
