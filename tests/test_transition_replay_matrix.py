"""
模块名称：状态求解器 replay matrix 测试

模块职责：
验证“多候选模型 -> 统一 replay 协议 -> summary/ranking 汇总”最小链路可运行，
并确认标准排行表能把更优方案排在前面。

主要功能：
1. 构造一个 perfect 数据集和一个 biased 数据集。
2. 为两组数据生成独立 train yaml、split/scaler 与零权重 checkpoint。
3. 调用 `cli.transition_replay_matrix`，检查 `summary.csv` 与 `ranking.csv` 的排序结果。

数据流：
synthetic datasets + replay matrix yaml
    ↓
transition_replay_matrix.main()
    ↓
summary.csv / ranking.csv / per-run replay artifact
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from uwnav_dynamics.cli import transition_replay_matrix
from uwnav_dynamics.dataset.normalize import save_scaler
from uwnav_dynamics.dataset.split import save_split_indices
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.train.config import load_train_config


def _write_train_yaml(tmp_path: Path, *, name: str, data_dir: Path) -> Path:
    train_yaml = tmp_path / f"{name}.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": f"{name}_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": name,
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
                "train": {
                    "epochs": 1,
                    "eval_every": 1,
                    "save_best": True,
                    "save_last": True,
                    "metric": "val_transition_score",
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return train_yaml


def _build_dataset(
    data_dir: Path,
    *,
    target_value: float,
    obs_value: float | None = None,
    with_obs_mask: bool = False,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 12
    hist_len = 4
    input_cols = [f"f{i}" for i in range(25)]
    target_cols = [f"s{i}" for i in range(9)]

    base = np.zeros((n_rows, 25), dtype=np.float32)
    base[:, 8:17] = 5.0
    base_df = {col: base[:, i] for i, col in enumerate(input_cols)}
    if with_obs_mask:
        obs_val = float(target_value if obs_value is None else obs_value)
        base_df["vel_obs_x"] = np.full(n_rows, obs_val, dtype=np.float32)
        base_df["vel_obs_y"] = np.full(n_rows, obs_val, dtype=np.float32)
        base_df["vel_obs_z"] = np.full(n_rows, obs_val, dtype=np.float32)
        mask = np.zeros(n_rows, dtype=np.float32)
        mask[hist_len - 1 :] = 1.0
        base_df["dvl_mask"] = mask

    import pandas as pd

    pd.DataFrame(base_df).assign(**{col: target_value for col in target_cols}).to_csv(
        data_dir.parent / f"{data_dir.name}_base.csv",
        index=False,
    )
    base_csv = data_dir.parent / f"{data_dir.name}_base.csv"

    idx0 = np.arange(0, n_rows - hist_len, dtype=np.int64)
    X = np.zeros((idx0.size, hist_len, 25), dtype=np.float32)
    X[:, :, 8:17] = 5.0
    Y = np.full((idx0.size, 1, 9), target_value, dtype=np.float32)
    np.savez_compressed(data_dir / "features.npz", X=X, idx0=idx0)
    np.savez_compressed(data_dir / "labels.npz", Y=Y, target_cols=np.asarray(target_cols, dtype=object))
    meta = {
        "name": f"synthetic_{data_dir.name}",
        "base_csv": str(base_csv),
        "hist_len": hist_len,
        "pred_len": 1,
        "input_cols": input_cols,
        "target_cols": target_cols,
    }
    (data_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _prepare_run_artifacts(train_yaml: Path) -> Path:
    cfg_train = load_train_config(train_yaml)
    run_dir = Path(cfg_train.run.out_dir) / cfg_train.run.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg_train.data.data_dir)
    with np.load(data_dir / "features.npz", allow_pickle=True) as feat_npz:
        idx0 = np.asarray(feat_npz["idx0"], dtype=np.int64)

    save_split_indices(
        run_dir / "split_indices.npz",
        {
            "train": np.asarray([], dtype=np.int64),
            "val": np.asarray([], dtype=np.int64),
            "test": np.asarray(list(range(idx0.size)), dtype=np.int64),
        },
    )
    save_scaler(
        run_dir / "scalers" / "x_scaler.npz",
        {"mean": np.zeros(25, dtype=np.float32), "std": np.ones(25, dtype=np.float32)},
    )
    save_scaler(
        run_dir / "scalers" / "y_scaler.npz",
        {"mean": np.zeros(9, dtype=np.float32), "std": np.ones(9, dtype=np.float32)},
    )

    model = S1Predictor(cfg_train.model)
    for param in model.parameters():
        param.data.zero_()
    ckpt = run_dir / "best.pth"
    torch.save({"model": model.state_dict()}, ckpt)
    return ckpt


def test_transition_replay_matrix_writes_summary_and_ranking(tmp_path, monkeypatch):
    perfect_data = tmp_path / "perfect_data"
    biased_data = tmp_path / "biased_data"
    _build_dataset(perfect_data, target_value=5.0)
    _build_dataset(biased_data, target_value=8.0)

    perfect_yaml = _write_train_yaml(tmp_path, name="perfect_run", data_dir=perfect_data)
    biased_yaml = _write_train_yaml(tmp_path, name="biased_run", data_dir=biased_data)
    perfect_ckpt = _prepare_run_artifacts(perfect_yaml)
    biased_ckpt = _prepare_run_artifacts(biased_yaml)

    cfg_path = tmp_path / "replay_matrix.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": str(tmp_path / "replay_matrix_out"),
                    "split": "test",
                    "device": "cpu",
                    "min_seconds": 3,
                    "max_seconds_per_segment": 8,
                    "dt_s": 1.0,
                    "save_samples": 2,
                    "fail_fast": False,
                },
                "runs": [
                    {
                        "name": "perfect",
                        "label": "Perfect",
                        "role": "primary",
                        "train_yaml": str(perfect_yaml),
                        "ckpt": str(perfect_ckpt),
                    },
                    {
                        "name": "biased",
                        "label": "Biased",
                        "role": "ablation",
                        "train_yaml": str(biased_yaml),
                        "ckpt": str(biased_ckpt),
                    },
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.transition_replay_matrix",
            "-c",
            str(cfg_path),
        ],
    )
    assert transition_replay_matrix.main() == 0

    summary_path = tmp_path / "replay_matrix_out" / "summary.csv"
    ranking_path = tmp_path / "replay_matrix_out" / "ranking.csv"
    assert summary_path.exists()
    assert ranking_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"perfect", "biased"}
    assert {row["status"] for row in rows} == {"ok"}

    by_name = {row["name"]: row for row in rows}
    assert float(by_name["perfect"]["rmse_global"]) == 0.0
    assert float(by_name["biased"]["rmse_global"]) > 0.0
    assert float(by_name["perfect"]["rmse_threshold_failure_rate"]) == 0.0
    assert float(by_name["biased"]["rmse_threshold_failure_rate"]) >= 0.0

    with ranking_path.open("r", encoding="utf-8", newline="") as f:
        ranking_rows = list(csv.DictReader(f))
    assert len(ranking_rows) == 2
    assert ranking_rows[0]["name"] == "perfect"
    assert ranking_rows[0]["overall_rank"] == "1"
    assert ranking_rows[1]["name"] == "biased"
    assert float(ranking_rows[0]["overall_rank_score"]) < float(ranking_rows[1]["overall_rank_score"])
    assert (tmp_path / "replay_matrix_out" / "compare_test" / "replay_model_compare.png").exists()
    assert (tmp_path / "replay_matrix_out" / "compare_test" / "replay_long_horizon_curves.png").exists()
    metrics = yaml.safe_load(
        (tmp_path / "replay_matrix_out" / "runs" / "perfect" / "metrics.yaml").read_text(encoding="utf-8")
    )
    assert metrics["cfg"]["min_seconds"] == 3.0
    assert metrics["cfg"]["min_steps"] == 3
    assert metrics["cfg"]["max_seconds_per_segment"] == 8.0
    assert metrics["cfg"]["max_steps_per_segment"] == 8


def test_transition_replay_matrix_supports_feature_row_eval_against_common_observation(tmp_path, monkeypatch):
    perfect_data = tmp_path / "perfect_obs_data"
    biased_data = tmp_path / "biased_obs_data"
    _build_dataset(perfect_data, target_value=1.0, obs_value=5.0, with_obs_mask=True)
    _build_dataset(biased_data, target_value=1.0, obs_value=8.0, with_obs_mask=True)

    perfect_yaml = _write_train_yaml(tmp_path, name="perfect_obs_run", data_dir=perfect_data)
    biased_yaml = _write_train_yaml(tmp_path, name="biased_obs_run", data_dir=biased_data)
    perfect_ckpt = _prepare_run_artifacts(perfect_yaml)
    biased_ckpt = _prepare_run_artifacts(biased_yaml)

    cfg_path = tmp_path / "replay_matrix_obs.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": str(tmp_path / "replay_matrix_obs_out"),
                    "split": "test",
                    "device": "cpu",
                    "min_steps": 3,
                    "save_samples": 2,
                },
                "runs": [
                    {
                        "name": "perfect_obs",
                        "label": "Perfect Obs",
                        "role": "primary",
                        "train_yaml": str(perfect_yaml),
                        "ckpt": str(perfect_ckpt),
                        "eval": {
                            "name": "observed_velocity",
                            "pred_source": "feature_row",
                            "pred_cols": ["f8", "f9", "f10"],
                            "target_cols": ["vel_obs_x", "vel_obs_y", "vel_obs_z"],
                            "mask_cols": ["dvl_mask"],
                        },
                    },
                    {
                        "name": "biased_obs",
                        "label": "Biased Obs",
                        "role": "ablation",
                        "train_yaml": str(biased_yaml),
                        "ckpt": str(biased_ckpt),
                        "eval": {
                            "name": "observed_velocity",
                            "pred_source": "feature_row",
                            "pred_cols": ["f8", "f9", "f10"],
                            "target_cols": ["vel_obs_x", "vel_obs_y", "vel_obs_z"],
                            "mask_cols": ["dvl_mask"],
                        },
                    },
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.transition_replay_matrix",
            "-c",
            str(cfg_path),
        ],
    )
    assert transition_replay_matrix.main() == 0

    summary_path = tmp_path / "replay_matrix_obs_out" / "summary.csv"
    ranking_path = tmp_path / "replay_matrix_obs_out" / "ranking.csv"
    assert summary_path.exists()
    assert ranking_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows_by_name = {row["name"]: row for row in rows}
    assert float(rows_by_name["perfect_obs"]["rmse_global"]) == 0.0
    assert float(rows_by_name["biased_obs"]["rmse_global"]) > 0.0

    with ranking_path.open("r", encoding="utf-8", newline="") as f:
        ranked_rows = list(csv.DictReader(f))
    assert ranked_rows[0]["name"] == "perfect_obs"
    assert ranked_rows[0]["overall_rank"] == "1"
    assert ranked_rows[1]["name"] == "biased_obs"
    assert (tmp_path / "replay_matrix_obs_out" / "compare_test" / "replay_model_compare.png").exists()


def test_transition_replay_matrix_accepts_paths_relative_to_config_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "nested" / "perfect_cfg_data"
    _build_dataset(data_dir, target_value=5.0)

    train_yaml = _write_train_yaml(tmp_path, name="perfect_cfg_run", data_dir=data_dir)
    ckpt = _prepare_run_artifacts(train_yaml)

    generated_dir = tmp_path / "server_out" / "generated_replay_matrix"
    generated_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = generated_dir / "replay_from_summary.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": "../../replay_matrix_cfg_out",
                    "split": "test",
                    "device": "cpu",
                    "min_steps": 3,
                    "save_samples": 2,
                    "fail_fast": False,
                },
                "runs": [
                    {
                        "name": "perfect_cfg",
                        "label": "Perfect Cfg",
                        "role": "primary",
                        "train_yaml": "../../perfect_cfg_run.yaml",
                        "ckpt": "../../out/perfect_cfg_run/best.pth",
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.transition_replay_matrix",
            "-c",
            str(cfg_path),
        ],
    )
    assert transition_replay_matrix.main() == 0

    summary_path = tmp_path / "replay_matrix_cfg_out" / "summary.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["name"] == "perfect_cfg"
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["rmse_global"]) == 0.0
