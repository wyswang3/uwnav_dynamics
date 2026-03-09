"""
模块名称：多 GPU 实验矩阵调度测试

模块职责：
验证 `cli.train_matrix` 的纯配置与规划逻辑，
确保基础 train yaml、公共 override 与局部 override 的合并结果稳定可追溯。

主要功能：
1. 检查 matrix yaml 解析后的 launcher 配置字段是否正确。
2. 检查生成的 train yaml 会自动补齐 `run.name / run.variant / run.out_dir`。
3. 检查不同变体的 `run_dir` 保持唯一，便于后续并发训练与评估。

数据流：
tmp matrix yaml + tmp base train yaml
    ↓
load_matrix_launcher_config()
    ↓
prepare_matrix_runs()
    ↓
generated train yamls / run_dir assertions

依赖模块：
- yaml
- uwnav_dynamics.cli.train_matrix

备注：
- 本测试只覆盖纯规划逻辑，不实际启动训练子进程。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from uwnav_dynamics.cli.train_matrix import (
    load_matrix_launcher_config,
    prepare_matrix_runs,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def test_prepare_matrix_runs_materializes_unique_generated_train_yamls(tmp_path: Path) -> None:
    base_train_yaml = tmp_path / "configs" / "train" / "base.yaml"
    _write_yaml(
        base_train_yaml,
        {
            "run": {
                "name": "base",
                "seed": 0,
                "device": "cpu",
                "amp": False,
                "out_dir": "out/ckpts/base",
                "variant": "base_variant",
            },
            "data": {
                "data_dir": "data/processed/demo",
                "batch_size": 64,
                "num_workers": 0,
                "pin_memory": False,
                "split": {"train_ratio": 0.7, "val_ratio": 0.15},
            },
            "model": {
                "name": "s1_predictor",
                "din": 25,
                "dout": 9,
                "pred_len": 10,
                "rnn_hidden": 64,
                "rnn_layers": 1,
                "dropout": 0.0,
                "u_in_idx": [0, 1, 2, 3, 4, 5, 6, 7],
                "y_in_idx": [8, 9, 10, 11, 12, 13, 14, 15, 16],
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
                        "hidden_dim": 64,
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
                        "feat_dim": 128,
                        "hidden": 64,
                        "logvar_min": -10.0,
                        "logvar_max": 6.0,
                    },
                },
            },
            "rollout": {"y0_source": "x_last_state", "mode": "delta_cumsum"},
            "loss": {"type": "nll_diag", "logvar_clip": [-10.0, 6.0]},
            "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0},
            "train": {"epochs": 3, "eval_every": 1, "save_best": True, "save_last": True, "metric": "val_loss"},
        },
    )

    matrix_yaml = tmp_path / "configs" / "launch" / "matrix.yaml"
    _write_yaml(
        matrix_yaml,
        {
            "base_train_yaml": "configs/train/base.yaml",
            "launcher": {
                "work_dir": "out/matrix/demo",
                "gpus": [0, 1],
                "max_parallel": 2,
                "run_eval": True,
                "eval_split": "test",
                "compare": True,
                "compare_metrics": ["rmse", "mae"],
            },
            "common_overrides": {
                "run": {"device": "cuda", "amp": True, "out_dir": "out/ckpts/matrix"},
                "data": {"batch_size": 512, "num_workers": 4, "pin_memory": True},
                "train": {"epochs": 50},
            },
            "runs": [
                {
                    "name": "b0_seed0",
                    "label": "B0 seed0",
                    "role": "baseline",
                    "overrides": {"run": {"seed": 0}},
                },
                {
                    "name": "b1_thruster",
                    "label": "B1 thruster",
                    "role": "ablation",
                    "overrides": {
                        "run": {"seed": 1, "variant": "B1_thruster_seed1"},
                        "model": {"blocks": {"thruster_lag": {"enabled": True}}},
                    },
                },
            ],
        },
    )

    cfg = load_matrix_launcher_config(matrix_yaml)
    prepared = prepare_matrix_runs(cfg, repo_root=tmp_path)

    assert len(prepared) == 2
    assert prepared[0].yaml_path.exists()
    assert prepared[1].yaml_path.exists()
    assert prepared[0].yaml_path.parent == tmp_path / "configs" / "train" / "generated" / "demo"
    assert prepared[1].yaml_path.parent == tmp_path / "configs" / "train" / "generated" / "demo"
    assert prepared[0].run_dir != prepared[1].run_dir
    assert prepared[0].eval_dir.name == "eval_test"

    generated0 = yaml.safe_load(prepared[0].yaml_path.read_text(encoding="utf-8"))
    generated1 = yaml.safe_load(prepared[1].yaml_path.read_text(encoding="utf-8"))

    assert generated0["run"]["device"] == "cuda"
    assert generated0["run"]["amp"] is True
    assert generated0["run"]["variant"] == "b0_seed0"
    assert generated0["data"]["batch_size"] == 512
    assert generated0["train"]["epochs"] == 50

    assert generated1["run"]["variant"] == "B1_thruster_seed1"
    assert generated1["model"]["blocks"]["thruster_lag"]["enabled"] is True
