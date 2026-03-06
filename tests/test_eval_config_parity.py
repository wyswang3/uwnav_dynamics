"""
模块名称：评估配置真源与职责边界测试

模块职责：
验证评估侧继续复用训练侧 canonical parser，
并确保 EvalConfig 只承载数值评估运行时字段，不重新接管绘图编排职责。

主要功能：
1. 检查 build_eval_config 复用训练配置解释结果。
2. 检查相同权重下 train / eval 模型前向结果一致。
3. 显式验证 plot runtime fields 已从 EvalConfig / build_eval_config 移出。

数据流：
临时 train yaml + run artifact
    ↓
build_eval_config()
    ↓
EvalConfig / S1PredictorConfig
    ↓
配置真源与职责边界断言

依赖模块：
- uwnav_dynamics.eval.config
- uwnav_dynamics.train.config
- uwnav_dynamics.experiment.layout

备注：
- 该测试覆盖 PR1 与 PR3 的配置契约。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import yaml

from uwnav_dynamics.dataset.normalize import save_scaler
from uwnav_dynamics.dataset.split import save_split_indices
from uwnav_dynamics.eval.config import build_eval_config
from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.train.config import load_train_config


def _train_yaml_dict(tmp_path: Path) -> dict:
    return {
        "run": {
            "name": "parity_demo",
            "seed": 7,
            "device": "cpu",
            "amp": False,
            "out_dir": str(tmp_path / "out"),
            "variant": "B3_parity",
        },
        "data": {
            "data_dir": str(tmp_path / "data"),
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
        },
        "model": {
            "name": "s1_predictor",
            "din": 25,
            "dout": 9,
            "pred_len": 4,
            "rnn_hidden": 32,
            "rnn_layers": 2,
            "dropout": 0.0,
            "u_in_idx": list(range(0, 8)),
            "y_in_idx": list(range(8, 17)),
            "use_thruster_as_replacement": True,
            "use_hydro_feat": True,
            "blocks": {
                "thruster_lag": {
                    "enabled": True,
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
                    "enabled": True,
                    "hidden_dim": 12,
                    "lambda_init": 0.85,
                    "act": "tanh",
                    "per_dim_lambda": True,
                },
                "damping": {
                    "enabled": True,
                    "v_start": 6,
                    "v_dim": 3,
                    "mode": "simple",
                    "d_init": 0.5,
                    "mlp_hidden": 16,
                },
                "uncertainty": {
                    "enabled": True,
                    "feat_dim": 20,
                    "hidden": 10,
                    "logvar_min": -10.0,
                    "logvar_max": 6.0,
                },
            },
        },
        "rollout": {
            "y0_source": "x_last_state",
            "mode": "delta_cumsum",
        },
        "loss": {
            "type": "nll_diag",
            "logvar_clip": [-10.0, 6.0],
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
        },
        "train": {
            "epochs": 2,
            "eval_every": 1,
            "save_best": True,
            "save_last": True,
            "metric": "val_loss",
        },
    }


def _prepare_eval_artifacts(layout: RunLayout, *, din: int, dout: int) -> None:
    save_split_indices(
        layout.split_indices_path,
        {
            "train": np.asarray([0, 1], dtype=np.int64),
            "val": np.asarray([2], dtype=np.int64),
            "test": np.asarray([3], dtype=np.int64),
        },
    )
    save_scaler(
        layout.x_scaler_path,
        {
            "mean": np.zeros(din, dtype=np.float32),
            "std": np.ones(din, dtype=np.float32),
        },
    )
    save_scaler(
        layout.y_scaler_path,
        {
            "mean": np.zeros(dout, dtype=np.float32),
            "std": np.ones(dout, dtype=np.float32),
        },
    )


def test_eval_config_reuses_canonical_train_parser(tmp_path):
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(_train_yaml_dict(tmp_path), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    cfg_train = load_train_config(train_yaml)
    layout = RunLayout(out_dir=cfg_train.run.out_dir, variant=cfg_train.run.variant)
    _prepare_eval_artifacts(layout, din=cfg_train.model.din, dout=cfg_train.model.dout)

    eval_out_dir = tmp_path / "custom_eval_out"
    cfg_eval, cfg_model = build_eval_config(
        train_yaml=train_yaml,
        ckpt=tmp_path / "best.pth",
        split="test",
        device="cpu",
        batch_size=8,
        out_dir=eval_out_dir,
        save_samples=12,
    )

    # 评估侧只允许覆盖 runtime 参数；模型拓扑必须直接复用训练侧 canonical parser。
    assert cfg_model == cfg_train.model
    assert cfg_eval.data_dir == cfg_train.data.data_dir
    assert cfg_eval.device == "cpu"
    assert cfg_eval.batch_size == 8
    assert cfg_eval.out_dir == eval_out_dir
    assert cfg_eval.split_indices_path == layout.split_indices_path
    assert cfg_eval.x_scaler_path == layout.x_scaler_path
    assert cfg_eval.y_scaler_path == layout.y_scaler_path
    assert cfg_eval.y0_source == cfg_train.rollout.y0_source
    assert cfg_eval.mode == cfg_train.rollout.mode

    assert "make_plots" not in cfg_eval.__dataclass_fields__
    assert "plot_fmt" not in cfg_eval.__dataclass_fields__
    assert "dt_s" not in cfg_eval.__dataclass_fields__
    assert "x_axis" not in cfg_eval.__dataclass_fields__
    assert "n_plot_samples" not in cfg_eval.__dataclass_fields__

    sig = inspect.signature(build_eval_config)
    assert "make_plots" not in sig.parameters
    assert "plot_fmt" not in sig.parameters
    assert "dt_s" not in sig.parameters
    assert "x_axis" not in sig.parameters
    assert "n_plot_samples" not in sig.parameters


def test_eval_model_forward_matches_train_model_when_loading_same_weights(tmp_path):
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(_train_yaml_dict(tmp_path), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    cfg_train = load_train_config(train_yaml)
    layout = RunLayout(out_dir=cfg_train.run.out_dir, variant=cfg_train.run.variant)
    _prepare_eval_artifacts(layout, din=cfg_train.model.din, dout=cfg_train.model.dout)

    _cfg_eval, cfg_model = build_eval_config(
        train_yaml=train_yaml,
        ckpt=tmp_path / "best.pth",
        split="test",
        device=None,
        batch_size=None,
        out_dir=None,
        save_samples=8,
    )

    torch.manual_seed(0)
    model_train = S1Predictor(cfg_train.model).eval()
    state = model_train.state_dict()

    model_eval = S1Predictor(cfg_model).eval()
    model_eval.load_state_dict(state, strict=True)

    x = torch.randn(2, 6, cfg_train.model.din)
    with torch.no_grad():
        dY_train, logvar_train = model_train(x)
        dY_eval, logvar_eval = model_eval(x)

    assert torch.allclose(dY_train, dY_eval)
    assert torch.allclose(logvar_train, logvar_eval)
