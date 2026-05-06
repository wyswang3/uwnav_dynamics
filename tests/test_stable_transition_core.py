"""
模块名称：稳定状态转移内核测试

模块职责：
验证 `stable_transition_core` 新模型 family 的最小训练接口契约，
确保它不依赖旧 optional blocks，并能通过模型工厂接入主运行链路。

主要功能：
1. 验证 canonical parser 接受 `model.name=stable_transition_core` 且不要求旧 blocks。
2. 验证模型工厂能构造 `StableTransitionCore`。
3. 验证 forward / forward_with_aux 输出 shape、有限性与稳定 key 契约。

数据流：
stable train yaml dict
    ↓
build_from_dict()
    ↓
build_state_predictor()
    ↓
dY / logvar / aux

依赖模块：
- torch
- uwnav_dynamics.train.config
- uwnav_dynamics.models.nets.factory
- uwnav_dynamics.models.nets.stable_transition_core

备注：
- 本测试只验证模块接口和基础数值安全，不代表长期 replay 已经优于当前默认模型。
"""

from __future__ import annotations

import torch

from uwnav_dynamics.models.nets.factory import build_state_predictor
from uwnav_dynamics.models.nets.stable_transition_core import StableTransitionCore
from uwnav_dynamics.train.config import build_from_dict


def _stable_train_yaml_dict(core_type: str = "stable_diag_damp") -> dict:
    return {
        "run": {
            "name": "stable_core_demo",
            "seed": 11,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/stable_core_demo",
            "variant": f"{core_type}_s11",
        },
        "data": {
            "data_dir": "data/processed/demo",
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
        },
        "model": {
            "name": "stable_transition_core",
            "din": 34,
            "dout": 9,
            "pred_len": 3,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "dropout": 0.0,
            "u_in_idx": list(range(0, 8)),
            "y_in_idx": list(range(8, 17)),
            "core_type": core_type,
            "core_hidden": 24,
            "residual_bound": 0.2,
            "damping_min": 1.0e-4,
            "damping_max": 10.0,
            "control_bound": 2.0,
            "dt": 0.01,
        },
        "rollout": {
            "y0_source": "x_last_state",
            "mode": "delta_cumsum",
        },
        "loss": {
            "type": "transition_balance",
            "logvar_clip": [-10.0, 6.0],
            "state_huber_weight": 1.0,
            "state_huber_delta": 1.0,
            "delta_huber_weight": 0.5,
            "delta_huber_delta": 1.0,
            "acc_weight": 3.0,
            "gyro_weight": 2.0,
            "vel_weight": 2.0,
        },
        "optim": {
            "name": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
        },
        "train": {
            "epochs": 2,
            "eval_every": 1,
            "save_best": True,
            "save_last": True,
            "metric": "val_transition_score",
        },
    }


def test_stable_transition_core_parser_accepts_config_without_legacy_blocks():
    cfg = build_from_dict(_stable_train_yaml_dict("stable_diag_damp"))

    assert cfg.model.name == "stable_transition_core"
    assert cfg.model.core_type == "stable_diag_damp"
    assert cfg.model.core_hidden == 24
    assert cfg.model.blocks.thruster_lag.enabled is False
    assert cfg.model.blocks.hydro_ssm.enabled is False
    assert cfg.model.blocks.damping.enabled is False
    assert cfg.model.blocks.uncertainty.enabled is False


def test_stable_transition_core_factory_and_forward_contract():
    cfg = build_from_dict(_stable_train_yaml_dict("implicit_euler"))
    model = build_state_predictor(cfg.model).eval()

    assert isinstance(model, StableTransitionCore)

    x = torch.randn(4, 12, cfg.model.din)
    with torch.no_grad():
        dY, logvar = model(x)
        dY_aux, logvar_aux, aux = model.forward_with_aux(x)

    assert dY.shape == (4, cfg.model.pred_len, cfg.model.dout)
    assert logvar.shape == (4, cfg.model.pred_len, cfg.model.dout)
    assert torch.allclose(dY, dY_aux)
    assert torch.allclose(logvar, logvar_aux)
    assert torch.isfinite(dY).all()
    assert torch.isfinite(logvar).all()
    assert "dvl_obs" in aux
    assert aux["dvl_obs"] is None


def test_stable_transition_core_supports_first_round_core_types():
    for core_type in ("stable_diag_damp", "implicit_euler", "control_affine", "residual_budget", "energy_budget"):
        cfg = build_from_dict(_stable_train_yaml_dict(core_type))
        model = build_state_predictor(cfg).eval()
        x = torch.randn(2, 8, cfg.model.din)
        with torch.no_grad():
            dY, logvar = model(x)
        assert dY.shape == (2, cfg.model.pred_len, cfg.model.dout)
        assert logvar.shape == dY.shape
        assert torch.isfinite(dY).all()
        assert torch.isfinite(logvar).all()
