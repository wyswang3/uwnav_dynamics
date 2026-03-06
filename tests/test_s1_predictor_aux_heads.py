"""
模块名称：S1 辅助观测头测试

模块职责：
验证 P0.1 在不破坏主 `forward()` 契约的前提下，
为 `S1Predictor` 新增的 `dvl_obs` 辅助头与 `forward_with_aux()` 接口。

主要功能：
1. 验证 `forward()` 仍只返回 `(dY, logvar)` 且 shape 不变。
2. 验证 `forward_with_aux()` 始终返回稳定 key `aux["dvl_obs"]`。
3. 验证 `dvl_obs` 输出维度来自 semantic output layout 的 velocity group。
4. 验证完整 train config 中 parser 已解析的辅助头配置能传到模型构造。

数据流：
train yaml dict / S1PredictorConfig
    ↓
S1Predictor(...)
    ↓
forward() / forward_with_aux()
    ↓
shape 与兼容性断言

依赖模块：
- torch
- uwnav_dynamics.train.config
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 本测试只覆盖 Patch B 的模型侧接线，不涉及训练 loss 或 eval artifact。
"""

from __future__ import annotations

import torch

from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout
from uwnav_dynamics.train.config import build_from_dict


def _base_train_yaml_dict() -> dict:
    return {
        "run": {
            "name": "p0_model_demo",
            "seed": 0,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/p0_model_demo",
            "variant": "B0",
        },
        "data": {
            "data_dir": "data/processed/demo",
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


def test_s1_predictor_forward_with_aux_returns_stable_none_key_when_disabled():
    cfg = S1PredictorConfig(din=25, dout=9, pred_len=4, rnn_hidden=32, rnn_layers=2, dropout=0.0)
    model = S1Predictor(cfg).eval()

    x = torch.randn(2, 6, cfg.din)
    with torch.no_grad():
        dY, logvar = model(x)
        dY_aux, logvar_aux, aux = model.forward_with_aux(x)

    assert dY.shape == (2, cfg.pred_len, cfg.dout)
    assert logvar.shape == (2, cfg.pred_len, cfg.dout)
    assert torch.allclose(dY, dY_aux)
    assert torch.allclose(logvar, logvar_aux)
    assert "dvl_obs" in aux
    assert aux["dvl_obs"] is None


def test_s1_predictor_forward_with_aux_uses_velocity_semantic_group_dim():
    raw = _base_train_yaml_dict()
    raw["model"]["aux_heads"] = {
        "dvl_obs": {
            "enabled": True,
            "hidden": 96,
        }
    }
    cfg = build_from_dict(raw)
    model = S1Predictor(cfg).eval()

    vel_indices = canonical_semantic_output_layout(cfg.model.dout).group_indices["vel"]
    x = torch.randn(3, 5, cfg.model.din)

    with torch.no_grad():
        dY, logvar = model(x)
        dY_aux, logvar_aux, aux = model.forward_with_aux(x)

    assert torch.allclose(dY, dY_aux)
    assert torch.allclose(logvar, logvar_aux)
    assert aux["dvl_obs"] is not None
    assert aux["dvl_obs"].shape == (3, cfg.model.pred_len, len(vel_indices))
    assert model.cfg.aux_heads.dvl_obs.enabled is True
    assert model.cfg.aux_heads.dvl_obs.hidden == 96


def test_s1_predictor_cfg_only_path_keeps_aux_disabled_by_default():
    raw = _base_train_yaml_dict()
    raw["model"]["aux_heads"] = {
        "dvl_obs": {
            "enabled": True,
            "hidden": 64,
        }
    }
    cfg = build_from_dict(raw)
    model = S1Predictor(cfg.model).eval()

    x = torch.randn(1, 4, cfg.model.din)
    with torch.no_grad():
        _dY, _logvar, aux = model.forward_with_aux(x)

    # Patch B 只接模型侧能力，不修改 run_train。
    # 因此直接传 cfg.model 时仍走 dataclass 默认值，保持旧路径兼容。
    assert model.cfg.aux_heads.dvl_obs.enabled is False
    assert aux["dvl_obs"] is None
