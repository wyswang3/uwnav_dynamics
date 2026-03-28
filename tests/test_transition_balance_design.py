"""
模块名称：状态转移重设计测试

模块职责：
验证当前新增的 grouped state head 与 transition-balance loss
已经以最小方式接入训练配置和模型主路径。

主要功能：
1. 验证 grouped head 不改变 `forward()` 的基本 shape 契约。
2. 验证新训练配置可以解析 `head_mode` 与 transition-balance loss 字段。
3. 验证语义组加权会放大对应状态组的损失惩罚。

数据流：
synthetic config / dummy model / synthetic tensors
    ↓
train.config / run_train.build_loss_fn
    ↓
shape / parsing / weighted-loss assertions

依赖模块：
- torch
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.train.config
- uwnav_dynamics.train.run_train
"""

from __future__ import annotations

import pytest
import torch

from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.train.config import build_from_dict
from uwnav_dynamics.train.run_train import build_loss_fn, build_monitor_fn


class _DummyModel:
    def __init__(self, dY: torch.Tensor, logvar: torch.Tensor):
        self._dY = dY
        self._logvar = logvar

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dY.clone(), self._logvar.clone()


def _base_train_yaml_dict() -> dict:
    return {
        "run": {
            "name": "transition_balance_demo",
            "seed": 8,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/transition_balance_demo",
            "variant": "T1_demo",
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
            "head_mode": "grouped",
            "group_head_hidden": 48,
            "aux_heads": {
                "dvl_obs": {
                    "enabled": True,
                    "hidden": 64,
                }
            },
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
            "type": "transition_balance",
            "logvar_clip": [-10.0, 6.0],
            "state_huber_weight": 1.0,
            "state_huber_delta": 1.0,
            "delta_huber_weight": 0.5,
            "delta_huber_delta": 1.0,
            "logvar_reg_weight": 0.01,
            "tail_weight_power": 1.0,
            "acc_weight": 3.0,
            "gyro_weight": 2.0,
            "vel_weight": 1.5,
            "dvl_obs_weight": 0.2,
            "dvl_obs_delta": 1.0,
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
            "metric": "val_transition_score",
        },
    }


def test_grouped_head_keeps_main_forward_shape_contract() -> None:
    cfg = S1PredictorConfig(
        din=25,
        dout=9,
        pred_len=4,
        rnn_hidden=32,
        rnn_layers=2,
        dropout=0.0,
        head_mode="grouped",
        group_head_hidden=24,
    )
    model = S1Predictor(cfg).eval()
    x = torch.randn(3, 6, cfg.din)
    with torch.no_grad():
        dY, logvar = model(x)
    assert dY.shape == (3, cfg.pred_len, cfg.dout)
    assert logvar.shape == (3, cfg.pred_len, cfg.dout)


def test_transition_balance_config_parses_grouped_head_and_loss_fields() -> None:
    cfg = build_from_dict(_base_train_yaml_dict())
    assert cfg.model.head_mode == "grouped"
    assert cfg.model.group_head_hidden == 48
    assert cfg.loss.type == "transition_balance"
    assert cfg.loss.state_huber_weight == 1.0
    assert cfg.loss.delta_huber_weight == 0.5
    assert cfg.loss.acc_weight == 3.0
    assert cfg.loss.gyro_weight == 2.0
    assert cfg.loss.vel_weight == 1.5
    assert cfg.train.metric == "val_transition_score"


def test_transition_balance_acc_weight_increases_loss_for_acc_only_error() -> None:
    x = torch.zeros(2, 5, 25)
    y = torch.zeros(2, 4, 9)
    target_mask = torch.ones(2, 4, 9, dtype=torch.bool)
    dY = torch.zeros(2, 4, 9)
    dY[:, :, 0] = 2.0
    logvar = torch.zeros_like(dY)
    model = _DummyModel(dY=dY, logvar=logvar)

    low_weight_loss = build_loss_fn(
        -10.0,
        6.0,
        tuple(range(8, 17)),
        dout=9,
        loss_type="transition_balance",
        state_huber_weight=1.0,
        delta_huber_weight=0.0,
        logvar_reg_weight=0.0,
        tail_weight_power=0.0,
        acc_weight=1.0,
        gyro_weight=1.0,
        vel_weight=1.0,
    )(model, x, y, target_mask)

    high_weight_loss = build_loss_fn(
        -10.0,
        6.0,
        tuple(range(8, 17)),
        dout=9,
        loss_type="transition_balance",
        state_huber_weight=1.0,
        delta_huber_weight=0.0,
        logvar_reg_weight=0.0,
        tail_weight_power=0.0,
        acc_weight=4.0,
        gyro_weight=1.0,
        vel_weight=1.0,
    )(model, x, y, target_mask)

    assert float(high_weight_loss.item()) > float(low_weight_loss.item())


def test_transition_monitor_ignores_logvar_inflation() -> None:
    x = torch.zeros(2, 5, 25)
    y = torch.zeros(2, 4, 9)
    target_mask = torch.ones(2, 4, 9, dtype=torch.bool)
    dY = torch.zeros(2, 4, 9)
    dY[:, :, 0] = 1.5
    low_logvar = torch.zeros_like(dY)
    high_logvar = torch.full_like(dY, 5.0)

    monitor_fn = build_monitor_fn(
        "val_transition_score",
        tuple(range(8, 17)),
        dout=9,
        state_huber_delta=1.0,
        delta_huber_delta=1.0,
        tail_weight_power=1.2,
        acc_weight=3.0,
        gyro_weight=2.0,
        vel_weight=2.0,
    )
    assert monitor_fn is not None

    low_score = monitor_fn(_DummyModel(dY=dY, logvar=low_logvar), x, y, target_mask)
    high_score = monitor_fn(_DummyModel(dY=dY, logvar=high_logvar), x, y, target_mask)

    assert float(low_score.item()) == pytest.approx(float(high_score.item()))
