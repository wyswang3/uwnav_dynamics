"""
模块名称：P0 训练配置解析测试

模块职责：
验证 P0.1 在 canonical train parser 中新增的最小配置字段，
同时确保旧 YAML 继续兼容，且 strict schema 仍然生效。

主要功能：
1. 验证 `model.aux_heads.dvl_obs.*` 与 `loss.dvl_obs_*` 能正确解析。
2. 验证旧 YAML 缺少新增字段时仍保持默认行为。
3. 验证非法组合与 unknown keys 会在 parser 阶段被拒绝。

数据流：
原始 train yaml dict
    ↓
build_from_dict()
    ↓
TrainYamlConfig
    ↓
配置字段与校验断言

依赖模块：
- pytest
- uwnav_dynamics.train.config

备注：
- Patch A 只收口 parser，不接模型构造与训练执行。
"""

from __future__ import annotations

import pytest

from uwnav_dynamics.train.config import build_from_dict


def _base_train_yaml_dict() -> dict:
    return {
        "run": {
            "name": "p0_parser_demo",
            "seed": 0,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/p0_parser_demo",
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


def test_train_config_p0_defaults_keep_old_yaml_compatible():
    cfg = build_from_dict(_base_train_yaml_dict())

    assert cfg.model_aux_heads.dvl_obs.enabled is False
    assert cfg.model_aux_heads.dvl_obs.hidden == 128
    assert cfg.loss.state_mse_weight == 0.0
    assert cfg.loss.dvl_obs_weight == 0.0
    assert cfg.loss.dvl_obs_delta == 1.0
    assert cfg.train.scheduler_name == "none"
    assert cfg.train.early_stopping_patience == 0
    assert cfg.train.early_stopping_min_delta == 0.0


def test_train_config_p0_parses_dvl_aux_fields():
    raw = _base_train_yaml_dict()
    raw["model"]["aux_heads"] = {
        "dvl_obs": {
            "enabled": True,
            "hidden": 96,
        }
    }
    raw["loss"]["dvl_obs_weight"] = 0.2
    raw["loss"]["dvl_obs_delta"] = 0.75

    cfg = build_from_dict(raw)

    assert cfg.model_aux_heads.dvl_obs.enabled is True
    assert cfg.model_aux_heads.dvl_obs.hidden == 96
    assert cfg.loss.dvl_obs_weight == pytest.approx(0.2)
    assert cfg.loss.dvl_obs_delta == pytest.approx(0.75)


def test_train_config_p0_parses_scheduler_and_early_stopping():
    raw = _base_train_yaml_dict()
    raw["optim"]["scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.4,
        "patience": 3,
        "min_lr": 1.0e-5,
    }
    raw["train"]["early_stopping"] = {
        "patience": 7,
        "min_delta": 1.0e-4,
    }

    cfg = build_from_dict(raw)

    assert cfg.train.scheduler_name == "reduce_on_plateau"
    assert cfg.train.scheduler_factor == pytest.approx(0.4)
    assert cfg.train.scheduler_patience == 3
    assert cfg.train.scheduler_min_lr == pytest.approx(1.0e-5)
    assert cfg.train.early_stopping_patience == 7
    assert cfg.train.early_stopping_min_delta == pytest.approx(1.0e-4)


def test_train_config_p0_rejects_weight_without_enabled_head():
    raw = _base_train_yaml_dict()
    raw["loss"]["dvl_obs_weight"] = 0.1

    with pytest.raises(ValueError, match="model\\.aux_heads\\.dvl_obs\\.enabled=true"):
        build_from_dict(raw)


def test_train_config_p0_rejects_unknown_aux_keys():
    raw = _base_train_yaml_dict()
    raw["model"]["aux_heads"] = {
        "dvl_obs": {
            "enabled": True,
            "hidden": 64,
            "dropout": 0.1,
        }
    }

    with pytest.raises(KeyError, match="Unknown keys in model\\.aux_heads\\.dvl_obs"):
        build_from_dict(raw)


def test_train_config_p0_rejects_unknown_loss_keys():
    raw = _base_train_yaml_dict()
    raw["loss"]["dvl_obs_gamma"] = 0.3

    with pytest.raises(KeyError, match="Unknown keys in loss"):
        build_from_dict(raw)


def test_train_config_p0_rejects_unknown_scheduler_keys():
    raw = _base_train_yaml_dict()
    raw["optim"]["scheduler"] = {
        "name": "reduce_on_plateau",
        "cooldown": 2,
    }

    with pytest.raises(KeyError, match="Unknown keys in optim\\.scheduler"):
        build_from_dict(raw)
