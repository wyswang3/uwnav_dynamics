"""
模块名称：常见网络 baseline 测试

模块职责：
验证论文对照实验所需的常见网络 baseline 能够通过统一模型工厂接入，
并保持现有训练、评估与 replay 的状态转移输出契约。

主要功能：
1. 验证 canonical parser 接受 baseline model.name 且不要求旧 blocks。
2. 验证 MLP / GRU / TCN / Transformer baseline 的 forward 输出 shape。
3. 验证 baseline 的 `forward_with_aux()` 保持稳定 aux key。

数据流：
baseline train yaml dict
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

备注：
- 本测试只验证模块接口和基础数值安全，不代表 baseline 已完成训练或优于主线。
"""

from __future__ import annotations

import pytest
import torch

from uwnav_dynamics.models.nets.factory import build_state_predictor
from uwnav_dynamics.train.config import build_from_dict


def _baseline_train_yaml_dict(model_name: str) -> dict:
    return {
        "run": {
            "name": "common_baseline_demo",
            "seed": 10,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/common_baseline_demo",
            "variant": model_name,
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
            "name": model_name,
            "din": 34,
            "dout": 9,
            "pred_len": 2,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "dropout": 0.0,
            "u_in_idx": list(range(0, 8)),
            "y_in_idx": list(range(8, 17)),
            "head_mode": "joint",
            "group_head_hidden": 32,
            "tcn_kernel_size": 3,
            "transformer_heads": 4,
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
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
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


@pytest.mark.parametrize(
    "model_name",
    ["baseline_mlp", "baseline_gru", "baseline_tcn", "baseline_transformer"],
)
def test_common_baseline_models_factory_and_forward_contract(model_name: str) -> None:
    cfg = build_from_dict(_baseline_train_yaml_dict(model_name))
    model = build_state_predictor(cfg).eval()

    assert cfg.model.name == model_name
    assert cfg.model.blocks.thruster_lag.enabled is False
    assert cfg.model.blocks.hydro_ssm.enabled is False
    assert cfg.model.blocks.damping.enabled is False
    assert cfg.model.blocks.uncertainty.enabled is False

    x = torch.randn(4, 12, cfg.model.din)
    with torch.no_grad():
        dY, logvar = model(x)
        dY_aux, logvar_aux, aux = model.forward_with_aux(x)

    assert dY.shape == (4, cfg.model.pred_len, cfg.model.dout)
    assert logvar.shape == dY.shape
    assert torch.allclose(dY, dY_aux)
    assert torch.allclose(logvar, logvar_aux)
    assert torch.isfinite(dY).all()
    assert torch.isfinite(logvar).all()
    assert "dvl_obs" in aux
    assert aux["dvl_obs"] is None


def test_common_baseline_transformer_rejects_incompatible_heads() -> None:
    payload = _baseline_train_yaml_dict("baseline_transformer")
    payload["model"]["rnn_hidden"] = 30
    payload["model"]["transformer_heads"] = 8

    with pytest.raises(ValueError, match="divisible"):
        build_from_dict(payload)
