"""
模块名称：训练目标消融测试

模块职责：
验证普通监督目标 baseline 所需的 `state_mse` 与 `state_huber`
能够通过 canonical parser 和训练损失构造入口。

主要功能：
1. 验证 `loss.type=state_mse/state_huber` 能被配置解析。
2. 验证两类 loss 保持现有 `model -> dY/logvar -> rollout` 数据流。
3. 验证普通监督目标不依赖 transition_balance 的额外权重字段。

数据流：
train yaml dict
    ↓
build_from_dict()
    ↓
build_loss_fn()
    ↓
dummy model + X/Y
    ↓
scalar loss

依赖模块：
- torch
- uwnav_dynamics.train.config
- uwnav_dynamics.train.run_train

备注：
- 本测试只验证训练目标消融的代码入口，不代表具体实验配置已经落地。
"""

from __future__ import annotations

import pytest
import torch

from uwnav_dynamics.train.config import build_from_dict
from uwnav_dynamics.train.run_train import build_loss_fn


def _loss_ablation_yaml_dict(loss_type: str) -> dict:
    return {
        "run": {
            "name": "loss_ablation_demo",
            "seed": 10,
            "device": "cpu",
            "amp": False,
            "out_dir": "out/ckpts/loss_ablation_demo",
            "variant": loss_type,
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
            "name": "baseline_mlp",
            "din": 34,
            "dout": 9,
            "pred_len": 2,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "dropout": 0.0,
            "u_in_idx": list(range(0, 8)),
            "y_in_idx": list(range(8, 17)),
        },
        "rollout": {
            "y0_source": "x_last_state",
            "mode": "delta_cumsum",
        },
        "loss": {
            "type": loss_type,
            "logvar_clip": [-10.0, 6.0],
            "state_huber_delta": 0.5,
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


class _ZeroDeltaModel(torch.nn.Module):
    def __init__(self, pred_len: int, dout: int):
        super().__init__()
        self.pred_len = pred_len
        self.dout = dout

    def forward_with_aux(self, x: torch.Tensor):
        dY = torch.zeros((x.shape[0], self.pred_len, self.dout), device=x.device, dtype=x.dtype)
        logvar = torch.zeros_like(dY)
        return dY, logvar, {"dvl_obs": None}


@pytest.mark.parametrize("loss_type", ["state_mse", "state_huber"])
def test_plain_state_loss_types_parse_and_return_finite_scalar(loss_type: str) -> None:
    cfg = build_from_dict(_loss_ablation_yaml_dict(loss_type))
    assert cfg.loss.type == loss_type

    loss_fn = build_loss_fn(
        loss_type=cfg.loss.type,
        logvar_clip_min=cfg.loss.logvar_clip_min,
        logvar_clip_max=cfg.loss.logvar_clip_max,
        y_in_idx=cfg.model.y_in_idx,
        dout=cfg.model.dout,
        state_huber_delta=cfg.loss.state_huber_delta,
    )
    model = _ZeroDeltaModel(pred_len=cfg.model.pred_len, dout=cfg.model.dout)
    x = torch.zeros(4, 6, cfg.model.din)
    y = torch.ones(4, cfg.model.pred_len, cfg.model.dout)

    loss = loss_fn(model, x, y)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_plain_state_loss_rejects_transition_balance_weights() -> None:
    payload = _loss_ablation_yaml_dict("state_mse")
    payload["loss"]["state_final_weight"] = 0.5

    with pytest.raises(ValueError, match="transition_balance fields"):
        build_from_dict(payload)
