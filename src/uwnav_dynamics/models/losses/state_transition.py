"""
模块名称：状态转移复合损失辅助

模块职责：
为状态转移模型训练提供按语义组与 horizon 加权的损失辅助，
把 rollout 状态误差、delta 转移误差与 log-variance 正则统一收口。

主要功能：
1. 生成 `acc / gyro / vel` 三组语义权重向量。
2. 生成强调尾部 horizon 的时间权重向量。
3. 从未来状态序列反推真实状态增量 `dY_true`。
4. 计算带 mask / 组权重 / horizon 权重的 Huber 损失与正则项。

数据流：
y0 / y_true / y_hat / dY / logvar / target_mask
    ↓
group weights + horizon weights
    ↓
weighted reduction
    ↓
state / delta / uncertainty regularization terms

依赖模块：
- torch
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 这里的加权只改变优化重点，不改变 train / eval 的 rollout 数学契约。
- 若 `target_mask` 存在，它仍然是监督有效性的唯一运行时真源。
"""

from __future__ import annotations

import torch

from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout


def build_group_weight_vector(
    dout: int,
    *,
    acc_weight: float,
    gyro_weight: float,
    vel_weight: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """按 canonical `acc / gyro / vel` 语义组生成维度权重向量。"""
    if float(acc_weight) <= 0.0 or float(gyro_weight) <= 0.0 or float(vel_weight) <= 0.0:
        raise ValueError("group weights must be > 0")
    layout = canonical_semantic_output_layout(int(dout))
    weights = torch.ones(int(dout), device=device, dtype=dtype if dtype is not None else torch.float32)
    for idx in layout.group_indices["acc"]:
        weights[int(idx)] = float(acc_weight)
    for idx in layout.group_indices["gyro"]:
        weights[int(idx)] = float(gyro_weight)
    for idx in layout.group_indices["vel"]:
        weights[int(idx)] = float(vel_weight)
    return weights


def build_horizon_weight_vector(
    horizon: int,
    *,
    tail_weight_power: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """生成均值归一化的 horizon 权重；`power=0` 时退化为全 1。"""
    horizon_i = int(horizon)
    if horizon_i <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    power = float(tail_weight_power)
    if power < 0.0:
        raise ValueError(f"tail_weight_power must be >= 0, got {tail_weight_power}")
    base = torch.arange(1, horizon_i + 1, device=device, dtype=dtype if dtype is not None else torch.float32)
    if power == 0.0:
        return torch.ones(horizon_i, device=base.device, dtype=base.dtype)
    weights = torch.pow(base / float(horizon_i), power)
    return weights / torch.clamp(weights.mean(), min=torch.finfo(weights.dtype).eps)


def build_delta_targets(y0: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """由未来状态序列反推真实状态转移增量。"""
    if y0.ndim != 2:
        raise ValueError(f"y0 must be (B,D), got {tuple(y0.shape)}")
    if y_true.ndim != 3:
        raise ValueError(f"y_true must be (B,H,D), got {tuple(y_true.shape)}")
    if y0.shape[0] != y_true.shape[0] or y0.shape[1] != y_true.shape[2]:
        raise ValueError(f"shape mismatch: y0{tuple(y0.shape)}, y_true{tuple(y_true.shape)}")
    first = y_true[:, :1, :] - y0.unsqueeze(1)
    if y_true.shape[1] == 1:
        return first
    rest = y_true[:, 1:, :] - y_true[:, :-1, :]
    return torch.cat([first, rest], dim=1)


def reduce_weighted_mean(
    values: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    component_weight: torch.Tensor | None = None,
    horizon_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """在 mask / 维度权重 / horizon 权重下做合法加权平均。"""
    if values.ndim != 3:
        raise ValueError(f"values must be (B,H,D), got {tuple(values.shape)}")
    weights = torch.ones_like(values)
    if component_weight is not None:
        if component_weight.ndim != 1 or component_weight.shape[0] != values.shape[2]:
            raise ValueError(
                f"component_weight must be (D,), got {tuple(component_weight.shape)} for values{tuple(values.shape)}"
            )
        weights = weights * component_weight.view(1, 1, -1).to(device=values.device, dtype=values.dtype)
    if horizon_weight is not None:
        if horizon_weight.ndim != 1 or horizon_weight.shape[0] != values.shape[1]:
            raise ValueError(
                f"horizon_weight must be (H,), got {tuple(horizon_weight.shape)} for values{tuple(values.shape)}"
            )
        weights = weights * horizon_weight.view(1, -1, 1).to(device=values.device, dtype=values.dtype)
    if target_mask is not None:
        if target_mask.shape != values.shape:
            raise ValueError(f"target_mask shape mismatch: expect {tuple(values.shape)}, got {tuple(target_mask.shape)}")
        weights = weights * target_mask.to(device=values.device, dtype=values.dtype)
    valid = weights.sum()
    if float(valid.item()) <= 0.0:
        raise ValueError("weighted reduction contains zero valid supervision elements")
    return (values * weights).sum() / valid


def masked_weighted_huber_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    component_weight: torch.Tensor | None = None,
    horizon_weight: torch.Tensor | None = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """计算可组合的带权 Huber 损失。"""
    if y_hat.shape != y_true.shape:
        raise ValueError(f"shape mismatch: y_hat{tuple(y_hat.shape)}, y_true{tuple(y_true.shape)}")
    if float(delta) <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    err = y_hat - y_true
    abs_err = torch.abs(err)
    delta_t = torch.as_tensor(float(delta), device=y_hat.device, dtype=y_hat.dtype)
    quad = 0.5 * err * err
    lin = delta_t * (abs_err - 0.5 * delta_t)
    huber = torch.where(abs_err <= delta_t, quad, lin)
    return reduce_weighted_mean(
        huber,
        target_mask=target_mask,
        component_weight=component_weight,
        horizon_weight=horizon_weight,
    )


def positive_logvar_penalty(
    logvar: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    component_weight: torch.Tensor | None = None,
    horizon_weight: torch.Tensor | None = None,
    free_logvar: float = 0.0,
) -> torch.Tensor:
    """只惩罚过大的正向 log-variance，避免模型用不确定度吞掉误差。"""
    penalty = torch.square(
        torch.relu(
            logvar - torch.as_tensor(float(free_logvar), device=logvar.device, dtype=logvar.dtype)
        )
    )
    return reduce_weighted_mean(
        penalty,
        target_mask=target_mask,
        component_weight=component_weight,
        horizon_weight=horizon_weight,
    )
