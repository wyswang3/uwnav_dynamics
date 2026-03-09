"""
模块名称：辅助监督损失

模块职责：
为 P0 阶段的辅助观测头提供轻量、可组合的回归损失，
当前只实现 DVL 稀疏辅助监督所需的 masked Huber loss。

主要功能：
1. 计算 `target_mask` 驱动的 masked Huber loss。
2. 在 shape 不匹配时显式报错，避免 silent bug。
3. 在全 False mask 时返回 0.0，兼容 DVL 稀疏监督的合法空批次。

数据流：
y_hat / y_true / target_mask
    ↓
elementwise huber
    ↓
masked valid-mean
    ↓
scalar auxiliary loss

依赖模块：
- torch

备注：
- 这里的“全 False mask 返回 0.0”只适用于辅助 DVL 稀疏监督。
- 它不同于主 state masked NLL 的 fail-fast 语义：
  主 state loss 无有效监督通常意味着训练路径异常；
  而 DVL 辅助头在某些 batch 内完全没有有效观测是合法情况。
"""

from __future__ import annotations

import torch


def masked_huber_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """在布尔掩码约束下计算 Huber 辅助监督损失。"""
    if y_hat.shape != y_true.shape:
        raise ValueError(f"shape mismatch: y_hat{y_hat.shape}, y_true{y_true.shape}")
    if target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: expect {y_hat.shape}, got {target_mask.shape}")
    if float(delta) <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")

    err = y_hat - y_true
    abs_err = torch.abs(err)
    delta_t = torch.as_tensor(float(delta), device=y_hat.device, dtype=y_hat.dtype)
    quad = 0.5 * err * err
    lin = delta_t * (abs_err - 0.5 * delta_t)
    huber = torch.where(abs_err <= delta_t, quad, lin)

    mask = target_mask.to(device=y_hat.device, dtype=y_hat.dtype)
    valid = mask.sum()
    if float(valid.item()) <= 0.0:
        # 这里返回 0.0 是合法的：
        # DVL 作为低频稀疏辅助监督，某些 batch 的 prediction horizon 内可能没有任何有效观测点。
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    return (huber * mask).sum() / valid
