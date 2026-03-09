"""
模块名称：对角高斯 NLL 损失

模块职责：
定义 S1 预测模型当前使用的对角高斯负对数似然损失，
同时提供 dense supervision 与 mask-aware supervision 两条计算路径。

主要功能：
1. 计算标准 dense diagonal Gaussian NLL。
2. 计算 `target_mask` 驱动的 masked diagonal Gaussian NLL。
3. 在有效监督元素数为 0 时采用 fail-fast 策略，避免静默返回无意义损失。

数据流：
y_hat / y_true / logvar / optional target_mask
    ↓
elementwise diagonal Gaussian NLL
    ↓
dense mean 或 masked valid-mean
    ↓
scalar loss

依赖模块：
- torch

备注：
- 本模块不负责构造 `target_mask`。
- `target_mask` 的生成请使用 `uwnav_dynamics.supervision_mask`。
"""

from __future__ import annotations

import torch


def _elementwise_gaussian_nll(y_hat: torch.Tensor, y_true: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if y_hat.shape != y_true.shape or y_hat.shape != logvar.shape:
        raise ValueError(f"shape mismatch: y_hat{y_hat.shape}, y_true{y_true.shape}, logvar{logvar.shape}")

    logvar = torch.clamp(logvar, min=-10.0, max=6.0)
    e2 = (y_hat - y_true) ** 2
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * e2 + logvar)


def gaussian_nll_diag(y_hat: torch.Tensor, y_true: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Diagonal heteroscedastic Gaussian NLL.

    y_hat, y_true, logvar: (B,H,D)
    Loss = 0.5 * [ exp(-logvar)*e^2 + logvar ]
    """
    return _elementwise_gaussian_nll(y_hat, y_true, logvar).mean()


def gaussian_nll_diag_masked(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    logvar: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """计算带 `target_mask` 的对角高斯负对数似然。"""
    if target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: expect {y_hat.shape}, got {target_mask.shape}")

    nll = _elementwise_gaussian_nll(y_hat, y_true, logvar)
    mask = target_mask.to(device=nll.device, dtype=nll.dtype)
    valid = mask.sum()
    if float(valid.item()) <= 0.0:
        raise ValueError("target_mask contains zero valid supervision elements for this batch")
    return (nll * mask).sum() / valid
