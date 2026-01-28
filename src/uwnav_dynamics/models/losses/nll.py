from __future__ import annotations

from typing import Dict, Tuple, Union, Optional

import torch


def gaussian_nll_diag(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    logvar: torch.Tensor,
    *,
    clamp_min: float = -10.0,
    clamp_max: float = 6.0,
) -> torch.Tensor:
    """
    Diagonal heteroscedastic Gaussian NLL.

    Shapes:
      y_hat, y_true, logvar: (B, H, D)

    Loss per element:
      0.5 * [ exp(-logvar) * (y_hat - y_true)^2 + logvar ]
    """
    if y_hat.shape != y_true.shape or y_hat.shape != logvar.shape:
        raise ValueError(
            f"shape mismatch: y_hat{tuple(y_hat.shape)}, y_true{tuple(y_true.shape)}, logvar{tuple(logvar.shape)}"
        )

    logvar_c = torch.clamp(logvar, min=clamp_min, max=clamp_max)
    e2 = (y_hat - y_true) ** 2
    inv_var = torch.exp(-logvar_c)

    nll = 0.5 * (inv_var * e2 + logvar_c)
    return nll.mean()


def _broadcast_mask_to_bhd(mask: torch.Tensor, B: int, H: int, D: int) -> torch.Tensor:
    """
    Normalize mask shape to (B, H, D).

    Accept:
      - (B,) -> expand to (B,H,D)
      - (B,H) -> expand to (B,H,D)
      - (B,H,1) -> expand to (B,H,D)
      - (B,H,D) -> keep
    """
    m = mask
    if m.ndim == 1:  # (B,)
        if m.shape[0] != B:
            raise ValueError(f"y_mask (B,) expected B={B}, got {tuple(m.shape)}")
        m = m.view(B, 1, 1).expand(B, H, D)
    elif m.ndim == 2:  # (B,H)
        if m.shape[0] != B or m.shape[1] != H:
            raise ValueError(f"y_mask (B,H) expected {(B, H)}, got {tuple(m.shape)}")
        m = m.view(B, H, 1).expand(B, H, D)
    elif m.ndim == 3:
        if m.shape[0] != B or m.shape[1] != H:
            raise ValueError(f"y_mask (B,H,*) expected {(B, H, '*')}, got {tuple(m.shape)}")
        if m.shape[2] == 1:
            m = m.expand(B, H, D)
        elif m.shape[2] != D:
            raise ValueError(f"y_mask last dim must be 1 or D={D}, got {m.shape[2]}")
    else:
        raise ValueError(f"y_mask must be 1D/2D/3D, got ndim={m.ndim}")
    return m


def gaussian_nll_diag_masked(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    logvar: torch.Tensor,
    y_mask: torch.Tensor,
    *,
    dim_weight: Optional[torch.Tensor] = None,
    min_count: int = 1,
    clamp_min: float = -10.0,
    clamp_max: float = 6.0,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    Masked diagonal heteroscedastic Gaussian NLL for async supervision.

    Shapes:
      y_hat, y_true, logvar: (B, H, D)
      y_mask: (B,H,D) or (B,H,1) or (B,H) or (B,)  with 1=valid/supervised, 0=missing.

    Optional:
      dim_weight: (D,) or (1,1,D) or (B,H,D) broadcastable weight.
                  Use it to upweight sparse dimensions (e.g., velocity dims).

    Reduction:
      loss = sum(nll * mask * weight) / (sum(mask * weight) + eps)

    Notes:
      - If effective valid_count < min_count: return 0 with zero gradient w.r.t. y_hat/logvar.
    """
    if y_hat.shape != y_true.shape or y_hat.shape != logvar.shape:
        raise ValueError(
            f"shape mismatch: y_hat{tuple(y_hat.shape)}, y_true{tuple(y_true.shape)}, logvar{tuple(logvar.shape)}"
        )

    B, H, D = y_hat.shape

    # ---- normalize mask to (B,H,D) on same device/dtype ----
    m = _broadcast_mask_to_bhd(y_mask, B, H, D).to(device=y_hat.device)
    m = m.to(dtype=y_hat.dtype)

    # ---- optional dim weights ----
    w: torch.Tensor
    if dim_weight is None:
        w = torch.ones((1, 1, D), device=y_hat.device, dtype=y_hat.dtype)
    else:
        w = dim_weight.to(device=y_hat.device, dtype=y_hat.dtype)
        # normalize to broadcastable; allow (D,) or already broadcastable
        if w.ndim == 1:
            if w.numel() != D:
                raise ValueError(f"dim_weight (D,) expected D={D}, got numel={w.numel()}")
            w = w.view(1, 1, D)
        # else: assume caller provides broadcastable shape

    mw = m * w  # effective mask weight

    # ---- elementwise nll ----
    logvar_c = torch.clamp(logvar, min=clamp_min, max=clamp_max)
    e2 = (y_hat - y_true) ** 2
    inv_var = torch.exp(-logvar_c)
    nll = 0.5 * (inv_var * e2 + logvar_c)  # (B,H,D)

    # ---- masked weighted average ----
    valid_weight = mw.sum()  # scalar
    # 用 detach 的标量判断是否跳过（避免把判断图结构搞复杂）
    if float(valid_weight.detach().cpu().item()) < float(min_count):
        loss = y_hat.sum() * 0.0  # zero grad
        if return_stats:
            stats = {
                "valid_weight": float(valid_weight.detach().cpu().item()),
                "total_count": float(B * H * D),
                "valid_ratio": float(valid_weight.detach().cpu().item() / max(B * H * D, 1)),
                "skipped": 1.0,
            }
            return loss, stats
        return loss

    loss = (nll * mw).sum() / (valid_weight + eps)

    if return_stats:
        stats = {
            "valid_weight": float(valid_weight.detach().cpu().item()),
            "total_count": float(B * H * D),
            "valid_ratio": float(valid_weight.detach().cpu().item() / max(B * H * D, 1)),
            "skipped": 0.0,
        }
        return loss, stats

    return loss

def masked_mse(y_hat: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B, H, D = y_hat.shape
    # 复用你 gaussian_nll_diag_masked 里的 mask 归一化逻辑也行
    m = y_mask
    if m.ndim == 1:
        m = m.view(B,1,1).expand(B,H,D)
    elif m.ndim == 2:
        m = m.view(B,H,1).expand(B,H,D)
    elif m.ndim == 3 and m.shape[2] == 1:
        m = m.expand(B,H,D)
    m = m.to(dtype=y_hat.dtype)

    e2 = (y_hat - y_true) ** 2
    denom = m.sum().clamp_min(eps)
    return (e2 * m).sum() / denom

def gaussian_nll_diag_masked_plus_mse(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    logvar: torch.Tensor,
    y_mask: torch.Tensor,
    *,
    alpha_mse: float = 0.01,
    min_count: int = 1,
) -> torch.Tensor:
    nll = gaussian_nll_diag_masked(y_hat, y_true, logvar, y_mask, min_count=min_count, return_stats=False)
    mse = masked_mse(y_hat, y_true, y_mask)
    return nll + float(alpha_mse) * mse
