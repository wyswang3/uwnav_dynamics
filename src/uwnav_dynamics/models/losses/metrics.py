from __future__ import annotations
import torch
from typing import Dict, Tuple

def _norm_mask(y_mask: torch.Tensor, B: int, H: int, D: int, dtype: torch.dtype) -> torch.Tensor:
    m = y_mask
    if m.ndim == 1:          # (B,)
        m = m.view(B, 1, 1).expand(B, H, D)
    elif m.ndim == 2:        # (B,H)
        m = m.view(B, H, 1).expand(B, H, D)
    elif m.ndim == 3:
        if m.shape[2] == 1:
            m = m.expand(B, H, D)
        elif m.shape[2] != D:
            raise ValueError(f"y_mask last dim must be 1 or D={D}, got {m.shape[2]}")
    else:
        raise ValueError(f"y_mask must be 1D/2D/3D, got ndim={m.ndim}")
    return m.to(dtype=dtype)

@torch.no_grad()
def masked_regression_metrics(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Returns scalar metrics over all supervised elements.
    y_hat/y_true: (B,H,D)
    y_mask: (B,H,D) or broadcastable variants
    """
    if y_hat.shape != y_true.shape:
        raise ValueError(f"shape mismatch: y_hat{y_hat.shape} vs y_true{y_true.shape}")
    B, H, D = y_hat.shape
    m = _norm_mask(y_mask, B, H, D, dtype=y_hat.dtype)

    err = (y_hat - y_true)
    abs_err = torch.abs(err)
    sq_err = err * err

    denom = m.sum().clamp_min(eps)
    mae = (abs_err * m).sum() / denom
    mse = (sq_err * m).sum() / denom
    rmse = torch.sqrt(mse)

    # 可选：分组（acc/gyro/vel）平均误差，D=9 时很有用
    out = {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "mse": float(mse.item()),
        "valid_ratio": float((m.mean()).item()),
        "valid_count": float(denom.item()),
    }

    if D == 9:
        def _slice_metric(sl: slice) -> Tuple[float, float]:
            denom_g = m[..., sl].sum().clamp_min(eps)
            mae_g = (abs_err[..., sl] * m[..., sl]).sum() / denom_g
            mse_g = (sq_err[..., sl] * m[..., sl]).sum() / denom_g
            rmse_g = torch.sqrt(mse_g)
            return float(mae_g.item()), float(rmse_g.item())

        out["mae_acc"], out["rmse_acc"] = _slice_metric(slice(0, 3))
        out["mae_gyro"], out["rmse_gyro"] = _slice_metric(slice(3, 6))
        out["mae_vel"], out["rmse_vel"] = _slice_metric(slice(6, 9))

    return out
