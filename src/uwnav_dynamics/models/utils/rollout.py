from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


# 默认：按你当前 S1 layout（Din=25）
# X: [PWM 8] + [Acc 3] + [Gyro 3] + [Vel_state 3] + [Power 8]
_DEFAULT_Y0_IDXS: List[int] = [8, 9, 10, 11, 12, 13, 14, 15, 16]
_DEFAULT_Y0_COLS: List[str] = [
    "AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2",
    "GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s",
    "VelX_state_mps", "VelY_state_mps", "VelZ_state_mps",
]


def _resolve_y0_indices(
    *,
    din: int,
    input_cols: Optional[Sequence[str]] = None,
    y0_cols: Optional[Sequence[str]] = None,
    y0_indices: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Resolve indices for y0 extraction.

    Priority:
      1) explicit y0_indices
      2) (input_cols + y0_cols) mapping by name
      3) fallback to default fixed indices

    Returns:
      idx tensor: (9,) long on device
    """
    if y0_indices is not None:
        idx_list = list(map(int, y0_indices))
        if len(idx_list) != 9:
            raise ValueError(f"y0_indices must have length 9, got {len(idx_list)}")
        for i in idx_list:
            if i < 0 or i >= din:
                raise ValueError(f"y0_indices contains out-of-range index {i} for din={din}")
        return torch.tensor(idx_list, dtype=torch.long, device=device)

    if input_cols is not None:
        cols = list(input_cols)
        ycols = list(y0_cols) if y0_cols is not None else list(_DEFAULT_Y0_COLS)
        idx_list = []
        for name in ycols:
            if name not in cols:
                raise KeyError(
                    f"y0 col {name!r} not found in input_cols. "
                    f"Available example: {cols[:8]} ... (total {len(cols)})"
                )
            idx_list.append(cols.index(name))
        if len(idx_list) != 9:
            raise ValueError(f"resolved y0 indices length != 9, got {len(idx_list)}")
        return torch.tensor(idx_list, dtype=torch.long, device=device)

    # fallback: fixed layout
    if din <= max(_DEFAULT_Y0_IDXS):
        raise ValueError(f"din={din} too small for default y0 indices {max(_DEFAULT_Y0_IDXS)}")
    return torch.tensor(_DEFAULT_Y0_IDXS, dtype=torch.long, device=device)


def extract_y0_from_x_last(
    x: torch.Tensor,
    *,
    input_cols: Optional[Sequence[str]] = None,
    y0_cols: Optional[Sequence[str]] = None,
    y0_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Extract y0 from the last timestep of x.

    Args:
      x: (B, L, Din)
      input_cols: optional list of feature names for x last-dim (Din).
                  If provided, we can map by names (robust to column order changes).
      y0_cols: optional override of y0 column names (default: Acc/Gyro/Vel_state 9 dims).
      y0_indices: optional explicit indices (length 9). Highest priority.

    Returns:
      y0: (B, 9)  = [Acc(3), Gyro(3), Vel_state(3)]
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (B,L,Din), got {tuple(x.shape)}")

    B, L, Din = x.shape
    if L <= 0:
        raise ValueError("x has zero length in time dimension")

    # take last timestep: (B, Din)
    xl = x[:, -1, :]

    idx = _resolve_y0_indices(
        din=Din,
        input_cols=input_cols,
        y0_cols=y0_cols,
        y0_indices=y0_indices,
        device=x.device,
    )

    # (B, 9)
    y0 = xl.index_select(dim=1, index=idx)
    return y0


def rollout_from_delta(
    y0: torch.Tensor,
    dY: torch.Tensor,
) -> torch.Tensor:
    """
    Roll out future states from increments.

    Args:
      y0: (B, D)
      dY: (B, H, D)   increments for each step

    Returns:
      y_hat: (B, H, D)  predicted future states y_{k+1..k+H}
    """
    if y0.ndim != 2:
        raise ValueError(f"y0 must be (B,D), got {tuple(y0.shape)}")
    if dY.ndim != 3:
        raise ValueError(f"dY must be (B,H,D), got {tuple(dY.shape)}")

    B0, D0 = y0.shape
    B1, H, D1 = dY.shape
    if B0 != B1 or D0 != D1:
        raise ValueError(
            f"shape mismatch: y0{tuple(y0.shape)} vs dY{tuple(dY.shape)}; expected same B and D"
        )

    # ensure same dtype/device
    if dY.device != y0.device:
        dY = dY.to(device=y0.device)
    if dY.dtype != y0.dtype:
        dY = dY.to(dtype=y0.dtype)

    # y_{k+t} = y0 + sum_{i=1..t} dY_i
    y_hat = y0.unsqueeze(1) + torch.cumsum(dY, dim=1)
    return y_hat
