from __future__ import annotations

import torch


def extract_y0_from_x_last(x: torch.Tensor) -> torch.Tensor:
    """
    X columns (by your YAML order):
      0..7   : ch1..ch8
      8..10  : Acc (3)
      11..13 : Gyro (3)
      14..16 : Vel_state (3)
      17..24 : Power (8)

    y0 = [Acc(3), Gyro(3), Vel(3)] => 9 dims
    """
    # (B, Din) take last timestep
    xl = x[:, -1, :]  # (B,25)
    idx = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16], device=x.device)
    y0 = xl.index_select(dim=1, index=idx)
    return y0


def rollout_from_delta(y0: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    """
    y0: (B, Dout=9)      state at time k
    dY: (B, H, Dout=9)   increments from k->k+1, k+1->k+2, ...
    y_hat: (B, H, Dout)  predicted future states y_{k+1..k+H}
    """
    # cumulative sum over time dimension
    y_hat = y0.unsqueeze(1) + torch.cumsum(dY, dim=1)
    return y_hat
