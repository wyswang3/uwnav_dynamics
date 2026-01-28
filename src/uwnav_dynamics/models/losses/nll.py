from __future__ import annotations

import torch


def gaussian_nll_diag(y_hat: torch.Tensor, y_true: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Diagonal heteroscedastic Gaussian NLL.

    y_hat, y_true, logvar: (B,H,D)
    Loss = 0.5 * [ exp(-logvar)*e^2 + logvar ]
    """
    if y_hat.shape != y_true.shape or y_hat.shape != logvar.shape:
        raise ValueError(f"shape mismatch: y_hat{y_hat.shape}, y_true{y_true.shape}, logvar{logvar.shape}")

    # Clamp logvar for numerical stability
    logvar = torch.clamp(logvar, min=-10.0, max=6.0)

    e2 = (y_hat - y_true) ** 2
    inv_var = torch.exp(-logvar)

    nll = 0.5 * (inv_var * e2 + logvar)
    return nll.mean()
