from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


Scaler = Dict[str, np.ndarray]


def fit_scaler(x_train: np.ndarray) -> Scaler:
    """
    Fit z-score statistics on train subset only.

    Works with arrays shaped (N, D) or (N, T, D); statistics are computed over all
    leading dimensions and keep the last dim as feature dim.
    """
    arr = np.asarray(x_train, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"fit_scaler expects ndim>=2, got shape={arr.shape}")

    feat_dim = arr.shape[-1]
    flat = arr.reshape(-1, feat_dim)

    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)

    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-8] = 1.0

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def transform(x: np.ndarray, scaler: Scaler) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"transform expects ndim>=2, got shape={arr.shape}")

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if arr.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            f"shape mismatch: x last dim={arr.shape[-1]}, mean={mean.shape}, std={std.shape}"
        )

    view_shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    return (arr - mean.reshape(view_shape)) / std.reshape(view_shape)


def save_scaler(path: str | Path, scaler: Scaler) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        mean=np.asarray(scaler["mean"], dtype=np.float32),
        std=np.asarray(scaler["std"], dtype=np.float32),
    )
    return p


def load_scaler(path: str | Path) -> Scaler:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"scaler not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        out: Scaler = {
            "mean": np.asarray(z["mean"], dtype=np.float32),
            "std": np.asarray(z["std"], dtype=np.float32),
        }
    return out
