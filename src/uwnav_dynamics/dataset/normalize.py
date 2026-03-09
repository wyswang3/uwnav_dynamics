"""
模块名称：数据归一化工具

模块职责：
负责训练与评估阶段共享的 z-score 归一化、反归一化与 scaler 落盘读取，
保证 dataset build、train、eval 对同一统计量的解释一致。

主要功能：
1. 在训练子集上拟合逐维 `mean/std`。
2. 对 `(N, D)` 或 `(N, T, D)` 数组执行 `transform / inverse_transform`。
3. 提供 `save_scaler / load_scaler`，供训练与评估复用同一 scaler artifact。

数据流：
train subset arrays
    ↓
fit_scaler()
    ↓
scalers/x_scaler.npz + scalers/y_scaler.npz
    ↓
transform() / inverse_transform()
    ↓
dataset build / train / eval 共享同一数值语义

依赖模块：
- numpy
- pathlib

备注：
- 统计量始终按最后一维解释为 feature 维。
- 非有限 `mean/std` 会被回退到安全默认值，避免 scaler artifact 污染后续流程。
"""

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
    """使用给定 scaler 对数组最后一维执行 z-score 标准化。"""
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


def inverse_transform(x: np.ndarray, scaler: Scaler) -> np.ndarray:
    """把 z-score 空间数组恢复到原始物理量纲。"""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"inverse_transform expects ndim>=2, got shape={arr.shape}")

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if arr.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            f"shape mismatch: x last dim={arr.shape[-1]}, mean={mean.shape}, std={std.shape}"
        )

    view_shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    return arr * std.reshape(view_shape) + mean.reshape(view_shape)


def save_scaler(path: str | Path, scaler: Scaler) -> Path:
    """将 scaler 以 `npz` 形式落盘到指定路径。"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        mean=np.asarray(scaler["mean"], dtype=np.float32),
        std=np.asarray(scaler["std"], dtype=np.float32),
    )
    return p


def load_scaler(path: str | Path) -> Scaler:
    """从磁盘读取先前保存的 scaler artifact。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"scaler not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        out: Scaler = {
            "mean": np.asarray(z["mean"], dtype=np.float32),
            "std": np.asarray(z["std"], dtype=np.float32),
        }
    return out
