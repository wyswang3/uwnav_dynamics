from __future__ import annotations

"""
Training data preparation pipeline.

This module owns the non-model part of the training contract:
  - load raw sliding-window tensors from `features.npz` / `labels.npz`
  - create and persist deterministic split indices
  - fit scalers on the train subset only
  - materialize DataLoaders for train/val/test
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from uwnav_dynamics.dataset.normalize import (
    fit_scaler,
    load_scaler,
    save_scaler,
    transform,
)
from uwnav_dynamics.dataset.split import (
    DEFAULT_SPLIT_STRATEGY,
    load_split_indices,
    make_split_indices,
    save_split_indices,
)
from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.train.data import DataConfig


@dataclass(frozen=True)
class PreparedTrainData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_total: int
    split_sizes: Dict[str, int]
    mask_shapes: Dict[str, tuple[int, ...]]


def _load_xy(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, tuple[int, ...]]]:
    """
    Load the dense training tensors and report any optional mask tensor shapes.

    Note: masks are currently exposed as metadata only; the loaders themselves
    still return `(X, Y)` so downstream code can decide if and how to consume
    the mask artifacts.
    """
    feat_npz = data_dir / "features.npz"
    lab_npz = data_dir / "labels.npz"
    if not feat_npz.exists():
        raise FileNotFoundError(f"Missing: {feat_npz}")
    if not lab_npz.exists():
        raise FileNotFoundError(f"Missing: {lab_npz}")

    mask_shapes: Dict[str, tuple[int, ...]] = {}
    with np.load(feat_npz, allow_pickle=False) as z:
        if "X" not in z:
            raise KeyError(f"'X' not found in {feat_npz}")
        X = np.asarray(z["X"], dtype=np.float32)
        for k in ("dvl_mask_hist", "power_mask_hist"):
            if k in z:
                mask_shapes[k] = tuple(np.asarray(z[k]).shape)
    with np.load(lab_npz, allow_pickle=False) as z:
        if "Y" not in z:
            raise KeyError(f"'Y' not found in {lab_npz}")
        Y = np.asarray(z["Y"], dtype=np.float32)
        for k in ("dvl_mask", "power_mask"):
            if k in z:
                mask_shapes[k] = tuple(np.asarray(z[k]).shape)

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expect X/Y to be 3D, got X={X.shape}, Y={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y window counts mismatch: {X.shape[0]} vs {Y.shape[0]}")
    return X, Y, mask_shapes


def _validate_indices(indices: Dict[str, np.ndarray], n: int) -> None:
    req = ("train", "val", "test")
    for k in req:
        if k not in indices:
            raise KeyError(f"missing split key: {k}")

    merged = np.concatenate(
        [
            np.asarray(indices["train"], dtype=np.int64),
            np.asarray(indices["val"], dtype=np.int64),
            np.asarray(indices["test"], dtype=np.int64),
        ],
        axis=0,
    )
    if merged.size == 0:
        raise ValueError("split indices are empty")
    if np.any(merged < 0) or np.any(merged >= n):
        raise ValueError(f"split indices out of range [0,{n})")
    if np.unique(merged).size != merged.size:
        raise ValueError("split indices contain overlap (data leakage risk)")
    if merged.size != n:
        raise ValueError(f"split indices do not cover all samples: {merged.size} vs {n}")


def _make_loader(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def prepare_train_data(cfg: DataConfig, run_layout: RunLayout) -> PreparedTrainData:
    """Prepare split/scaler/DataLoader artifacts for one concrete run layout."""
    X_all, Y_all, mask_shapes = _load_xy(Path(cfg.data_dir))
    n_total = int(X_all.shape[0])
    if mask_shapes:
        print(f"[DATA] found mask tensors: {mask_shapes}")

    split_path = run_layout.split_indices_path
    if split_path.exists():
        split_indices = load_split_indices(split_path)
        print(f"[SPLIT] reuse: {split_path}")
    else:
        # 对滑窗数据集而言，“索引不重叠”不等于“时间无泄漏”。
        # 当前 canonical 策略使用 contiguous_v1：按时间顺序切 train/val/test，
        # 再把确切索引落盘，供 train / eval 共享。
        split_indices = make_split_indices(
            n=n_total,
            seed=int(cfg.seed),
            ratios={
                "train": float(cfg.train_ratio),
                "val": float(cfg.val_ratio),
                "test": float(1.0 - cfg.train_ratio - cfg.val_ratio),
            },
        )
        save_split_indices(split_path, split_indices)
        print(f"[SPLIT] created: {split_path} strategy={DEFAULT_SPLIT_STRATEGY}")

    _validate_indices(split_indices, n_total)
    train_idx = np.asarray(split_indices["train"], dtype=np.int64)
    val_idx = np.asarray(split_indices["val"], dtype=np.int64)
    test_idx = np.asarray(split_indices["test"], dtype=np.int64)
    print(f"[SPLIT] N={n_total} train={train_idx.size} val={val_idx.size} test={test_idx.size}")

    x_scaler_path = run_layout.x_scaler_path
    y_scaler_path = run_layout.y_scaler_path
    if x_scaler_path.exists() and y_scaler_path.exists():
        x_scaler = load_scaler(x_scaler_path)
        y_scaler = load_scaler(y_scaler_path)
        print(f"[SCALER] reuse: {run_layout.scalers_dir}")
    else:
        x_scaler = fit_scaler(X_all[train_idx])
        y_scaler = fit_scaler(Y_all[train_idx])
        save_scaler(x_scaler_path, x_scaler)
        save_scaler(y_scaler_path, y_scaler)
        print(f"[SCALER] fitted on train split and saved to: {run_layout.scalers_dir}")

    X_train = transform(X_all[train_idx], x_scaler).astype(np.float32, copy=False)
    Y_train = transform(Y_all[train_idx], y_scaler).astype(np.float32, copy=False)
    X_val = transform(X_all[val_idx], x_scaler).astype(np.float32, copy=False)
    Y_val = transform(Y_all[val_idx], y_scaler).astype(np.float32, copy=False)
    X_test = transform(X_all[test_idx], x_scaler).astype(np.float32, copy=False)
    Y_test = transform(Y_all[test_idx], y_scaler).astype(np.float32, copy=False)

    train_loader = _make_loader(
        X_train,
        Y_train,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=True,
    )
    val_loader = _make_loader(
        X_val,
        Y_val,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=False,
    )
    test_loader = _make_loader(
        X_test,
        Y_test,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=False,
    )

    return PreparedTrainData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_total=n_total,
        split_sizes={
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
        mask_shapes=mask_shapes,
    )
