from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 0


def _maybe_build_memmap_cache(data_dir: Path) -> Tuple[Path, Path]:
    """
    Convert features.npz/labels.npz to X.npy/Y.npy for memmap reading.
    This avoids loading huge arrays into RAM every time.

    Expected:
      - data_dir/features.npz: contains 'X'
      - data_dir/labels.npz:   contains 'Y'
    Produces:
      - data_dir/X.npy
      - data_dir/Y.npy
    """
    data_dir = Path(data_dir)
    x_npy = data_dir / "X.npy"
    y_npy = data_dir / "Y.npy"

    if x_npy.exists() and y_npy.exists():
        return x_npy, y_npy

    feat_npz = data_dir / "features.npz"
    lab_npz = data_dir / "labels.npz"
    if not feat_npz.exists():
        raise FileNotFoundError(f"Missing: {feat_npz}")
    if not lab_npz.exists():
        raise FileNotFoundError(f"Missing: {lab_npz}")

    print(f"[DATA] building memmap cache ...")
    print(f"[DATA] reading: {feat_npz}")
    with np.load(feat_npz, allow_pickle=False) as z:
        if "X" not in z:
            raise KeyError(f"'X' not found in {feat_npz}")
        X = z["X"].astype(np.float32, copy=False)

    print(f"[DATA] reading: {lab_npz}")
    with np.load(lab_npz, allow_pickle=False) as z:
        if "Y" not in z:
            raise KeyError(f"'Y' not found in {lab_npz}")
        Y = z["Y"].astype(np.float32, copy=False)

    print(f"[DATA] saving: {x_npy} shape={X.shape} dtype={X.dtype}")
    np.save(x_npy, X)
    print(f"[DATA] saving: {y_npy} shape={Y.shape} dtype={Y.dtype}")
    np.save(y_npy, Y)

    # free memory quickly
    del X, Y
    print("[DATA] memmap cache ready.")
    return x_npy, y_npy


class WindowDataset(Dataset):
    """
    Minimal dataset for:
      X: (N, L, Din)
      Y: (N, H, Dout)
    Both loaded via memmap-backed .npy.
    """

    def __init__(self, x_npy: Path, y_npy: Path, indices: np.ndarray):
        self.X = np.load(x_npy, mmap_mode="r")
        self.Y = np.load(y_npy, mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"X and Y N mismatch: {self.X.shape[0]} vs {self.Y.shape[0]}")

        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-ordered split by contiguous indices to avoid leakage.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)
    return train_idx, val_idx, test_idx


def build_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_npy, y_npy = _maybe_build_memmap_cache(cfg.data_dir)

    # Determine N
    X = np.load(x_npy, mmap_mode="r")
    n = int(X.shape[0])
    del X

    train_idx, val_idx, test_idx = _split_indices(n, cfg.train_ratio, cfg.val_ratio)

    train_ds = WindowDataset(x_npy, y_npy, train_idx)
    val_ds = WindowDataset(x_npy, y_npy, val_idx)
    test_ds = WindowDataset(x_npy, y_npy, test_idx)

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        )

    train_loader = _make_loader(train_ds, shuffle=True)   # shuffle within train chunk is OK
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    print(f"[DATA] N={n}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader
