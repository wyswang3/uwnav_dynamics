"""
模块名称：遗留训练数据加载器

模块职责：
提供兼容旧训练路径的最小 DataLoader 构建能力，
基于 `features.npz / labels.npz` 或缓存的 `X.npy / Y.npy` 生成 train/val/test loader。

主要功能：
1. 将 `features.npz / labels.npz` 转换为便于 memmap 读取的 `X.npy / Y.npy`。
2. 基于 canonical split helper 构造连续 train/val/test 划分。
3. 为旧训练入口提供 `WindowDataset` 与 `build_loaders()` 兼容层。

数据流：
data_dir/features.npz + labels.npz
    ↓
_maybe_build_memmap_cache()
    ↓
X.npy + Y.npy
    ↓
WindowDataset / DataLoader
    ↓
legacy train helper

依赖模块：
- numpy
- torch
- uwnav_dynamics.dataset.split

备注：
- 正式训练链路优先使用 `train.data_pipeline.prepare_train_data()`。
- 本模块保留的主要目的是兼容历史调用，而不是继续扩展新语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from uwnav_dynamics.dataset.split import make_split_indices


@dataclass(frozen=True)
class DataConfig:
    """遗留 DataLoader 构建路径需要的最小数据配置。"""
    data_dir: Path
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 0


def _maybe_build_memmap_cache(data_dir: Path) -> Tuple[Path, Path]:
    """
    把 `features.npz/labels.npz` 转成 `X.npy/Y.npy`，便于用 memmap 读取。

    预期输入：
    - `data_dir/features.npz`：包含 `X`
    - `data_dir/labels.npz`：包含 `Y`

    输出：
    - `data_dir/X.npy`
    - `data_dir/Y.npy`
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
    遗留训练路径使用的最小窗口数据集。

    数据约定：
    - `X: (N, L, Din)`
    - `Y: (N, H, Dout)`

    两者都通过 memmap `.npy` 读取，避免初始化时把全量数组复制进内存。
    """

    def __init__(self, x_npy: Path, y_npy: Path, indices: np.ndarray):
        """绑定 memmap 数组与样本索引，不在初始化阶段复制完整数据。"""
        self.X = np.load(x_npy, mmap_mode="r")
        self.Y = np.load(y_npy, mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"X and Y N mismatch: {self.X.shape[0]} vs {self.Y.shape[0]}")

        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        """返回当前 split 下可见窗口数。"""
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        """按索引读取单个窗口，并转成训练侧期望的 float32 tensor。"""
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def build_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    为旧训练链路构造 train/val/test DataLoader。

    该路径只保留兼容用途：
    - 它已经复用 canonical split helper，避免与正式链路的切分语义漂移。
    - 但它仍不会把 split/scaler artifact 持久化到 run 目录。

    正式 train/eval 流程应优先使用 `train.data_pipeline.prepare_train_data()`。
    """
    warnings.warn(
        "train.data.build_loaders is a legacy helper and does not persist "
        "run-scoped split/scaler artifacts. Prefer train.data_pipeline.prepare_train_data.",
        DeprecationWarning,
        stacklevel=2,
    )
    x_npy, y_npy = _maybe_build_memmap_cache(cfg.data_dir)

    # Determine N
    X = np.load(x_npy, mmap_mode="r")
    n = int(X.shape[0])
    del X

    split_indices = make_split_indices(
        n=n,
        seed=int(cfg.seed),
        ratios={
            "train": float(cfg.train_ratio),
            "val": float(cfg.val_ratio),
            "test": float(1.0 - cfg.train_ratio - cfg.val_ratio),
        },
    )
    train_idx = np.asarray(split_indices["train"], dtype=np.int64)
    val_idx = np.asarray(split_indices["val"], dtype=np.int64)
    test_idx = np.asarray(split_indices["test"], dtype=np.int64)

    train_ds = WindowDataset(x_npy, y_npy, train_idx)
    val_ds = WindowDataset(x_npy, y_npy, val_idx)
    test_ds = WindowDataset(x_npy, y_npy, test_idx)

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        """统一构造 train/val/test DataLoader，保持参数收口在单处。"""
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
