from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True

    # split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 0

    # loader stability / speed
    persistent_workers: bool = True
    prefetch_factor: int = 2
    make_writable_copy: bool = True  # 推荐 True：消除 non-writable warning


# =============================================================================
# Cache builder
# =============================================================================
def _maybe_build_memmap_cache(data_dir: Path) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """
    Convert features.npz/labels.npz to X.npy/Y.npy (and optional XM.npy/YM.npy) for memmap reading.

    Produces:
      - X.npy, Y.npy
      - XM.npy (optional), YM.npy (optional)
    """
    data_dir = Path(data_dir)
    x_npy  = data_dir / "X.npy"
    y_npy  = data_dir / "Y.npy"
    xm_npy = data_dir / "XM.npy"
    ym_npy = data_dir / "YM.npy"

    feat_npz = data_dir / "features.npz"
    lab_npz  = data_dir / "labels.npz"
    if not feat_npz.exists():
        raise FileNotFoundError(f"Missing: {feat_npz}")
    if not lab_npz.exists():
        raise FileNotFoundError(f"Missing: {lab_npz}")

    # 如果已经有 cache（包含 X/Y），就尝试补齐 mask cache（如果缺）
    need_build_xy = not (x_npy.exists() and y_npy.exists())
    need_build_m  = not (xm_npy.exists() and ym_npy.exists())

    if (not need_build_xy) and (not need_build_m):
        return x_npy, y_npy, xm_npy, ym_npy

    print("[DATA] building memmap cache ...")

    # ---- load features ----
    with np.load(feat_npz, allow_pickle=False) as z:
        if "X" not in z:
            raise KeyError(f"'X' not found in {feat_npz}")
        X = z["X"].astype(np.float32, copy=False)

        # 兼容旧/新命名：M 或 X_mask
        XM = None
        if "M" in z:
            XM = z["M"]
        elif "X_mask" in z:
            XM = z["X_mask"]

        if XM is not None:
            XM = XM.astype(np.uint8, copy=False)

    # ---- load labels ----
    with np.load(lab_npz, allow_pickle=False) as z:
        if "Y" not in z:
            raise KeyError(f"'Y' not found in {lab_npz}")
        Y = z["Y"].astype(np.float32, copy=False)

        # 兼容旧/新命名：M 或 Y_mask
        YM = None
        if "M" in z:
            YM = z["M"]
        elif "Y_mask" in z:
            YM = z["Y_mask"]

        if YM is not None:
            YM = YM.astype(np.uint8, copy=False)

    # ---- write X/Y ----
    if need_build_xy:
        print(f"[DATA] saving: {x_npy} shape={X.shape} dtype={X.dtype}")
        np.save(x_npy, X)
        print(f"[DATA] saving: {y_npy} shape={Y.shape} dtype={Y.dtype}")
        np.save(y_npy, Y)

    # ---- write masks (optional) ----
    xm_path_out: Optional[Path] = None
    ym_path_out: Optional[Path] = None

    if XM is not None:
        if not xm_npy.exists():
            print(f"[DATA] saving: {xm_npy} shape={XM.shape} dtype={XM.dtype}")
            np.save(xm_npy, XM)
        xm_path_out = xm_npy

    if YM is not None:
        if not ym_npy.exists():
            print(f"[DATA] saving: {ym_npy} shape={YM.shape} dtype={YM.dtype}")
            np.save(ym_npy, YM)
        ym_path_out = ym_npy

    del X, Y, XM, YM
    print("[DATA] memmap cache ready.")
    return x_npy, y_npy, xm_path_out, ym_path_out


# =============================================================================
# Dataset
# =============================================================================
class WindowDataset(Dataset):
    """
    Returns dict batch to support async supervision:
      {
        "X":  (L,Din) float32,
        "Y":  (H,Dout) float32,
        "XM": (L,Din) float32  (1=valid, 0=missing)  [always tensor],
        "YM": (H,Dout) float32  (1=supervised, 0=missing) [always tensor],
      }

    Key rule: NEVER return None (default_collate can't handle None).
    If mask files don't exist -> return all-ones masks.
    """
    def __init__(
        self,
        x_npy: Path,
        y_npy: Path,
        indices: np.ndarray,
        *,
        xm_npy: Optional[Path] = None,
        ym_npy: Optional[Path] = None,
        make_writable_copy: bool = True,
    ):
        self.X = np.load(Path(x_npy), mmap_mode="r")
        self.Y = np.load(Path(y_npy), mmap_mode="r")

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"X and Y N mismatch: {self.X.shape[0]} vs {self.Y.shape[0]}")

        self.XM = np.load(Path(xm_npy), mmap_mode="r") if xm_npy is not None else None
        self.YM = np.load(Path(ym_npy), mmap_mode="r") if ym_npy is not None else None

        if self.XM is not None and self.XM.shape != self.X.shape:
            raise ValueError(f"XM shape mismatch: XM{self.XM.shape} vs X{self.X.shape}")
        if self.YM is not None and self.YM.shape != self.Y.shape:
            raise ValueError(f"YM shape mismatch: YM{self.YM.shape} vs Y{self.Y.shape}")

        self.indices = indices.astype(np.int64, copy=False)
        self.make_writable_copy = bool(make_writable_copy)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])

        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.float32)

        # mask：没有就补全 1
        if self.XM is None:
            xm = np.ones_like(x, dtype=np.uint8)
        else:
            xm = np.asarray(self.XM[idx], dtype=np.uint8)

        if self.YM is None:
            ym = np.ones_like(y, dtype=np.uint8)
        else:
            ym = np.asarray(self.YM[idx], dtype=np.uint8)

        if self.make_writable_copy:
            x  = np.array(x,  dtype=np.float32, copy=True)
            y  = np.array(y,  dtype=np.float32, copy=True)
            xm = np.array(xm, dtype=np.uint8,   copy=True)
            ym = np.array(ym, dtype=np.uint8,   copy=True)

        return {
            "X": torch.from_numpy(x),
            "Y": torch.from_numpy(y),
            "XM": torch.from_numpy(xm),
            "YM": torch.from_numpy(ym),
        }

# =============================================================================
# Split
# =============================================================================

def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# =============================================================================
# Loader
# =============================================================================

def build_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_npy, y_npy, xm_npy, ym_npy = _maybe_build_memmap_cache(cfg.data_dir)
    assert x_npy is not None and y_npy is not None

    # Determine N without loading into RAM
    X = np.load(x_npy, mmap_mode="r")
    n = int(X.shape[0])
    del X

    train_idx, val_idx, test_idx = _split_indices(n, cfg.train_ratio, cfg.val_ratio)

    train_ds = WindowDataset(x_npy, y_npy, train_idx, xm_npy=xm_npy, ym_npy=ym_npy, make_writable_copy=cfg.make_writable_copy)
    val_ds   = WindowDataset(x_npy, y_npy, val_idx,   xm_npy=xm_npy, ym_npy=ym_npy, make_writable_copy=cfg.make_writable_copy)
    test_ds  = WindowDataset(x_npy, y_npy, test_idx,  xm_npy=xm_npy, ym_npy=ym_npy, make_writable_copy=cfg.make_writable_copy)
    mask_x = "yes" if xm_npy is not None else "no"
    mask_y = "yes" if ym_npy is not None else "no"
    print(f"[DATA] N={n}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  mask_X={mask_x}  mask_Y={mask_y}  copy={'ndarray' if cfg.make_writable_copy else 'memmap'}")

    g = torch.Generator()
    g.manual_seed(int(cfg.seed))

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        kw = dict(
            dataset=ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            generator=g if shuffle else None,
        )
        if cfg.num_workers > 0:
            kw.update(
                worker_init_fn=_seed_worker,
                persistent_workers=bool(cfg.persistent_workers),
                prefetch_factor=int(cfg.prefetch_factor),
            )
        return DataLoader(**kw)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    return train_loader, val_loader, test_loader
