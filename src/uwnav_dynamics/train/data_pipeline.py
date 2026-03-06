"""
模块名称：训练数据准备管线

模块职责：
负责训练阶段非模型部分的主数据契约：
加载滑窗数据、复用或创建 split/scaler artifact、构造 DataLoader，
并在 PR5 中将 supervision mask 接入训练 batch。

主要功能：
1. 从 `features.npz / labels.npz` 读取 `X / Y` 与原始 mask artifact。
2. 复用 PR2 的 split/scaler 真源路径，保证 train / eval 一致。
3. 基于 `dvl_mask + semantic output layout` 构造与 `Y` 对齐的 `target_mask`。
4. 产出 `(X, Y, target_mask)` DataLoader，供训练主路径直接消费。

数据流：
features.npz / labels.npz
    ↓
X / Y / dvl_mask / target_cols
    ↓
semantic output layout 校验
    ↓
target_mask:(N,H,D)
    ↓
split/scaler
    ↓
train/val/test DataLoader

依赖模块：
- numpy
- torch
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
- uwnav_dynamics.models.utils.semantic_output_layout
- uwnav_dynamics.supervision_mask

备注：
- runtime mask 的唯一执行真源是 batch 中的 `target_mask`。
- `meta.yaml` 只做记录，不参与训练运行时 mask 裁决。
"""

from __future__ import annotations

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
from uwnav_dynamics.models.utils.semantic_output_layout import (
    SemanticOutputLayout,
    resolve_semantic_output_layout,
)
from uwnav_dynamics.supervision_mask import (
    build_dense_target_mask,
    build_target_mask_from_dvl_mask,
)
from uwnav_dynamics.train.data import DataConfig


@dataclass(frozen=True)
class PreparedTrainData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_total: int
    split_sizes: Dict[str, int]
    mask_shapes: Dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class _LoadedDatasetArrays:
    X: np.ndarray
    Y: np.ndarray
    target_mask: np.ndarray
    semantic_layout: SemanticOutputLayout
    raw_mask_source: str
    mask_shapes: Dict[str, tuple[int, ...]]


def _extract_target_cols(label_npz: np.lib.npyio.NpzFile) -> tuple[str, ...] | None:
    if "target_cols" not in label_npz:
        return None
    target_cols = np.asarray(label_npz["target_cols"])
    if target_cols.ndim == 0:
        return (str(target_cols.item()),)
    return tuple(str(x) for x in target_cols.tolist())


def _build_target_mask(
    *,
    dvl_mask: np.ndarray | None,
    semantic_layout: SemanticOutputLayout,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    if dvl_mask is None:
        return build_dense_target_mask(target_shape), "implicit_all_true"
    return (
        build_target_mask_from_dvl_mask(dvl_mask, semantic_layout, target_shape=target_shape),
        "dvl_mask",
    )


def _load_dataset_arrays(data_dir: Path) -> _LoadedDatasetArrays:
    feat_npz = data_dir / "features.npz"
    lab_npz = data_dir / "labels.npz"
    if not feat_npz.exists():
        raise FileNotFoundError(f"Missing: {feat_npz}")
    if not lab_npz.exists():
        raise FileNotFoundError(f"Missing: {lab_npz}")

    mask_shapes: Dict[str, tuple[int, ...]] = {}
    with np.load(feat_npz, allow_pickle=False) as z_feat:
        if "X" not in z_feat:
            raise KeyError(f"'X' not found in {feat_npz}")
        X = np.asarray(z_feat["X"], dtype=np.float32)
        for k in ("dvl_mask_hist", "power_mask_hist"):
            if k in z_feat:
                mask_shapes[k] = tuple(np.asarray(z_feat[k]).shape)

    with np.load(lab_npz, allow_pickle=True) as z_lab:
        if "Y" not in z_lab:
            raise KeyError(f"'Y' not found in {lab_npz}")
        Y = np.asarray(z_lab["Y"], dtype=np.float32)
        dvl_mask = np.asarray(z_lab["dvl_mask"], dtype=bool) if "dvl_mask" in z_lab else None
        for k in ("dvl_mask", "power_mask"):
            if k in z_lab:
                mask_shapes[k] = tuple(np.asarray(z_lab[k]).shape)
        target_cols = _extract_target_cols(z_lab)

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expect X/Y to be 3D, got X={X.shape}, Y={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y window counts mismatch: {X.shape[0]} vs {Y.shape[0]}")

    semantic_layout = resolve_semantic_output_layout(dout=Y.shape[-1], target_cols=target_cols)
    target_mask, raw_mask_source = _build_target_mask(
        dvl_mask=dvl_mask,
        semantic_layout=semantic_layout,
        target_shape=tuple(int(v) for v in Y.shape),
    )
    mask_shapes["target_mask"] = tuple(target_mask.shape)

    return _LoadedDatasetArrays(
        X=X,
        Y=Y,
        target_mask=target_mask,
        semantic_layout=semantic_layout,
        raw_mask_source=raw_mask_source,
        mask_shapes=mask_shapes,
    )


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
    target_mask: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y),
        torch.from_numpy(target_mask.astype(np.bool_, copy=False)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def prepare_train_data(cfg: DataConfig, run_layout: RunLayout) -> PreparedTrainData:
    loaded = _load_dataset_arrays(Path(cfg.data_dir))
    X_all = loaded.X
    Y_all = loaded.Y
    target_mask_all = loaded.target_mask
    n_total = int(X_all.shape[0])
    if loaded.mask_shapes:
        print(f"[DATA] found mask tensors: {loaded.mask_shapes}")
        print(f"[DATA] target_mask source={loaded.raw_mask_source}")

    split_path = run_layout.split_indices_path
    if split_path.exists():
        split_indices = load_split_indices(split_path)
        print(f"[SPLIT] reuse: {split_path}")
    else:
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

    mask_train = target_mask_all[train_idx]
    mask_val = target_mask_all[val_idx]
    mask_test = target_mask_all[test_idx]

    train_loader = _make_loader(
        X_train,
        Y_train,
        mask_train,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=True,
    )
    val_loader = _make_loader(
        X_val,
        Y_val,
        mask_val,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=False,
    )
    test_loader = _make_loader(
        X_test,
        Y_test,
        mask_test,
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
        mask_shapes=loaded.mask_shapes,
    )
