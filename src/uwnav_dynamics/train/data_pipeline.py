"""
模块名称：训练数据准备管线

模块职责：
负责训练阶段非模型部分的主数据契约：
加载滑窗数据、复用或创建 split/scaler artifact、构造 DataLoader，
并在 PR5 中将 supervision mask 接入训练 batch。

主要功能：
1. 从 `features.npz / labels.npz` 读取 `X / Y` 与原始 mask artifact。
2. 优先使用原始时间轴上的 purged split，避免滑窗边界泄漏，并复用 split/scaler 真源路径。
3. 基于 `dvl_mask + semantic output layout` 构造与 `Y` 对齐的 `target_mask`。
4. 对经过 scaler 的输入特征做非有限值清洗，避免稀疏辅助通道中的 `NaN` 直接进入 RNN。
5. 对历史 artifact 中“velocity mask=true 但目标仍为 NaN”的情况做保守降级。
6. 对仅发生在 `target_mask=False` 位置的目标侧非有限值做兼容清洗。
7. 产出 `(X, Y, target_mask)` DataLoader，供训练主路径直接消费。

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
输入 X 非有限值清洗（置 0，对应 train 均值）
    ↓
velocity mask 与有限 target 的保守交集
    ↓
masked-out Y 非有限值兼容清洗
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
- 对输入 `X` 的非有限值清洗只发生在训练消费端，不修改原始 dataset artifact。
- 对 velocity 语义组允许“mask=true 但目标非有限”的历史 artifact 兼容降级；
  该降级只发生在 runtime `target_mask` 构造阶段，不回写 dataset。
- 对目标 `Y` 仅兼容清洗那些已经被 `target_mask=False` 排除的非有限值；
  活跃监督位置若仍存在 `NaN/Inf`，继续显式报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from uwnav_dynamics.dataset.normalize import (
    fit_scaler,
    load_scaler,
    save_scaler,
    transform,
)
from uwnav_dynamics.dataset.split import (
    LEGACY_SPLIT_STRATEGY,
    SplitBuildSummary,
    load_split_indices,
    make_split_indices,
    make_purged_split_indices,
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
    refine_velocity_target_mask_with_finite_targets,
)
from uwnav_dynamics.train.data import DataConfig
from uwnav_dynamics.train.runtime import make_dataloader_worker_init_fn, make_torch_generator


@dataclass(frozen=True)
class PreparedTrainData:
    """训练阶段准备好的 DataLoader 与相关统计信息。"""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_total: int
    split_sizes: Dict[str, int]
    mask_shapes: Dict[str, tuple[int, ...]]
    split_strategy: str
    dropped_window_count: int


@dataclass(frozen=True)
class _LoadedDatasetArrays:
    X: np.ndarray
    Y: np.ndarray
    target_mask: np.ndarray
    semantic_layout: SemanticOutputLayout
    raw_mask_source: str
    mask_shapes: Dict[str, tuple[int, ...]]
    window_start_indices: np.ndarray | None
    window_span: int | None
    total_rows: int | None


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
        idx0 = np.asarray(z_feat["idx0"], dtype=np.int64) if "idx0" in z_feat else None
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

    # 运行时监督掩码不直接相信历史 artifact 的“列名约定”，
    # 而是先恢复 semantic layout，再把稀疏 dvl_mask 扩展到 `(N,H,D)`。
    semantic_layout = resolve_semantic_output_layout(dout=Y.shape[-1], target_cols=target_cols)
    target_mask, raw_mask_source = _build_target_mask(
        dvl_mask=dvl_mask,
        semantic_layout=semantic_layout,
        target_shape=tuple(int(v) for v in Y.shape),
    )
    target_mask, downgraded_velocity_count = refine_velocity_target_mask_with_finite_targets(
        target_mask,
        Y,
        semantic_layout,
    )
    if downgraded_velocity_count > 0:
        print(
            "[DATA] downgraded velocity supervision due to non-finite targets: "
            f"{downgraded_velocity_count} elements"
        )
    mask_shapes["target_mask"] = tuple(target_mask.shape)

    meta_path = data_dir / "meta.yaml"
    window_span: int | None = None
    total_rows: int | None = None
    if idx0 is not None and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        if isinstance(meta, dict):
            hist_len = meta.get("hist_len")
            pred_len = meta.get("pred_len")
            total_rows = meta.get("n_rows_base")
            if hist_len is not None and pred_len is not None and total_rows is not None:
                window_span = int(hist_len) + int(pred_len)
                total_rows = int(total_rows)
    elif idx0 is not None:
        print("[DATA] idx0 found but meta.yaml missing; fallback to legacy split semantics.")

    return _LoadedDatasetArrays(
        X=X,
        Y=Y,
        target_mask=target_mask,
        semantic_layout=semantic_layout,
        raw_mask_source=raw_mask_source,
        mask_shapes=mask_shapes,
        window_start_indices=idx0,
        window_span=window_span,
        total_rows=total_rows,
    )


def _validate_indices(
    indices: Dict[str, np.ndarray],
    n: int,
    *,
    window_start_indices: np.ndarray | None = None,
    window_span: int | None = None,
    allow_dropped: bool = False,
) -> None:
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
    if (not allow_dropped) and merged.size != n:
        raise ValueError(f"split indices do not cover all samples: {merged.size} vs {n}")
    if allow_dropped and merged.size > n:
        raise ValueError(f"split indices cover more samples than available: {merged.size} vs {n}")
    if window_start_indices is not None and window_span is not None:
        _validate_cross_split_window_overlap(
            indices,
            window_start_indices=np.asarray(window_start_indices, dtype=np.int64),
            window_span=int(window_span),
        )


def _validate_cross_split_window_overlap(
    indices: Dict[str, np.ndarray],
    *,
    window_start_indices: np.ndarray,
    window_span: int,
) -> None:
    intervals: list[tuple[int, int, str]] = []
    for split_name in ("train", "val", "test"):
        for idx in np.asarray(indices[split_name], dtype=np.int64):
            start = int(window_start_indices[int(idx)])
            intervals.append((start, start + int(window_span), split_name))
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))

    active_end: dict[str, int] = {}
    for start, end, split_name in intervals:
        expired = [name for name, other_end in active_end.items() if other_end <= start]
        for name in expired:
            del active_end[name]
        for other_split, other_end in active_end.items():
            if other_split != split_name and other_end > start:
                raise ValueError(
                    "split indices overlap on raw timeline after window expansion: "
                    f"{other_split} end={other_end} > {split_name} start={start}"
                )
        active_end[split_name] = max(active_end.get(split_name, end), end)


def _load_saved_split_summary(path: Path) -> SplitBuildSummary:
    with np.load(path, allow_pickle=False) as z:
        strategy = (
            str(np.asarray(z["split_strategy"]).item())
            if "split_strategy" in z
            else LEGACY_SPLIT_STRATEGY
        )
        dropped = int(np.asarray(z["dropped_window_count"]).item()) if "dropped_window_count" in z else 0
        total_rows = int(np.asarray(z["total_rows"]).item()) if "total_rows" in z else None
        window_span = int(np.asarray(z["window_span"]).item()) if "window_span" in z else None
        train_raw_end = int(np.asarray(z["train_raw_end"]).item()) if "train_raw_end" in z else None
        val_raw_end = int(np.asarray(z["val_raw_end"]).item()) if "val_raw_end" in z else None
    return SplitBuildSummary(
        strategy=strategy,
        dropped_window_count=dropped,
        total_rows=total_rows,
        window_span=window_span,
        train_raw_end=train_raw_end,
        val_raw_end=val_raw_end,
    )


def _sanitize_scaled_inputs(x: np.ndarray, *, split_name: str) -> np.ndarray:
    """
    对 scaler 之后的输入特征做最小非有限值清洗。

    设计原因：
    - 原始 `features.npz` 中允许保留稀疏辅助通道的 NaN（例如 power 缺测段）。
    - `fit_scaler()` 会忽略 NaN 拟合统计量，但 `transform()` 会保留 NaN。
    - 若直接把这些 NaN 喂给 LSTM，训练首个 batch 就会产出 NaN loss。

    这里在消费端把非有限值统一置为 0.0，语义上等价于“回到 z-score 后的 train 均值”，
    同时不改变原始 dataset artifact。
    """
    bad = ~np.isfinite(x)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return x

    x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    print(
        "[DATA] sanitized non-finite X on "
        f"{split_name} split: replaced {bad_count} values with 0.0 after z-score transform"
    )
    return x_safe.astype(np.float32, copy=False)


def _sanitize_targets_with_runtime_mask(
    y: np.ndarray,
    target_mask: np.ndarray,
    *,
    split_name: str,
) -> np.ndarray:
    """
    对训练目标 `Y` 做最小兼容清洗。

    规则：
    - 若非有限值只出现在 `target_mask=False` 的位置，则统一置为 0.0。
      这类位置不会进入主 state masked NLL，也不会进入 DVL auxiliary loss，
      因此允许作为历史/stale artifact 的兼容修复。
    - 若非有限值出现在活跃监督位置（`target_mask=True`），则立即显式报错。

    设计原因：
    - PR5 明确规定 runtime 监督有效性真源是 batch `target_mask`。
    - 部分历史 dataset artifact 可能在 velocity 稀疏监督位置保留了 NaN，
      但这些位置本就不应参与 loss 计算。
    """
    if y.shape != target_mask.shape:
        raise ValueError(
            f"Y / target_mask shape mismatch on {split_name} split: "
            f"Y={y.shape}, target_mask={target_mask.shape}"
        )

    bad = ~np.isfinite(y)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return y

    active_bad = bad & target_mask
    active_bad_count = int(np.count_nonzero(active_bad))
    if active_bad_count > 0:
        raise ValueError(
            "non-finite values found in active Y supervision after scaling on "
            f"{split_name} split: active_count={active_bad_count} total_bad={bad_count}"
        )

    y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    print(
        "[DATA] sanitized non-finite Y on "
        f"{split_name} split: replaced {bad_count} masked-out values with 0.0"
    )
    return y_safe.astype(np.float32, copy=False)


def _make_loader(
    X: np.ndarray,
    Y: np.ndarray,
    target_mask: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
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
        worker_init_fn=make_dataloader_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )


def prepare_train_data(cfg: DataConfig, run_layout: RunLayout) -> PreparedTrainData:
    """按运行目录契约准备训练/验证/测试数据与 scaler/split artifact。"""
    loaded = _load_dataset_arrays(Path(cfg.data_dir))
    X_all = loaded.X
    Y_all = loaded.Y
    target_mask_all = loaded.target_mask
    n_total = int(X_all.shape[0])
    if loaded.mask_shapes:
        print(f"[DATA] found mask tensors: {loaded.mask_shapes}")
        print(f"[DATA] target_mask source={loaded.raw_mask_source}")

    # split artifact 是 train / eval 共享的真源；已有则复用，没有才创建。
    split_path = run_layout.split_indices_path
    ratios = {
        "train": float(cfg.train_ratio),
        "val": float(cfg.val_ratio),
        "test": float(1.0 - cfg.train_ratio - cfg.val_ratio),
    }

    def _build_split() -> tuple[Dict[str, np.ndarray], SplitBuildSummary]:
        if (
            loaded.window_start_indices is not None
            and loaded.window_span is not None
            and loaded.total_rows is not None
        ):
            built_indices, built_summary = make_purged_split_indices(
                window_start_indices=loaded.window_start_indices,
                total_rows=int(loaded.total_rows),
                window_span=int(loaded.window_span),
                seed=int(cfg.seed),
                ratios=ratios,
            )
            save_split_indices(
                split_path,
                built_indices,
                split_strategy=built_summary.strategy,
                metadata={
                    "dropped_window_count": built_summary.dropped_window_count,
                    "total_rows": built_summary.total_rows,
                    "window_span": built_summary.window_span,
                    "train_raw_end": built_summary.train_raw_end,
                    "val_raw_end": built_summary.val_raw_end,
                },
            )
            return built_indices, built_summary

        built_indices = make_split_indices(
            n=n_total,
            seed=int(cfg.seed),
            ratios=ratios,
        )
        built_summary = SplitBuildSummary(strategy=LEGACY_SPLIT_STRATEGY, dropped_window_count=0)
        save_split_indices(
            split_path,
            built_indices,
            split_strategy=LEGACY_SPLIT_STRATEGY,
        )
        return built_indices, built_summary

    if split_path.exists():
        split_indices = load_split_indices(split_path)
        split_summary = _load_saved_split_summary(split_path)
        if (
            split_summary.strategy == LEGACY_SPLIT_STRATEGY
            and loaded.window_start_indices is not None
            and loaded.window_span is not None
            and loaded.total_rows is not None
        ):
            print(f"[SPLIT] upgrading legacy split artifact to purged strategy: {split_path}")
            split_indices, split_summary = _build_split()
        else:
            print(f"[SPLIT] reuse: {split_path}")
    else:
        split_indices, split_summary = _build_split()
        print(
            f"[SPLIT] created: {split_path} strategy={split_summary.strategy} "
            f"dropped={split_summary.dropped_window_count}"
        )

    _validate_indices(
        split_indices,
        n_total,
        window_start_indices=loaded.window_start_indices,
        window_span=loaded.window_span,
        allow_dropped=(split_summary.dropped_window_count > 0),
    )
    train_idx = np.asarray(split_indices["train"], dtype=np.int64)
    val_idx = np.asarray(split_indices["val"], dtype=np.int64)
    test_idx = np.asarray(split_indices["test"], dtype=np.int64)
    print(
        f"[SPLIT] N={n_total} train={train_idx.size} val={val_idx.size} test={test_idx.size} "
        f"strategy={split_summary.strategy} dropped={split_summary.dropped_window_count}"
    )

    # scaler 只允许在 train split 上拟合，避免把验证/测试统计量泄漏回训练阶段。
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

    mask_train = target_mask_all[train_idx]
    mask_val = target_mask_all[val_idx]
    mask_test = target_mask_all[test_idx]

    # 先做 split-specific transform，再做最小非有限值兜底；
    # 这样保留了“scaler 在 z-space 里工作”的语义，也避免 NaN 直接进模型。
    X_train = _sanitize_scaled_inputs(
        transform(X_all[train_idx], x_scaler).astype(np.float32, copy=False),
        split_name="train",
    )
    Y_train = _sanitize_targets_with_runtime_mask(
        transform(Y_all[train_idx], y_scaler).astype(np.float32, copy=False),
        mask_train,
        split_name="train",
    )
    X_val = _sanitize_scaled_inputs(
        transform(X_all[val_idx], x_scaler).astype(np.float32, copy=False),
        split_name="val",
    )
    Y_val = _sanitize_targets_with_runtime_mask(
        transform(Y_all[val_idx], y_scaler).astype(np.float32, copy=False),
        mask_val,
        split_name="val",
    )
    X_test = _sanitize_scaled_inputs(
        transform(X_all[test_idx], x_scaler).astype(np.float32, copy=False),
        split_name="test",
    )
    Y_test = _sanitize_targets_with_runtime_mask(
        transform(Y_all[test_idx], y_scaler).astype(np.float32, copy=False),
        mask_test,
        split_name="test",
    )

    # 三个 loader 的张量结构保持完全一致，差别只在 shuffle 和样本子集。
    train_loader = _make_loader(
        X_train,
        Y_train,
        mask_train,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=True,
        seed=int(cfg.seed),
    )
    val_loader = _make_loader(
        X_val,
        Y_val,
        mask_val,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=False,
        seed=int(cfg.seed) + 1,
    )
    test_loader = _make_loader(
        X_test,
        Y_test,
        mask_test,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        shuffle=False,
        seed=int(cfg.seed) + 2,
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
        split_strategy=split_summary.strategy,
        dropped_window_count=int(split_summary.dropped_window_count),
    )
