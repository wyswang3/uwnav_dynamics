"""
模块名称：数据集划分工具

模块职责：
负责生成并持久化滑窗样本的 train/val/test 划分索引，
保证训练与评估共享完全一致的样本边界。

主要功能：
1. 规范化 `train/val/test` ratio 输入。
2. 提供兼容旧链路的窗口索引连续划分 helper。
3. 提供基于原始时间轴的 purged contiguous split，避免滑窗边界泄漏。
4. 读写 `split_indices.npz`，供 train/eval 共用。

数据流：
window count + split ratios
    ↓
make_split_indices() / make_purged_split_indices()
    ↓
split_indices.npz
    ↓
train / eval 读取同一份索引 artifact

依赖模块：
- numpy
- pathlib

备注：
- `make_split_indices()` 保留旧的窗口索引连续切分语义，供兼容链路使用。
- 训练主链路应优先使用 `make_purged_split_indices()`，
  它会在原始时间轴上切分并丢弃跨边界窗口，从而避免泄漏。
- `seed` 只为兼容旧接口保留，在连续划分语义下不影响结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import warnings

import numpy as np


SplitIndices = Dict[str, np.ndarray]
LEGACY_SPLIT_STRATEGY = "contiguous_v1"
PURGED_SPLIT_STRATEGY = "contiguous_purged_v2"
DEFAULT_SPLIT_STRATEGY = LEGACY_SPLIT_STRATEGY


@dataclass(frozen=True)
class SplitBuildSummary:
    """记录一次 split 构造的最小审计摘要。"""
    strategy: str
    dropped_window_count: int = 0
    total_rows: int | None = None
    window_span: int | None = None
    train_raw_end: int | None = None
    val_raw_end: int | None = None


def _parse_ratios(ratios: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    """把 mapping 或 sequence 形式的比例输入规范成显式 train/val/test 三元组。"""
    if isinstance(ratios, Mapping):
        tr = float(ratios.get("train", 0.0))
        va = float(ratios.get("val", 0.0))
        te = float(ratios.get("test", 1.0 - tr - va))
    else:
        seq = list(ratios)
        if len(seq) == 2:
            tr, va = float(seq[0]), float(seq[1])
            te = 1.0 - tr - va
        elif len(seq) == 3:
            tr, va, te = float(seq[0]), float(seq[1]), float(seq[2])
        else:
            raise ValueError("ratios must be mapping(train/val[/test]) or sequence of len 2/3")

    if tr <= 0.0 or va < 0.0 or te < 0.0:
        raise ValueError(f"invalid ratios train/val/test={tr}/{va}/{te}")
    s = tr + va + te
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {s}")
    return tr, va, te


def make_split_indices(
    n: int,
    seed: int,
    ratios: Mapping[str, float] | Sequence[float],
) -> SplitIndices:
    """
    在窗口索引轴上构造确定性的连续 train/val/test 划分。

    设计原因：
    - 对滑窗任务而言，随机打散并不天然等于“无泄漏”，因为相邻窗口仍可能高度重叠。
    - 该 helper 保留历史“窗口索引轴 contiguous split”语义。
    - 训练主链路若要严格避免滑窗边界泄漏，应改用 `make_purged_split_indices()`。
    - `seed` 仅为兼容旧调用方保留，在该策略下不会改变结果。
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    tr, va, _te = _parse_ratios(ratios)
    if int(seed) != 0:
        warnings.warn(
            f"split seed is ignored by split_strategy={LEGACY_SPLIT_STRATEGY!r}; "
            "seed is kept only for API compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_train = int(n * tr)
    n_val = int(n * va)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_test < 0:
        raise ValueError(f"invalid split sizes: train={n_train}, val={n_val}, test={n_test}, n={n}")

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def make_purged_split_indices(
    *,
    window_start_indices: Sequence[int] | np.ndarray,
    total_rows: int,
    window_span: int,
    seed: int,
    ratios: Mapping[str, float] | Sequence[float],
) -> tuple[SplitIndices, SplitBuildSummary]:
    """
    在原始时间轴上做连续切分，并主动丢弃跨 split 边界的滑窗。

    规则：
    - 先在原始 row 轴上按照 ratio 划出 train / val / test 三段。
    - 仅保留“完整窗口跨度完全落在该时间段内”的窗口。
    - 任何跨边界窗口都会被 purge 掉，避免历史窗/预测窗与相邻 split 共享原始样本。
    """
    tr, va, _te = _parse_ratios(ratios)
    if int(seed) != 0:
        warnings.warn(
            f"split seed is ignored by split_strategy={PURGED_SPLIT_STRATEGY!r}; "
            "seed is kept only for API compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )

    idx0 = np.asarray(window_start_indices, dtype=np.int64).reshape(-1)
    n_total = int(idx0.shape[0])
    if n_total <= 0:
        raise ValueError("window_start_indices must be non-empty")

    total_rows_i = int(total_rows)
    window_span_i = int(window_span)
    if total_rows_i <= 0:
        raise ValueError(f"total_rows must be positive, got {total_rows_i}")
    if window_span_i <= 0:
        raise ValueError(f"window_span must be positive, got {window_span_i}")

    train_raw_end = int(total_rows_i * tr)
    val_raw_end = int(total_rows_i * (tr + va))
    if train_raw_end < window_span_i:
        raise ValueError(
            "train raw slice is too short for one full window: "
            f"train_raw_end={train_raw_end}, window_span={window_span_i}"
        )
    if val_raw_end - train_raw_end < window_span_i:
        raise ValueError(
            "val raw slice is too short for one full window: "
            f"train_raw_end={train_raw_end}, val_raw_end={val_raw_end}, window_span={window_span_i}"
        )
    if total_rows_i - val_raw_end < window_span_i:
        raise ValueError(
            "test raw slice is too short for one full window: "
            f"val_raw_end={val_raw_end}, total_rows={total_rows_i}, window_span={window_span_i}"
        )

    window_end = idx0 + window_span_i
    if np.any(idx0 < 0) or np.any(window_end > total_rows_i):
        raise ValueError(
            "window_start_indices contain invalid span outside raw row range: "
            f"total_rows={total_rows_i}, window_span={window_span_i}"
        )

    train_mask = window_end <= train_raw_end
    val_mask = (idx0 >= train_raw_end) & (window_end <= val_raw_end)
    test_mask = idx0 >= val_raw_end

    train_idx = np.flatnonzero(train_mask).astype(np.int64, copy=False)
    val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
    test_idx = np.flatnonzero(test_mask).astype(np.int64, copy=False)

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "purged split produced empty subset: "
            f"train={train_idx.size}, val={val_idx.size}, test={test_idx.size}, "
            f"total_rows={total_rows_i}, window_span={window_span_i}"
        )

    kept = train_idx.size + val_idx.size + test_idx.size
    summary = SplitBuildSummary(
        strategy=PURGED_SPLIT_STRATEGY,
        dropped_window_count=int(n_total - kept),
        total_rows=total_rows_i,
        window_span=window_span_i,
        train_raw_end=train_raw_end,
        val_raw_end=val_raw_end,
    )
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }, summary


def save_split_indices(
    path: str | Path,
    indices: SplitIndices,
    *,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """把 split 索引与最小策略标记一起写成紧凑的 `.npz` artifact。"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "train": np.asarray(indices["train"], dtype=np.int64),
        "val": np.asarray(indices["val"], dtype=np.int64),
        "test": np.asarray(indices["test"], dtype=np.int64),
        "split_strategy": np.asarray(str(split_strategy)),
    }
    if metadata is not None:
        for key, value in metadata.items():
            if value is None:
                continue
            payload[str(key)] = np.asarray(value)
    np.savez_compressed(p, **payload)
    return p


def load_split_indices(path: str | Path) -> SplitIndices:
    """读取既有 split artifact，并对缺失或不一致的策略元信息给出兼容告警。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"split indices not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        if "split_strategy" not in z:
            warnings.warn(
                f"legacy split artifact without split_strategy metadata: {p}. "
                "It remains loadable, but its semantics may predate the current "
                f"'{DEFAULT_SPLIT_STRATEGY}' contract.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            split_strategy = str(np.asarray(z["split_strategy"]).item())
            if split_strategy not in {LEGACY_SPLIT_STRATEGY, PURGED_SPLIT_STRATEGY}:
                warnings.warn(
                    f"split artifact {p} uses split_strategy={split_strategy!r}; "
                    f"known strategies are {LEGACY_SPLIT_STRATEGY!r} / {PURGED_SPLIT_STRATEGY!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        out: SplitIndices = {
            "train": np.asarray(z["train"], dtype=np.int64),
            "val": np.asarray(z["val"], dtype=np.int64),
            "test": np.asarray(z["test"], dtype=np.int64),
        }
    return out
