"""
模块名称：数据集划分工具

模块职责：
负责生成并持久化滑窗样本的 train/val/test 划分索引，
保证训练与评估共享完全一致的样本边界。

主要功能：
1. 规范化 `train/val/test` ratio 输入。
2. 生成当前仓库默认的连续时间划分索引。
3. 读写 `split_indices.npz`，供 train/eval 共用。

数据流：
window count + split ratios
    ↓
make_split_indices()
    ↓
split_indices.npz
    ↓
train / eval 读取同一份索引 artifact

依赖模块：
- numpy
- pathlib

备注：
- 当前 canonical 策略是 `contiguous_v1`，不是随机打乱划分。
- `seed` 只为兼容旧接口保留，在连续划分语义下不影响结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence
import warnings

import numpy as np


SplitIndices = Dict[str, np.ndarray]
DEFAULT_SPLIT_STRATEGY = "contiguous_v1"


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
    - 当前实验契约因此固定采用时间顺序连续切分。
    - `seed` 仅为兼容旧调用方保留，在该策略下不会改变结果。
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    tr, va, _te = _parse_ratios(ratios)
    if int(seed) != 0:
        warnings.warn(
            "split seed is ignored by split_strategy='contiguous_v1'; "
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


def save_split_indices(path: str | Path, indices: SplitIndices) -> Path:
    """把 split 索引与最小策略标记一起写成紧凑的 `.npz` artifact。"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        train=np.asarray(indices["train"], dtype=np.int64),
        val=np.asarray(indices["val"], dtype=np.int64),
        test=np.asarray(indices["test"], dtype=np.int64),
        split_strategy=np.asarray(DEFAULT_SPLIT_STRATEGY),
    )
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
            if split_strategy != DEFAULT_SPLIT_STRATEGY:
                warnings.warn(
                    f"split artifact {p} uses split_strategy={split_strategy!r}; "
                    f"current default is {DEFAULT_SPLIT_STRATEGY!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        out: SplitIndices = {
            "train": np.asarray(z["train"], dtype=np.int64),
            "val": np.asarray(z["val"], dtype=np.int64),
            "test": np.asarray(z["test"], dtype=np.int64),
        }
    return out
