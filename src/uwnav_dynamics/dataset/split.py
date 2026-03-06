from __future__ import annotations

"""
Deterministic dataset split helpers.

These utilities persist the exact train/val/test window indices so training and
evaluation can share the same subset boundaries without recomputing them.
"""

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


SplitIndices = Dict[str, np.ndarray]


def _parse_ratios(ratios: Mapping[str, float] | Sequence[float]) -> tuple[float, float, float]:
    """Normalize mapping/sequence ratio inputs to an explicit train/val/test triple."""
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
    Build deterministic disjoint split indices by a seeded permutation.

    Split sizes follow floor strategy:
      n_train = int(n * train_ratio)
      n_val   = int(n * val_ratio)
      n_test  = n - n_train - n_val
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    tr, va, _te = _parse_ratios(ratios)

    n_train = int(n * tr)
    n_val = int(n * va)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_test < 0:
        raise ValueError(f"invalid split sizes: train={n_train}, val={n_val}, test={n_test}, n={n}")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n).astype(np.int64, copy=False)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def save_split_indices(path: str | Path, indices: SplitIndices) -> Path:
    """Persist split indices as a compact `.npz` artifact under the run directory."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        train=np.asarray(indices["train"], dtype=np.int64),
        val=np.asarray(indices["val"], dtype=np.int64),
        test=np.asarray(indices["test"], dtype=np.int64),
    )
    return p


def load_split_indices(path: str | Path) -> SplitIndices:
    """Load previously persisted split indices for exact train/eval reuse."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"split indices not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        out: SplitIndices = {
            "train": np.asarray(z["train"], dtype=np.int64),
            "val": np.asarray(z["val"], dtype=np.int64),
            "test": np.asarray(z["test"], dtype=np.int64),
        }
    return out
