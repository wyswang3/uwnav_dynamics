# src/uwnav_dynamics/preprocess/sliding_window.py
from __future__ import annotations

"""
滑动窗口数据集构造器（通用版）

目标（适配“异步监督学习”）：
  - 不要求所有传感器在每个时刻都有效（例如 DVL 稀疏）
  - 滑窗阶段尽量“保留窗口”，不要因为稀疏监督导致几乎没有样本
  - 对缺失值（NaN）不做强制过滤：交给下游 build_dataset / loss 做 mask

仍保留可选的窗口过滤：
  - valid_mask_col + min_valid_ratio：用于“窗口级别”剔除（可关闭）
  - drop_incomplete：是否丢弃尾部不足 L+H 的窗口（可关闭）
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any, Literal

import numpy as np
import pandas as pd


# =============================================================================
# 配置与结果数据结构
# =============================================================================

@dataclass
class SlidingWindowConfig:
    """
    滑动窗口构造配置（只负责时间维切片，不关心物理含义）。

    关键设计（为异步监督服务）：
      - min_valid_ratio 允许为 0.0：表示不做窗口过滤
      - drop_incomplete=False 时允许构造尾部窗口：
          * pad_mode="nan": 不足部分用 NaN 填充（推荐，配合下游 mask）
          * pad_mode="edge": 用最后一行重复填充（不推荐用于监督，但有时便于部署）
    """
    input_cols: Sequence[str]
    target_cols: Sequence[str]

    hist_len: int
    pred_len: int
    stride: int = 1

    valid_mask_col: Optional[str] = None
    min_valid_ratio: float = 1.0

    drop_incomplete: bool = True

    # only used when drop_incomplete=False
    pad_mode: Literal["nan", "edge"] = "nan"

    def total_span(self) -> int:
        return int(self.hist_len) + int(self.pred_len)

    def check_valid(self) -> None:
        if self.hist_len <= 0 or self.pred_len <= 0:
            raise ValueError(f"hist_len/pred_len must be positive, got L={self.hist_len}, H={self.pred_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if not self.input_cols:
            raise ValueError("input_cols must be non-empty")
        if not self.target_cols:
            raise ValueError("target_cols must be non-empty")
        # 允许 0.0：表示完全不筛窗
        if self.min_valid_ratio < 0.0 or self.min_valid_ratio > 1.0:
            raise ValueError(f"min_valid_ratio must be in [0,1], got {self.min_valid_ratio}")
        if self.pad_mode not in ("nan", "edge"):
            raise ValueError(f"pad_mode must be 'nan' or 'edge', got {self.pad_mode!r}")


@dataclass
class SlidingWindowResult:
    X: np.ndarray
    Y: np.ndarray
    t0: np.ndarray
    idx0: np.ndarray
    cfg: SlidingWindowConfig


# =============================================================================
# 内部工具
# =============================================================================

def _make_indices(n: int, cfg: SlidingWindowConfig) -> List[int]:
    """
    计算所有窗口的起始下标列表。
    - drop_incomplete=True : 仅生成能完整覆盖 L+H 的窗口
    - drop_incomplete=False: 允许生成直到 n-hist_len（预测不足部分后续 pad）
    """
    cfg.check_valid()
    span = cfg.total_span()
    idx_list: List[int] = []

    if cfg.drop_incomplete:
        max_start = n - span
    else:
        max_start = n - cfg.hist_len

    if max_start < 0:
        return []

    i = 0
    while i <= max_start:
        idx_list.append(i)
        i += cfg.stride
    return idx_list


def _window_slice_with_pad(
    src: np.ndarray,
    start: int,
    length: int,
    *,
    pad_mode: Literal["nan", "edge"],
) -> np.ndarray:
    """
    从 src[start:start+length] 取窗口；若越界则按 pad_mode 补齐到 length。
    src: (n, D)
    return: (length, D)
    """
    n = src.shape[0]
    end = start + length
    if start >= n:
        # 完全越界：返回全 NaN（或 edge 也只能 NaN）
        out = np.full((length, src.shape[1]), np.nan, dtype=float)
        return out

    if end <= n:
        return src[start:end]

    # 需要 pad
    part = src[start:n]
    need = end - n
    if pad_mode == "nan":
        pad = np.full((need, src.shape[1]), np.nan, dtype=float)
    else:  # "edge"
        last = src[n - 1:n]
        pad = np.repeat(last, repeats=need, axis=0)
    return np.concatenate([part, pad], axis=0)


# =============================================================================
# 核心 API
# =============================================================================

def make_sliding_windows(
    df: pd.DataFrame,
    time_col: str,
    cfg: SlidingWindowConfig,
) -> SlidingWindowResult:
    """
    从 DataFrame 构造监督学习窗口数据。

    异步监督友好策略：
      - 不要求窗口内所有步都有效（min_valid_ratio=0.0 即不筛窗）
      - 不要求未来 H 步都有 DVL（允许 Y 中出现 NaN）
      - drop_incomplete=False 时，允许尾部窗口，缺失部分 pad（默认 NaN）

    注意：
      - 这里不生成 mask；mask 在 build_dataset.py 里统一生成（X_mask/Y_mask）。
    """
    cfg.check_valid()

    if time_col not in df.columns:
        raise KeyError(f"time_col={time_col!r} not in DataFrame columns.")
    n = len(df)
    if n == 0:
        raise ValueError("Empty DataFrame for sliding window construction.")

    # 列存在性
    for c in cfg.input_cols:
        if c not in df.columns:
            raise KeyError(f"input column {c!r} not in DataFrame.")
    for c in cfg.target_cols:
        if c not in df.columns:
            raise KeyError(f"target column {c!r} not in DataFrame.")

    time_arr = df[time_col].to_numpy(dtype=float)
    X_source = df[list(cfg.input_cols)].to_numpy(dtype=float)
    Y_source = df[list(cfg.target_cols)].to_numpy(dtype=float)

    mask_arr: Optional[np.ndarray] = None
    if cfg.valid_mask_col is not None:
        if cfg.valid_mask_col not in df.columns:
            raise KeyError(f"valid_mask_col={cfg.valid_mask_col!r} not in DataFrame.")
        # 只用于窗口级过滤
        mask_arr = df[cfg.valid_mask_col].to_numpy(dtype=float)

    idx_candidates = _make_indices(n, cfg)
    if not idx_candidates:
        raise ValueError(
            f"No valid window start indices for n={n}, L={cfg.hist_len}, H={cfg.pred_len}, "
            f"drop_incomplete={cfg.drop_incomplete}"
        )

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    t0_list: List[float] = []
    idx0_list: List[int] = []

    n_drop_mask = 0

    for i0 in idx_candidates:
        i_hist_start = i0
        i_hist_end = i0 + cfg.hist_len
        i_pred_start = i_hist_end
        i_pred_end = i_hist_end + cfg.pred_len

        # 1) 可选：窗口级过滤（异步监督时通常关闭）
        if mask_arr is not None and cfg.min_valid_ratio > 0.0:
            # 过滤范围定义：覆盖 L+H（对于异步监督，通常你会把 min_valid_ratio=0）
            j_end = min(i_pred_end, n) if not cfg.drop_incomplete else i_pred_end
            if j_end > n:
                # drop_incomplete=True 时这里理论上不可能
                continue
            m_win = mask_arr[i_hist_start:j_end]
            valid = np.isfinite(m_win) & (m_win > 0.5)
            valid_ratio = float(valid.mean()) if valid.size > 0 else 0.0
            if valid_ratio < cfg.min_valid_ratio:
                n_drop_mask += 1
                continue

        # 2) 取 X / Y（必要时 pad）
        if cfg.drop_incomplete:
            # 必须完整
            if i_pred_end > n:
                continue
            X_win = X_source[i_hist_start:i_hist_end]
            Y_win = Y_source[i_pred_start:i_pred_end]
        else:
            X_win = _window_slice_with_pad(X_source, i_hist_start, cfg.hist_len, pad_mode=cfg.pad_mode)
            Y_win = _window_slice_with_pad(Y_source, i_pred_start, cfg.pred_len, pad_mode=cfg.pad_mode)

        # shape sanity
        if X_win.shape[0] != cfg.hist_len or Y_win.shape[0] != cfg.pred_len:
            # 理论上 pad 后不会发生
            continue

        X_list.append(X_win)
        Y_list.append(Y_win)
        t0_list.append(float(time_arr[i0]))
        idx0_list.append(i0)

    if len(X_list) == 0:
        raise RuntimeError(
            "Sliding window construction produced 0 samples. "
            "Check hist_len/pred_len/stride and whether drop_incomplete is too strict."
        )

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    t0 = np.asarray(t0_list, dtype=float)
    idx0 = np.asarray(idx0_list, dtype=int)

    print(
        f"[SLIDING] built {X.shape[0]} windows from n={n} rows; "
        f"L={cfg.hist_len}, H={cfg.pred_len}, stride={cfg.stride}, "
        f"mask_col={cfg.valid_mask_col}, min_valid_ratio={cfg.min_valid_ratio}, "
        f"drop_incomplete={cfg.drop_incomplete}, pad_mode={cfg.pad_mode}, "
        f"dropped_by_mask={n_drop_mask}"
    )

    return SlidingWindowResult(X=X, Y=Y, t0=t0, idx0=idx0, cfg=cfg)


# =============================================================================
# 从 YAML dict 构造配置
# =============================================================================

def sliding_config_from_dict(d: Dict[str, Any]) -> SlidingWindowConfig:
    """
    从 YAML dict 构造 SlidingWindowConfig。

    兼容你现在的异步监督配置：
      valid_mask_col: null
      min_valid_ratio: 0.0
      drop_incomplete: false
    """
    return SlidingWindowConfig(
        input_cols=d["input_cols"],
        target_cols=d["target_cols"],
        hist_len=int(d["hist_len"]),
        pred_len=int(d["pred_len"]),
        stride=int(d.get("stride", 1)),
        valid_mask_col=d.get("valid_mask_col"),
        min_valid_ratio=float(d.get("min_valid_ratio", 1.0)),
        drop_incomplete=bool(d.get("drop_incomplete", True)),
        pad_mode=str(d.get("pad_mode", "nan")).lower(),
    )
