# src/uwnav_dynamics/preprocess/sliding_window.py
from __future__ import annotations

"""
滑动窗口数据集构造器（通用版）

核心场景（配合 100 Hz 控制频率）：
  - 已有一张按时间排序、对齐到「主时间轴」的训练基础表 train_base：
        t_s,  PWM（控制量）, IMU / DVL（观测量）, 其他辅助量 ...
  - 希望构造监督学习样本，用于「控制导向」的动力学建模：

        X: (N_win, L, D_in)   # 历史窗口，通常包含：
                              #   - 过去 L 步的控制输入（PWM 等）
                              #   - 过去 L 步的观测（IMU a/ω, DVL v 等）
        Y: (N_win, H, D_out)  # 预测窗口，通常是：
                              #   - 未来 H 步的观测量（例如 a, v, ω）

  典型用法（概念上）：
    - input_cols  = control_cols + obs_cols
    - target_cols = future_obs_cols （同一个表里的列，只是时间上平移 H 步）

  其中：
    - hist_len L  : 历史长度（例如 100 → 1s）
    - pred_len H  : 预测长度（例如 50 → 0.5s）
    - stride      : 窗口滑动步长（通常为 1）

  可选：
    - valid_mask_col : 用于剔除包含大量无效样本的窗口（例如 DVL 掉测）。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd


# =============================================================================
# 配置与结果数据结构
# =============================================================================

@dataclass
class SlidingWindowConfig:
    """
    滑动窗口构造配置（不关心具体物理含义，只负责时间维度切片）。

    约定（推荐用法）：
      - input_cols  : 每个时间步喂给网络的所有特征
                      典型：control_cols + obs_cols
                      例如：PWM(8) + IMU(a, ω, 6) + DVL(v_body, 3) + 辅助量...
      - target_cols : 需要网络预测的量（在未来 H 步上的轨迹）
                      典型：未来的 a / v / ω 中的一部分列
    """
    # --- 基本列 ---
    input_cols: Sequence[str]   # X 的特征列名（历史 L 步）
    target_cols: Sequence[str]  # Y 的特征列名（未来 H 步）

    # --- 窗口长度（单位：步）---
    hist_len: int               # L，历史长度
    pred_len: int               # H，预测长度
    stride: int = 1             # 滑动步长（通常为 1）

    # --- 有效性掩码（可选）---
    #  典型用法：valid_mask_col 是一个 0/1 标志（例如 has_valid_dvl）：
    #    - 1: 此时刻观测有效
    #    - 0: 此时刻观测无效（掉测 / 插值过多等）
    #  滑窗时会统计窗口内“有效步”的比例，并与 min_valid_ratio 进行比较。
    valid_mask_col: Optional[str] = None
    min_valid_ratio: float = 1.0  # 1.0 表示窗口内所有步都需有效，0.8 允许 20% 无效

    # --- 边界处理 ---
    drop_incomplete: bool = True  # True：丢弃尾部不满 L+H 的窗口

    def total_span(self) -> int:
        """每个样本覆盖的总步数 L+H。"""
        return int(self.hist_len) + int(self.pred_len)

    def check_valid(self) -> None:
        """参数合法性检查。"""
        if self.hist_len <= 0 or self.pred_len <= 0:
            raise ValueError(
                f"hist_len/pred_len must be positive, got L={self.hist_len}, H={self.pred_len}"
            )
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if not self.input_cols:
            raise ValueError("input_cols must be non-empty")
        if not self.target_cols:
            raise ValueError("target_cols must be non-empty")
        if self.min_valid_ratio <= 0.0 or self.min_valid_ratio > 1.0:
            raise ValueError(
                f"min_valid_ratio must be in (0,1], got {self.min_valid_ratio}"
            )


@dataclass
class SlidingWindowResult:
    """
    滑窗构造结果，面向后续存盘 / 训练。

    Shapes:
      - X : (N_win, L, D_in)
      - Y : (N_win, H, D_out)
      - t0: (N_win,)  每个窗口的起始时间（来自 base 表的 time_col）
      - idx0: (N_win,) 每个窗口在 base 表中的起始下标
    """
    X: np.ndarray
    Y: np.ndarray
    t0: np.ndarray
    idx0: np.ndarray
    cfg: SlidingWindowConfig


# =============================================================================
# 内部工具：生成窗口起始索引
# =============================================================================

def _make_indices(n: int, cfg: SlidingWindowConfig) -> List[int]:
    """
    计算所有窗口的起始下标列表。

    n: base 表总长度（行数）
    """
    cfg.check_valid()
    span = cfg.total_span()
    idx_list: List[int] = []

    if cfg.drop_incomplete:
        # 只允许完整覆盖 L+H 步的窗口
        max_start = n - span
    else:
        # 允许尾部预测窗口不足 H 步（当前工程中通常不使用）
        max_start = n - cfg.hist_len

    if max_start < 0:
        return []

    i = 0
    while i <= max_start:
        idx_list.append(i)
        i += cfg.stride
    return idx_list


# =============================================================================
# 核心 API：从 DataFrame 生成 (X, Y)
# =============================================================================

def make_sliding_windows(
    df: pd.DataFrame,
    time_col: str,
    cfg: SlidingWindowConfig,
) -> SlidingWindowResult:
    """
    从一张按时间排序的 DataFrame 构造监督学习窗口数据。

    Parameters
    ----------
    df : DataFrame
        训练基础表，要求 time_col 单调递增（已在前面的对齐阶段保证）。
        典型示例：50 Hz 主时间轴下，包含 t_s / PWM / IMU / DVL 等列。
    time_col : str
        用作 t0_s 的时间列，一般为 't_s'（50 Hz 主时间轴）。
    cfg : SlidingWindowConfig
        滑窗配置：
          - input_cols  : X 的列名（历史）
          - target_cols : Y 的列名（未来）
          - hist_len    : L
          - pred_len    : H

    Returns
    -------
    SlidingWindowResult
        包含 X/Y/t0/idx0 的数组打包结果。
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col={time_col!r} not in DataFrame columns.")

    n = len(df)
    if n == 0:
        raise ValueError("Empty DataFrame for sliding window construction.")

    # 保证列存在
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
        # 注意：mask_arr 仅用于「窗口有效性筛选」，不会进入 X/Y
        mask_arr = df[cfg.valid_mask_col].to_numpy(dtype=float)

    idx_candidates = _make_indices(n, cfg)
    if not idx_candidates:
        raise ValueError(
            f"No valid window start indices for n={n}, "
            f"L={cfg.hist_len}, H={cfg.pred_len}, drop_incomplete={cfg.drop_incomplete}"
        )

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    t0_list: List[float] = []
    idx0_list: List[int] = []

    total = len(idx_candidates)
    n_kept = 0

    for i0 in idx_candidates:
        i_hist_start = i0
        i_hist_end = i0 + cfg.hist_len
        i_pred_start = i_hist_end
        i_pred_end = i_hist_end + cfg.pred_len

        if cfg.drop_incomplete and i_pred_end > n:
            # 理论上不会走到这里，因为 _make_indices 已经约束 max_start
            continue

        # 窗口范围 [i_hist_start, i_pred_end)
        if mask_arr is not None:
            m_win = mask_arr[i_hist_start:i_pred_end]
            # 非 NaN & >0.5 视为“有效”，允许掩码列为 0/1 或 NaN/1 等
            valid = np.isfinite(m_win) & (m_win > 0.5)
            valid_ratio = float(valid.mean()) if valid.size > 0 else 0.0
            if valid_ratio < cfg.min_valid_ratio:
                continue  # 丢弃该窗口

        X_win = X_source[i_hist_start:i_hist_end]
        Y_win = Y_source[i_pred_start:i_pred_end]

        # 长度 sanity check
        if X_win.shape[0] != cfg.hist_len:
            continue
        if Y_win.shape[0] != cfg.pred_len:
            if cfg.drop_incomplete:
                continue

        X_list.append(X_win)
        Y_list.append(Y_win)
        t0_list.append(float(time_arr[i0]))
        idx0_list.append(i0)
        n_kept += 1

    if n_kept == 0:
        raise RuntimeError(
            "Sliding window construction produced 0 samples. "
            "Check min_valid_ratio, mask 列与窗口长度设置。"
        )

    X = np.stack(X_list, axis=0)      # (N_win, L, D_in)
    Y = np.stack(Y_list, axis=0)      # (N_win, H, D_out)
    t0 = np.asarray(t0_list, dtype=float)  # (N_win,)
    idx0 = np.asarray(idx0_list, dtype=int)

    print(
        f"[SLIDING] built {X.shape[0]} windows from n={n} rows; "
        f"L={cfg.hist_len}, H={cfg.pred_len}, stride={cfg.stride}, "
        f"mask_col={cfg.valid_mask_col}, min_valid_ratio={cfg.min_valid_ratio}"
    )

    return SlidingWindowResult(
        X=X,
        Y=Y,
        t0=t0,
        idx0=idx0,
        cfg=cfg,
    )


# =============================================================================
# 从 YAML dict 构造配置（build_dataset 用）
# =============================================================================

def sliding_config_from_dict(d: Dict[str, Any]) -> SlidingWindowConfig:
    """
    方便从 YAML dict 构造 SlidingWindowConfig。

    典型 YAML 结构示例：

      sliding:
        input_cols:
          - ch1_cmd
          - ch2_cmd
          - ...
          - AccX_body_mps2
          - AccY_body_mps2
          - ...
        target_cols:
          - AccX_body_mps2
          - AccY_body_mps2
          - AccZ_body_mps2
          - VelBx_body_mps
          - VelBy_body_mps
          - VelBz_body_mps
        hist_len: 100
        pred_len: 50
        stride: 1
        valid_mask_col: dvl_valid_mask   # 可选
        min_valid_ratio: 0.8
        drop_incomplete: true
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
    )
