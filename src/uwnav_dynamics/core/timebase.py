# src/uwnav_dynamics/core/timebase.py
from __future__ import annotations

"""
uwnav_dynamics.core.timebase

统一时间轴与时间网格工具。

设计目标
--------
- 提供一个轻量的 TimeGrid 抽象，描述统一的等间隔时间网格（如 50 Hz）；
- 提供从 IMU / 数据帧推导 TimeGrid 的便捷函数；
- 提供高频数据聚合到时间网格的工具（mean / first / last）；
- 提供低频数据按最近邻映射到时间网格的工具（带 max_dt 与 mask）。

约定
----
- 所有时间轴均使用「秒」为单位的 float64；
- 默认主频率为 50 Hz（dt = 0.02 s），但接口不写死，可配置；
- 不依赖具体传感器列，仅做纯时间 / 数组级操作。
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 默认主时间轴频率（控制频率）
DEFAULT_MAIN_FREQ_HZ: float = 50.0
DEFAULT_MAIN_DT_S: float = 1.0 / DEFAULT_MAIN_FREQ_HZ


# =============================================================================
# 1. TimeGrid 抽象
# =============================================================================


@dataclass(frozen=True)
class TimeGrid:
    """
    等间隔时间网格。

    Attributes
    ----------
    t0 : float
        网格起始时间（秒）。
    t1 : float
        网格结束时间（秒），包含在网格内（即 t1 ≈ t0 + (N-1)*dt）。
    dt : float
        网格时间步长（秒），如 0.02 对应 50 Hz。
    """

    t0: float
    t1: float
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"TimeGrid.dt must be > 0, got {self.dt}")
        if self.t1 <= self.t0:
            raise ValueError(f"TimeGrid.t1 must be > t0, got t0={self.t0}, t1={self.t1}")

    @property
    def n_steps(self) -> int:
        """网格步数 N（包含两端）。"""
        n = int(np.floor((self.t1 - self.t0) / self.dt)) + 1
        return max(n, 1)

    @property
    def t_s(self) -> np.ndarray:
        """时间网格数组，shape = (N,)，float64 秒。"""
        return self.t0 + np.arange(self.n_steps, dtype=float) * self.dt

    def contains(self, t: float) -> bool:
        """判断某个时间 t（秒）是否落在 [t0, t1] 间。"""
        return (t >= self.t0) and (t <= self.t1)

    def clamp(self, t: float) -> float:
        """将 t 裁剪到 [t0, t1] 内。"""
        return float(np.clip(t, self.t0, self.t1))


# =============================================================================
# 2. 从时间数组 / DataFrame 构造 TimeGrid
# =============================================================================


def make_time_grid_from_array(
    t_s: np.ndarray,
    *,
    dt_s: float = DEFAULT_MAIN_DT_S,
    margin_start_s: float = 0.0,
    margin_end_s: float = 0.0,
) -> TimeGrid:
    """
    根据给定时间数组 t_s 构造等间隔 TimeGrid。

    常用场景：基于 IMU 处理后的 t_s（100 Hz）构造 50 Hz 主时间轴。

    Parameters
    ----------
    t_s : np.ndarray
        原始时间数组（秒），须单调递增。
    dt_s : float, default DEFAULT_MAIN_DT_S (50 Hz)
        目标网格步长。
    margin_start_s : float, default 0.0
        起始处裁剪的时间（秒），例如 3.0 表示从 t_s[0] + 3s 开始。
    margin_end_s : float, default 0.0
        末尾裁剪的时间（秒），例如 3.0 表示在 t_s[-1] - 3s 结束。

    Returns
    -------
    TimeGrid
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("make_time_grid_from_array requires at least 2 samples.")

    t0_raw = float(t[0])
    t1_raw = float(t[-1])

    t0 = t0_raw + float(margin_start_s)
    t1 = t1_raw - float(margin_end_s)

    if t1 <= t0:
        raise ValueError(
            f"Invalid margins: t0={t0_raw}, t1={t1_raw}, "
            f"margin_start_s={margin_start_s}, margin_end_s={margin_end_s}"
        )

    return TimeGrid(t0=t0, t1=t1, dt=float(dt_s))


def make_time_grid_from_imu_df(
    imu_df: pd.DataFrame,
    *,
    t_col: str = "t_s",
    dt_s: float = DEFAULT_MAIN_DT_S,
    margin_start_s: float = 0.0,
    margin_end_s: float = 0.0,
) -> TimeGrid:
    """
    便捷函数：从 IMU 预处理 CSV DataFrame 构造 TimeGrid。

    默认使用 't_s' 列作为时间轴。
    """
    if t_col not in imu_df.columns:
        raise ValueError(f"IMU DataFrame missing time column '{t_col}'.")

    t_s = imu_df[t_col].to_numpy(dtype=float)
    return make_time_grid_from_array(
        t_s,
        dt_s=dt_s,
        margin_start_s=margin_start_s,
        margin_end_s=margin_end_s,
    )


# =============================================================================
# 3. 高频数据聚合到 TimeGrid（100 Hz → 50 Hz 等）
# =============================================================================

AggKind = Literal["mean", "first", "last"]


def aggregate_to_grid(
    t_src: np.ndarray,
    v_src: np.ndarray,
    grid: TimeGrid,
    *,
    agg: AggKind = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将高频时间序列按 TimeGrid 聚合（binning）。

    用于：
      - IMU 100 Hz → 50 Hz（mean / last）
      - PWM 100 Hz → 50 Hz（通常 last）

    Parameters
    ----------
    t_src : np.ndarray, shape (Ns,)
        源时间轴（秒），单调递增。
    v_src : np.ndarray, shape (Ns,) 或 (Ns, D)
        源数据。若为一维则自动视为 (Ns,1) 处理。
    grid : TimeGrid
        目标时间网格。
    agg : {"mean", "first", "last"}, default "mean"
        聚合方式：
          - "mean": 区间内样本均值；
          - "first": 区间内第一个样本；
          - "last": 区间内最后一个样本。

    Returns
    -------
    v_grid : np.ndarray, shape (Ng, D)
        聚合后的数据（对无样本的 bin，填 NaN）。
    counts : np.ndarray, shape (Ng,)
        每个网格 bin 包含的源样本数。
    """
    t = np.asarray(t_src, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("aggregate_to_grid: empty t_src.")

    v = np.asarray(v_src)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.shape[0] != t.size:
        raise ValueError(
            f"aggregate_to_grid: t_src and v_src length mismatch: "
            f"{t.size} vs {v.shape[0]}"
        )

    t_grid = grid.t_s
    dt = grid.dt
    Ng = t_grid.size
    D = v.shape[1]

    # 构造 bin 边界：每个中心 t_grid[k] 对应 [t_k - dt/2, t_k + dt/2)
    edges = np.empty(Ng + 1, dtype=float)
    edges[:-1] = t_grid - 0.5 * dt
    edges[-1] = t_grid[-1] + 0.5 * dt

    # 使用 searchsorted 找每个 bin 的 [left, right) 索引
    idx_left = np.searchsorted(t, edges[:-1], side="left")
    idx_right = np.searchsorted(t, edges[1:], side="left")

    v_grid = np.full((Ng, D), np.nan, dtype=float)
    counts = np.zeros(Ng, dtype=int)

    for k in range(Ng):
        i0 = int(idx_left[k])
        i1 = int(idx_right[k])
        if i1 <= i0:
            continue  # 此 bin 内无样本

        sl = v[i0:i1, :]
        counts[k] = sl.shape[0]

        if agg == "mean":
            v_grid[k, :] = np.nanmean(sl, axis=0)
        elif agg == "first":
            v_grid[k, :] = sl[0, :]
        elif agg == "last":
            v_grid[k, :] = sl[-1, :]
        else:
            raise ValueError(f"Unknown agg kind: {agg!r}")

    return v_grid, counts


# =============================================================================
# 4. 低频数据映射到 TimeGrid（最近邻 + 最大时间差约束）
# =============================================================================


def map_to_grid_nearest(
    t_src: np.ndarray,
    v_src: np.ndarray,
    grid: TimeGrid,
    *,
    max_dt_s: Optional[float] = None,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将低频时间序列按最近邻映射到 TimeGrid。

    用于：
      - DVL 10 Hz → 50 Hz：每个 50 Hz 时刻最多关联最近一条 DVL；
      - Power 5 Hz → 50 Hz：每个 50 Hz 时刻最多关联最近一条功率测量。

    Parameters
    ----------
    t_src : np.ndarray, shape (Ns,)
        源时间轴（秒），单调递增。
    v_src : np.ndarray, shape (Ns,) 或 (Ns, D)
        源数据。
    grid : TimeGrid
        目标时间网格。
    max_dt_s : float, optional
        最近邻允许的最大时间差（秒）。如果最近样本的时间差超过该阈值，
        则该网格点不赋值（保持 fill_value，mask 为 False）。
        若为 None，则不做距离限制。
    fill_value : float, default NaN
        网格点无匹配源样本时填充的值。

    Returns
    -------
    v_grid : np.ndarray, shape (Ng, D)
        映射后的数据，对无匹配点为 fill_value。
    mask : np.ndarray, shape (Ng,)
        bool 掩码，True 表示该网格点有有效映射（且满足 max_dt_s）。
    """
    t = np.asarray(t_src, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("map_to_grid_nearest: empty t_src.")

    v = np.asarray(v_src)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.shape[0] != t.size:
        raise ValueError(
            f"map_to_grid_nearest: t_src and v_src length mismatch: "
            f"{t.size} vs {v.shape[0]}"
        )

    t_grid = grid.t_s
    Ng = t_grid.size
    D = v.shape[1]

    v_grid = np.full((Ng, D), fill_value, dtype=float)
    mask = np.zeros(Ng, dtype=bool)

    # 双指针（假设 t 单调递增）
    j = 0
    Ns = t.size

    for k in range(Ng):
        tk = t_grid[k]

        # 往前推进 j，使其尽量靠近 tk
        while j + 1 < Ns and abs(t[j + 1] - tk) <= abs(t[j] - tk):
            j += 1

        dt = abs(t[j] - tk)
        if (max_dt_s is not None) and (dt > max_dt_s):
            # 时间差太远，不使用该样本
            continue

        v_grid[k, :] = v[j, :]
        mask[k] = True

    return v_grid, mask


# =============================================================================
# 5. 帮助函数：简单检查 / 日志用
# =============================================================================


def summarize_time_axis(t_s: Sequence[float]) -> str:
    """
    生成一个简短的时间轴摘要字符串，用于日志打印。

    例如：
      "[t_s] N=50000, t=[351.044, 906.720], dt_med=0.0100"
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    if t.size == 0:
        return "[t_s] N=0"

    if t.size == 1:
        return f"[t_s] N=1, t=[{t[0]:.3f}]"

    dt = np.diff(t)
    dt_med = float(np.median(dt))
    return (
        f"[t_s] N={t.size}, "
        f"t=[{t[0]:.3f}, {t[-1]:.3f}], "
        f"dt_med={dt_med:.5f}"
    )
