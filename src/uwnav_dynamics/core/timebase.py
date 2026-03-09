"""
模块名称：统一时间基工具

模块职责：
提供与具体传感器解耦的时间网格、聚合和最近邻映射工具，
为多传感器对齐与采样频率变换提供基础能力。

主要功能：
1. 用 `TimeGrid` 描述统一等间隔主时间轴。
2. 从时间数组或 DataFrame 生成时间网格。
3. 将高频信号按 mean/first/last 聚合到目标网格。
4. 将低频或稀疏信号按最近邻映射到目标网格，并输出有效性 mask。

数据流：
source time arrays / DataFrame
    ↓
TimeGrid
    ↓
aggregate_to_grid() / map_to_grid_nearest()
    ↓
aligned arrays + mask

依赖模块：
- numpy
- pandas
"""

from __future__ import annotations

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
    """根据原始时间数组构造等间隔主时间网格。"""
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
    """从包含时间列的 DataFrame 直接推导主时间网格。"""
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
    """将高频数据按目标时间网格分箱聚合，并返回聚合值与样本计数。"""
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
