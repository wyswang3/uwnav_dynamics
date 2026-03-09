"""
模块名称：IMU 日志读取器

模块职责：
负责将原始 IMU CSV 读取为统一的 `ImuFrame`，
只做 I/O 层的时间轴整理与基础单位换算，不引入算法处理。

主要功能：
1. 读取 IMU CSV 并解析 canonical 时间轴 `t_s / t_rel_s / dt_s`。
2. 提取加速度、角速度和姿态角原始列。
3. 做最小单位换算，得到 `acc_mps2 / gyro_rad_s / ang_rad`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TIME_COL_DEFAULT_PRIORITY: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS")


@dataclass(frozen=True)
class ImuFrame:
    """单个 IMU 文件读取后的原始表、时间轴和派生 SI 单位数组。"""
    path: Path
    kind: str

    df: pd.DataFrame
    time_col: str

    # time axes
    t_s: np.ndarray        # (N,)
    t_rel_s: np.ndarray    # (N,)
    dt_s: np.ndarray       # (N-1,)

    # raw arrays (vendor units)
    acc_g: np.ndarray        # (N,3)
    gyro_deg_s: np.ndarray   # (N,3)
    ang_deg: np.ndarray      # (N,3)

    # derived arrays (SI)
    acc_mps2: np.ndarray     # (N,3)
    gyro_rad_s: np.ndarray   # (N,3)
    ang_rad: np.ndarray      # (N,3)

    # optional column (may be all-NaN)
    yaw_deg: Optional[np.ndarray] = None  # (N,) or None


def _pick_time_col(df: pd.DataFrame, priority: Sequence[str]) -> str:
    """按优先级选择 IMU CSV 中可用的时间列。"""
    for c in priority:
        if c in df.columns:
            return c
    raise KeyError(f"IMU CSV has no supported time column in {list(priority)}")


def _extract_time_s(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """把秒或纳秒时间列统一转换为秒级时间轴。"""
    t = df[time_col].to_numpy(dtype=float)
    if time_col.endswith("NS"):
        t = t * 1e-9
    return t


def read_imu_csv(
    path: str | Path,
    *,
    kind: str = "unknown",
    time_priority: Sequence[str] = _TIME_COL_DEFAULT_PRIORITY,
    g0_mps2: float = 9.78,
) -> ImuFrame:
    """读取 IMU CSV，并返回带原始值与 SI 单位派生值的 `ImuFrame`。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"IMU CSV not found: {p}")

    df = pd.read_csv(p)

    # required columns
    req = ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "AngX", "AngY", "AngZ")
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"IMU CSV missing columns {missing}: {p}")
    # 选择 time_col 后：
    time_col = _pick_time_col(df, time_priority)
    t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    if time_col.endswith("NS"):
        t_raw = t_raw * 1e-9    

    valid = np.isfinite(t_raw)
    # 若存在无效时间行：丢弃整行（联动所有列）
    if not np.all(valid):
        df = df.loc[valid].reset_index(drop=True)
        t_raw = t_raw[valid]

    if t_raw.size < 2:
        raise ValueError(f"IMU time column invalid (valid<2): {p} (col={time_col})")
         
    t_s = _extract_time_s(df, time_col)

    # basic time sanity (do not "fix" here; just ensure usable)
    if t_s.size < 2:
        raise ValueError(f"IMU CSV too short (N={t_s.size}): {p}")
    if not np.all(np.isfinite(t_s)):
        raise ValueError(f"IMU time column has NaN/Inf: {p} (col={time_col})")

    t_rel_s = t_s - float(t_s[0])
    dt_s = np.diff(t_s)

    # raw arrays
    acc_g = df[["AccX", "AccY", "AccZ"]].to_numpy(dtype=float)
    gyro_deg_s = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)
    ang_deg = df[["AngX", "AngY", "AngZ"]].to_numpy(dtype=float)

    # derived arrays
    acc_mps2 = acc_g * float(g0_mps2)
    gyro_rad_s = gyro_deg_s * (np.pi / 180.0)
    ang_rad = ang_deg * (np.pi / 180.0)

    yaw_deg: Optional[np.ndarray]
    if "YawDeg" in df.columns:
        # allow empty column -> becomes NaN array
        yaw_deg = df["YawDeg"].to_numpy(dtype=float)
    else:
        yaw_deg = None

    return ImuFrame(
        path=p,
        kind=kind,
        df=df,
        time_col=time_col,
        t_s=t_s,
        t_rel_s=t_rel_s,
        dt_s=dt_s,
        acc_g=acc_g,
        gyro_deg_s=gyro_deg_s,
        ang_deg=ang_deg,
        acc_mps2=acc_mps2,
        gyro_rad_s=gyro_rad_s,
        ang_rad=ang_rad,
        yaw_deg=yaw_deg,
    )
