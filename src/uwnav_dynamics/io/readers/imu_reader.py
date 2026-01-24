# src/uwnav_dynamics/io/readers/imu_reader.py
from __future__ import annotations

"""
IMU CSV reader (I/O only; no algorithms).

Input CSV columns (typical for min_imu_tb_*.csv):
  - Time columns (at least one must exist):
      MonoNS, EstNS, MonoS, EstS
  - IMU raw measurements:
      AccX, AccY, AccZ        # unit: g
      GyroX, GyroY, GyroZ     # unit: deg/s
      AngX, AngY, AngZ        # unit: deg (Euler angles in vendor convention)
      YawDeg                  # optional, may be empty

This module:
  - Reads CSV into DataFrame
  - Extracts a canonical time axis t_s (seconds), plus t_rel_s and dt_s
  - Packs arrays into ImuFrame with basic unit conversions:
      acc_mps2 = acc_g * g0
      gyro_rad_s = gyro_deg_s * pi/180
      ang_rad = ang_deg * pi/180

This module must NOT:
  - do coordinate-frame transforms (RFU/FRD/ENU, etc.)
  - do gravity compensation / bias correction / filtering
  - align with DVL/PWM/Volt
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TIME_COL_DEFAULT_PRIORITY: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS")


@dataclass(frozen=True)
class ImuFrame:
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
    for c in priority:
        if c in df.columns:
            return c
    raise KeyError(f"IMU CSV has no supported time column in {list(priority)}")


def _extract_time_s(df: pd.DataFrame, time_col: str) -> np.ndarray:
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
    """
    Read IMU CSV and return an ImuFrame.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to IMU CSV.
    kind : str
        Tag from dataset selection, e.g. "min_tb".
    time_priority : Sequence[str]
        Preferred time columns order.
    g0_mps2 : float
        Conversion from g to m/s^2 (project convention).

    Returns
    -------
    ImuFrame
    """
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
