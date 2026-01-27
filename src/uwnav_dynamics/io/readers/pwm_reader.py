# src/uwnav_dynamics/io/readers/pwm_reader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PwmData:
    """
    Parsed PWM log with sensor-time mapping.

    Fields
    ------
    t_s : (N,) float64
        PWM local relative time in seconds.
    est_s : (N,) float64
        Absolute time in EstS (seconds) aligned to sensor time system.
    est_ns : (N,) int64
        Absolute time in EstNS (nanoseconds), derived from est_s.
    cmd : (N,8) float64
        Commanded PWM (or normalized) values for 8 channels.
    applied : (N,8) float64
        Applied PWM values for 8 channels.
    df : pd.DataFrame
        Original dataframe with added columns for debugging / downstream joins.
    """
    t_s: np.ndarray
    est_s: np.ndarray
    est_ns: np.ndarray
    cmd: np.ndarray
    applied: np.ndarray
    df: pd.DataFrame


# -----------------------------------------------------------------------------
# Reader
# -----------------------------------------------------------------------------
_DEFAULT_CMD_COLS = [f"ch{i}_cmd" for i in range(1, 9)]
_DEFAULT_APPLIED_COLS = [f"ch{i}_applied" for i in range(1, 9)]


def read_pwm_csv(
    *,
    csv_path: str | Path,
    ests0_pwm: float,
    ts0_pwm: float = 0.0,
    timebase_method: str = "none",
    cmd_cols: Optional[Sequence[str]] = None,
    applied_cols: Optional[Sequence[str]] = None,
    add_time_columns: bool = True,
) -> PwmData:
    """
    Read PWM log CSV and add sensor-time columns (EstS/EstNS).

    Time mapping
    -----------
    EstS = ests0_pwm + (t_s - ts0_pwm)

    Parameters
    ----------
    csv_path : str | Path
        PWM csv path.
    ests0_pwm : float
        PWM timebase anchor in sensor EstS system.
    ts0_pwm : float, default 0.0
        The t_s value that corresponds to ests0_pwm (usually first t_s).
    timebase_method : str, default "none"
        Reserved for future use (e.g. "filename_offset_to_imu_ests0").
        Currently not used in math; only stored for traceability.
    cmd_cols / applied_cols : optional
        Column names for command/applied channels. Defaults to ch1..ch8.
    add_time_columns : bool, default True
        If True, add 'EstS' and 'EstNS' columns into returned df.

    Returns
    -------
    PwmData
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"PWM csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ---- required columns ----
    if "t_s" not in df.columns:
        raise ValueError(f"PWM csv missing required column 't_s': {csv_path}")

    cmd_cols = list(cmd_cols) if cmd_cols is not None else list(_DEFAULT_CMD_COLS)
    applied_cols = list(applied_cols) if applied_cols is not None else list(_DEFAULT_APPLIED_COLS)

    missing = [c for c in (cmd_cols + applied_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"PWM csv missing columns: {missing} in {csv_path}")

    # ---- parse time ----
    t_s = df["t_s"].to_numpy(dtype=float)

    # Basic sanity: nondecreasing
    if t_s.size >= 2:
        dt = np.diff(t_s)
        if np.any(dt < -1e-9):
            # allow tiny numerical noise but not real decrease
            bad_i = int(np.where(dt < -1e-9)[0][0])
            raise ValueError(
                f"PWM t_s is not nondecreasing at idx={bad_i}: "
                f"t_s[{bad_i}]={t_s[bad_i]} -> t_s[{bad_i+1}]={t_s[bad_i+1]} "
                f"in {csv_path}"
            )

    est_s = float(ests0_pwm) + (t_s - float(ts0_pwm))
    est_ns = np.round(est_s * 1e9).astype(np.int64)

    if add_time_columns:
        df = df.copy()
        df["EstS"] = est_s
        df["EstNS"] = est_ns
        # optional traceability columns
        df.attrs["pwm_timebase_method"] = str(timebase_method)
        df.attrs["pwm_ests0_pwm"] = float(ests0_pwm)
        df.attrs["pwm_ts0_pwm"] = float(ts0_pwm)

    cmd = df[cmd_cols].to_numpy(dtype=float)
    applied = df[applied_cols].to_numpy(dtype=float)

    return PwmData(
        t_s=t_s,
        est_s=est_s,
        est_ns=est_ns,
        cmd=cmd,
        applied=applied,
        df=df,
    )
