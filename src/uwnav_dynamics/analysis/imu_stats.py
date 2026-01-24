# src/uwnav_dynamics/analysis/imu_stats.py
from __future__ import annotations

"""
IMU stats / diagnostics (analysis only; no filtering, no alignment).

- Input: ImuFrame (from uwnav_dynamics.io.readers.imu_reader)
- Output:
    1) ImuStats (dataclass)
    2) a human-readable TXT report saved under:
         <out_root>/<imu_file_stem>/imu_stats.txt

This module is intended for:
  - quickly validating data integrity (time monotonicity, NaNs, sampling jitter)
  - summarizing bias/noise levels from an initial "bias window"
  - producing stable artifacts for experiment archiving / debugging
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from uwnav_dynamics.io.readers.imu_reader import ImuFrame


@dataclass(frozen=True)
class AxisStats:
    mean: np.ndarray      # (3,)
    std: np.ndarray       # (3,)
    p95_abs: np.ndarray   # (3,)
    max_abs: np.ndarray   # (3,)


@dataclass(frozen=True)
class ImuStats:
    path: Path
    kind: str
    time_col: str

    n: int
    t0_s: float
    t1_s: float
    duration_s: float

    dt_mean_s: float
    dt_std_s: float
    dt_p95_s: float
    dt_min_s: float
    dt_max_s: float
    fs_est_hz: float

    n_dt_nonpositive: int
    n_dt_large: int
    dt_large_threshold_s: float

    nan_ratio: Dict[str, float]

    bias_window_s: float
    bias_n: int
    acc_g: AxisStats
    gyro_deg_s: AxisStats
    ang_deg: AxisStats


# -----------------------------
# helpers
# -----------------------------
def _axis_stats(x: np.ndarray) -> AxisStats:
    """
    x: (N,3), may contain NaNs
    """
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    p95_abs = np.nanpercentile(np.abs(x), 95, axis=0)
    max_abs = np.nanmax(np.abs(x), axis=0)
    return AxisStats(mean=mean, std=std, p95_abs=p95_abs, max_abs=max_abs)


def _nan_ratio_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(a)))


def _nan_ratio_2d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(a)))


def analyze_imu(
    imu: ImuFrame,
    *,
    bias_window_s: float = 20.0,
    dt_large_threshold_s: float = 0.05,
) -> ImuStats:
    t = imu.t_s
    dt = imu.dt_s

    n = int(t.size)
    t0 = float(t[0])
    t1 = float(t[-1])
    duration = float(t1 - t0)

    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))
    dt_p95 = float(np.percentile(dt, 95))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    fs_est = float(1.0 / dt_mean) if dt_mean > 0 else float("nan")

    n_nonpos = int(np.sum(dt <= 0.0))
    n_large = int(np.sum(dt > float(dt_large_threshold_s)))

    # NaN/Inf ratios
    nan_ratio: Dict[str, float] = {
        "t_s": _nan_ratio_1d(imu.t_s),
        "acc_g": _nan_ratio_2d(imu.acc_g),
        "gyro_deg_s": _nan_ratio_2d(imu.gyro_deg_s),
        "ang_deg": _nan_ratio_2d(imu.ang_deg),
    }
    if imu.yaw_deg is not None:
        nan_ratio["yaw_deg"] = _nan_ratio_1d(imu.yaw_deg)

    # bias window: first bias_window_s seconds based on t_rel_s
    if bias_window_s <= 0:
        bias_idx = np.arange(n, dtype=int)
    else:
        bias_idx = np.where(imu.t_rel_s <= float(bias_window_s))[0]
        if bias_idx.size == 0:
            # fallback: at least take first 1 second worth samples
            bias_idx = np.where(imu.t_rel_s <= 1.0)[0]

    bias_n = int(bias_idx.size)
    acc_stat = _axis_stats(imu.acc_g[bias_idx, :])
    gyro_stat = _axis_stats(imu.gyro_deg_s[bias_idx, :])
    ang_stat = _axis_stats(imu.ang_deg[bias_idx, :])

    return ImuStats(
        path=imu.path,
        kind=imu.kind,
        time_col=imu.time_col,
        n=n,
        t0_s=t0,
        t1_s=t1,
        duration_s=duration,
        dt_mean_s=dt_mean,
        dt_std_s=dt_std,
        dt_p95_s=dt_p95,
        dt_min_s=dt_min,
        dt_max_s=dt_max,
        fs_est_hz=fs_est,
        n_dt_nonpositive=n_nonpos,
        n_dt_large=n_large,
        dt_large_threshold_s=float(dt_large_threshold_s),
        nan_ratio=nan_ratio,
        bias_window_s=float(bias_window_s),
        bias_n=bias_n,
        acc_g=acc_stat,
        gyro_deg_s=gyro_stat,
        ang_deg=ang_stat,
    )


def _fmt_vec3(v: np.ndarray, fmt: str = "{:+.6f}") -> str:
    v = np.asarray(v, dtype=float).reshape(3)
    return "[" + ", ".join(fmt.format(float(x)) for x in v) + "]"


def render_imu_stats_txt(stats: ImuStats) -> str:
    """
    Create a stable, human-readable TXT report (suitable for archiving and diff).
    """
    lines = []
    lines.append("[IMU-STATS]")
    lines.append(f"file: {stats.path}")
    lines.append(f"kind: {stats.kind}")
    lines.append(f"time_col: {stats.time_col}")
    lines.append("")
    lines.append("[TIME]")
    lines.append(f"N: {stats.n}")
    lines.append(f"t0_s: {stats.t0_s:.6f}")
    lines.append(f"t1_s: {stats.t1_s:.6f}")
    lines.append(f"duration_s: {stats.duration_s:.6f}")
    lines.append("")
    lines.append("[SAMPLING]")
    lines.append(f"dt_mean_s: {stats.dt_mean_s:.6f}")
    lines.append(f"dt_std_s:  {stats.dt_std_s:.6f}")
    lines.append(f"dt_p95_s:  {stats.dt_p95_s:.6f}")
    lines.append(f"dt_min_s:  {stats.dt_min_s:.6f}")
    lines.append(f"dt_max_s:  {stats.dt_max_s:.6f}")
    lines.append(f"fs_est_hz: {stats.fs_est_hz:.3f}")
    lines.append(f"n_dt_nonpositive: {stats.n_dt_nonpositive}")
    lines.append(f"n_dt_large(>{stats.dt_large_threshold_s:.3f}s): {stats.n_dt_large}")
    lines.append("")
    lines.append("[NAN_RATIO]")
    for k in sorted(stats.nan_ratio.keys()):
        lines.append(f"{k}: {stats.nan_ratio[k]*100.0:.3f}%")
    lines.append("")
    lines.append("[BIAS_WINDOW]")
    lines.append(f"bias_window_s: {stats.bias_window_s:.3f}")
    lines.append(f"bias_n: {stats.bias_n}")
    lines.append("")
    lines.append("acc_g:")
    lines.append(f"  mean:    {_fmt_vec3(stats.acc_g.mean)}")
    lines.append(f"  std:     {_fmt_vec3(stats.acc_g.std)}")
    lines.append(f"  p95|x|:  {_fmt_vec3(stats.acc_g.p95_abs, fmt='{:.6f}')}")
    lines.append(f"  max|x|:  {_fmt_vec3(stats.acc_g.max_abs, fmt='{:.6f}')}")
    lines.append("")
    lines.append("gyro_deg_s:")
    lines.append(f"  mean:    {_fmt_vec3(stats.gyro_deg_s.mean)}")
    lines.append(f"  std:     {_fmt_vec3(stats.gyro_deg_s.std)}")
    lines.append(f"  p95|x|:  {_fmt_vec3(stats.gyro_deg_s.p95_abs, fmt='{:.6f}')}")
    lines.append(f"  max|x|:  {_fmt_vec3(stats.gyro_deg_s.max_abs, fmt='{:.6f}')}")
    lines.append("")
    lines.append("ang_deg:")
    lines.append(f"  mean:    {_fmt_vec3(stats.ang_deg.mean)}")
    lines.append(f"  std:     {_fmt_vec3(stats.ang_deg.std)}")
    lines.append(f"  p95|x|:  {_fmt_vec3(stats.ang_deg.p95_abs, fmt='{:.6f}')}")
    lines.append(f"  max|x|:  {_fmt_vec3(stats.ang_deg.max_abs, fmt='{:.6f}')}")
    lines.append("")
    return "\n".join(lines)


def save_imu_stats_txt(
    stats: ImuStats,
    *,
    out_root: str | Path = "out/imu_stats",
) -> Path:
    """
    Save imu_stats.txt under:
      <out_root>/<imu_file_stem>/imu_stats.txt
    """
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / stats.path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "imu_stats.txt"
    out_path.write_text(render_imu_stats_txt(stats), encoding="utf-8")
    return out_path
