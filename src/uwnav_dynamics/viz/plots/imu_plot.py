# src/uwnav_dynamics/viz/plots/imu_plots.py
from __future__ import annotations

"""
IMU plotting (raw / processed share the same style layer).

Outputs (default):
  - <out_root>/<imu_file_stem>/plots/imu_raw_9axis.png
  - <out_root>/<imu_file_stem>/plots/imu_dt.png

All figures follow:
  - global scientific rcParams via viz.style.sci_style.setup_mpl()
  - IMU-specific anchored layout via viz.style.imu_style
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from uwnav_dynamics.io.readers.imu_reader import ImuFrame
from uwnav_dynamics.viz.style.sci_style import setup_mpl  # 你已有的全局风格模块（按实际文件名调整）
from uwnav_dynamics.viz.style.imu_style import (
    Imu3RowLayout,
    make_imu_3rows_canvas,
    finalize_imu_axes,
    plot_xyz_lines,
    add_xyz_legend,
    set_y_ticks_pretty_3,
)


@dataclass(frozen=True)
class ImuPlotPaths:
    run_dir: Path
    plots_dir: Path
    imu_raw_9axis_png: Path
    imu_dt_png: Path


def _resolve_out_dirs(imu: ImuFrame, out_root: str | Path) -> ImuPlotPaths:
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / imu.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return ImuPlotPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        imu_raw_9axis_png=plots_dir / "imu_raw_9axis.png",
        imu_dt_png=plots_dir / "imu_dt.png",
    )


def save_imu_raw_9axis(
    imu: ImuFrame,
    *,
    out_root: str | Path = "out/imu_plots",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Path:
    """
    Save 3-row raw IMU figure (acc/gyro/attitude).

    Parameters
    ----------
    imu : ImuFrame
    out_root : output root directory
    layout : anchored layout policy
    use_rel_time : if True, x-axis uses t_rel_s; else uses absolute t_s
    """
    setup_mpl()

    paths = _resolve_out_dirs(imu, out_root)

    t = imu.t_rel_s if use_rel_time else imu.t_s
    xlab = "Time (s)" if use_rel_time else f"Time (s) [{imu.time_col}]"

    fig, axes, layout = make_imu_3rows_canvas(layout)
    lw = layout.lw()

    # Row 1: Acc (g)
    ax = axes[0]
    lines = plot_xyz_lines(ax, t, imu.acc_g, linewidth=lw)
    ax.set_title("Acceleration (g)", fontsize=layout.title_fs())
    add_xyz_legend(ax, lines, layout)

    # Row 2: Gyro (deg/s)
    ax = axes[1]
    lines = plot_xyz_lines(ax, t, imu.gyro_deg_s, linewidth=lw)
    ax.set_title("Angular rate (deg/s)", fontsize=layout.title_fs())
    add_xyz_legend(ax, lines, layout)

    # Row 3: Attitude (deg)
    ax = axes[2]
    lines = plot_xyz_lines(ax, t, imu.ang_deg, linewidth=lw)
    ax.set_title("Attitude (deg)", fontsize=layout.title_fs())
    ax.set_xlabel(xlab, fontsize=layout.label_fs())
    add_xyz_legend(ax, lines, layout)

    # Apply y-tick policy after plotting
    finalize_imu_axes(axes, y_pad_frac=layout.y_pad_frac)

    fig.savefig(paths.imu_raw_9axis_png)
    plt.close(fig)
    return paths.imu_raw_9axis_png


def save_imu_dt(
    imu: ImuFrame,
    *,
    out_root: str | Path = "out/imu_plots",
    use_rel_time: bool = True,
) -> Path:
    """
    Save dt plot for IMU timestamps (sampling stability).

    By default uses relative time on x-axis.
    """
    setup_mpl()
    paths = _resolve_out_dirs(imu, out_root)

    # dt is (N-1,), align to midpoints in time for nicer plot
    if use_rel_time:
        t = imu.t_rel_s
        xlab = "Time (s)"
    else:
        t = imu.t_s
        xlab = f"Time (s) [{imu.time_col}]"

    t_mid = 0.5 * (t[1:] + t[:-1])
    dt = imu.dt_s

    fig = plt.figure(figsize=(4.7, 2.6), dpi=450)  # 单图更扁一些更像论文附图
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t_mid, dt, linewidth=1.0)
    ax.set_title("IMU sampling interval $\\Delta t$ (s)", fontsize=10)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel("$\\Delta t$ (s)", fontsize=9)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.tick_params(direction="out")
    set_y_ticks_pretty_3(ax, y_pad_frac=0.05)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.88)
    fig.savefig(paths.imu_dt_png)
    plt.close(fig)
    return paths.imu_dt_png


def save_imu_raw_figures(
    imu: ImuFrame,
    *,
    out_root: str | Path = "out/imu_plots",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Tuple[Path, Path]:
    """
    Convenience wrapper: save both raw_9axis and dt plots.
    """
    p1 = save_imu_raw_9axis(imu, out_root=out_root, layout=layout, use_rel_time=use_rel_time)
    p2 = save_imu_dt(imu, out_root=out_root, use_rel_time=True)
    return p1, p2
