"""
模块名称：DVL 绘图

模块职责：
生成原始 DVL BI/BE 速度图与预处理后 DVL CSV 的科研风格图像，
统一遵循无标题、共享时间轴与最小 legend 的视觉规范。

主要功能：
1. 为原始 DVL 的 BI / BE 速度生成 2 行共享 x 轴图。
2. 为预处理后的 DVL CSV 生成 3 行共享 x 轴图，覆盖体速度、垂向速度与深度。
3. 对缺失字段的面板使用最小轴内文本提示，而不是标题或多重 legend。

数据流：
DvlFrame 或 dvl_proc.csv
    ↓
统一 style token + 多行传感器布局
    ↓
无标题、共享 x 轴、最小 legend 的 figure
    ↓
out/.../plots/dvl_vel_BI_BE.png / dvl_proc_BI_BE_BD.png

依赖模块：
- pandas
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.style.imu_style

备注：
- 原始 BI / BE 图的三轴颜色与 IMU 的 X/Y/Z 语义保持一致。
- 单变量面板默认不使用 legend。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uwnav_dynamics.io.readers.dvl_reader import DvlFrame
from uwnav_dynamics.viz.style.imu_style import (
    IMU_AXIS_COLORS,
    Imu3RowLayout,
    add_xyz_legend,
    plot_xyz_lines,
    set_y_ticks_pretty_3,
)
from uwnav_dynamics.viz.style.sci_style import apply_axes_style, get_figure_size, setup_mpl


@dataclass(frozen=True)
class DvlPlotPaths:
    """原始 DVL 图产物的标准路径集合。"""
    run_dir: Path
    plots_dir: Path
    dvl_vel_png: Path


def _resolve_out_dirs(dvl: DvlFrame, out_root: str | Path) -> DvlPlotPaths:
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / dvl.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return DvlPlotPaths(run_dir=run_dir, plots_dir=plots_dir, dvl_vel_png=plots_dir / "dvl_vel_BI_BE.png")


def save_dvl_bi_be_vel_2rows(
    dvl: DvlFrame,
    *,
    out_root: str | Path = "out/dvl_plots",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Path:
    """保存原始 DVL 的 BI/BE 两行速度图。"""
    setup_mpl()
    paths = _resolve_out_dirs(dvl, out_root)

    bi = dvl.view_kind("BI", require_valid=True)
    be = dvl.view_kind("BE", require_valid=True)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=get_figure_size("sensor_2row"), dpi=layout.dpi)
    ax1, ax2 = axes

    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_major(),
            length=layout.tick_len_major(),
            direction="out",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_minor(),
            length=layout.tick_len_minor(),
            direction="out",
        )
        apply_axes_style(ax, grid=False)

    if bi.t_s.size > 0 and bi.v_body_mps is not None:
        t_bi = bi.t_s - float(bi.t_s[0]) if use_rel_time else bi.t_s
        lines = plot_xyz_lines(ax1, t_bi, np.asarray(bi.v_body_mps, dtype=float), linewidth=layout.lw())
        add_xyz_legend(ax1, lines, layout)
    else:
        ax1.text(0.5, 0.5, "Body velocity missing", transform=ax1.transAxes, ha="center", va="center")

    ax1.set_ylabel(r"$v_{body}$ (m/s)")
    set_y_ticks_pretty_3(ax1, y_pad_frac=layout.y_pad_frac)

    if be.t_s.size > 0 and be.v_enu_mps is not None:
        t_be = be.t_s - float(be.t_s[0]) if use_rel_time else be.t_s
        plot_xyz_lines(ax2, t_be, np.asarray(be.v_enu_mps, dtype=float), linewidth=layout.lw())
    else:
        ax2.text(0.5, 0.5, "ENU velocity missing", transform=ax2.transAxes, ha="center", va="center")

    ax2.set_ylabel(r"$v_{ENU}$ (m/s)")
    ax2.set_xlabel("Time (s)")
    set_y_ticks_pretty_3(ax2, y_pad_frac=layout.y_pad_frac)

    fig.subplots_adjust(left=layout.left, right=layout.right, bottom=layout.bottom, top=layout.top, hspace=0.18)
    fig.savefig(paths.dvl_vel_png)
    plt.close(fig)
    return paths.dvl_vel_png


@dataclass(frozen=True)
class DvlProcPlotPaths:
    """预处理 DVL 图产物的标准路径集合。"""
    run_dir: Path
    plots_dir: Path
    combined_png: Path


def _resolve_proc_out_dirs(proc_csv: str | Path, out_root: str | Path) -> DvlProcPlotPaths:
    proc_csv = Path(proc_csv)
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / proc_csv.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return DvlProcPlotPaths(run_dir=run_dir, plots_dir=plots_dir, combined_png=plots_dir / "dvl_proc_BI_BE_BD.png")


def save_dvl_proc_figures(
    proc_csv: str | Path,
    *,
    out_root: str | Path = "out/dvl_plots_proc",
    use_rel_time: bool = False,
) -> Path:
    """从预处理 DVL CSV 生成体速度、垂向速度和深度三行图。"""
    setup_mpl()
    paths = _resolve_proc_out_dirs(proc_csv, out_root)

    df = pd.read_csv(proc_csv)
    if "t_s" not in df.columns:
        raise ValueError(f"[DVL-PROC-PLOT] processed CSV {proc_csv} has no 't_s' column.")

    t = df["t_s"].to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("[DVL-PROC-PLOT] empty processed DVL file.")
    t_plot = t - float(t[0]) if use_rel_time else t

    layout = Imu3RowLayout()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("sensor_3row"), dpi=layout.dpi)
    ax_bi, ax_be, ax_bd = axes

    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_major(),
            length=layout.tick_len_major(),
            direction="out",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=layout.tick_fs(),
            width=layout.tick_w_minor(),
            length=layout.tick_len_minor(),
            direction="out",
        )
        apply_axes_style(ax, grid=False)

    bi_cols = ("VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps")
    if all(c in df.columns for c in bi_cols):
        v_body = df[list(bi_cols)].to_numpy(dtype=float)
        lines = plot_xyz_lines(ax_bi, t_plot, v_body, linewidth=layout.lw(), colors=IMU_AXIS_COLORS)
        add_xyz_legend(ax_bi, lines, layout, labels=("Vx", "Vy", "Vz"))
    else:
        ax_bi.text(0.5, 0.5, "Body velocity missing", transform=ax_bi.transAxes, ha="center", va="center")
    ax_bi.set_ylabel("Body vel (m/s)")
    set_y_ticks_pretty_3(ax_bi, y_pad_frac=layout.y_pad_frac)

    be_col = "VelU_enu_mps"
    if be_col in df.columns:
        ax_be.plot(t_plot, df[be_col].to_numpy(dtype=float), color="#30343A", linewidth=layout.lw())
    else:
        ax_be.text(0.5, 0.5, "Vertical velocity missing", transform=ax_be.transAxes, ha="center", va="center")
    ax_be.set_ylabel("Vertical vel (m/s)")
    set_y_ticks_pretty_3(ax_be, y_pad_frac=layout.y_pad_frac)

    depth_col = "Depth_m"
    if depth_col in df.columns:
        ax_bd.plot(t_plot, df[depth_col].to_numpy(dtype=float), color="#4C78A8", linewidth=layout.lw())
    else:
        ax_bd.text(0.5, 0.5, "Depth missing", transform=ax_bd.transAxes, ha="center", va="center")
    ax_bd.set_ylabel("Depth (m)")
    ax_bd.set_xlabel("Time (s)")
    set_y_ticks_pretty_3(ax_bd, y_pad_frac=layout.y_pad_frac)

    fig.subplots_adjust(left=layout.left, right=layout.right, bottom=layout.bottom, top=layout.top, hspace=0.20)
    fig.savefig(paths.combined_png)
    plt.close(fig)
    return paths.combined_png
