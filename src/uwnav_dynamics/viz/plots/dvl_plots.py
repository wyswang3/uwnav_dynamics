from __future__ import annotations

"""
DVL plotting (BI / BE velocities).

设计目标：
  - 原始 DVL:
      * save_dvl_bi_be_vel_2rows(DvlFrame, ...) 画 BI/BE 两行子图
  - 预处理后的 DVL CSV:
      * save_dvl_proc_figures(dvl_proc_csv, ...) 画 3 个独立图窗：
          1) BI 体坐标系速度 v_body (X/Y/Z)
          2) BE 垂向速度
          3) BD 深度

依赖：
  - uwnav_dynamics.io.readers.dvl_reader.DvlFrame
  - sci_style.setup_mpl()
  - imu_style 中的布局 / 画线 / y 轴刻度策略
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from uwnav_dynamics.io.readers.dvl_reader import DvlFrame
from uwnav_dynamics.viz.style.sci_style import setup_mpl
from uwnav_dynamics.viz.style.imu_style import (
    Imu3RowLayout,
    set_y_ticks_pretty_3,
    plot_xyz_lines,
    add_xyz_legend,
    IMU_AXIS_COLORS,
)


# ----------------------------------------------------------------------
# 路径管理（原始 DVL）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DvlPlotPaths:
    run_dir: Path
    plots_dir: Path
    dvl_vel_png: Path


def _resolve_out_dirs(dvl: DvlFrame, out_root: str | Path) -> DvlPlotPaths:
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / dvl.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return DvlPlotPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        dvl_vel_png=plots_dir / "dvl_vel_BI_BE.png",
    )


# ----------------------------------------------------------------------
# 主绘图函数：原始 DVL BI / BE 两行子图
# ----------------------------------------------------------------------
def save_dvl_bi_be_vel_2rows(
    dvl: DvlFrame,
    *,
    out_root: str | Path = "out/dvl_plots",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Path:
    """
    绘制 DVL BI / BE 速度的两行子图：

      Row 1: $v_{body}$ (m/s)
      Row 2: $v_{ENU}$  (m/s)

    仅使用简短行标题，不使用纵轴文字。
    """
    setup_mpl()
    paths = _resolve_out_dirs(dvl, out_root)

    # ---------- 时间轴（底部统一用 Time (s)） ----------
    t_all = dvl.t_s
    if use_rel_time:
        t_all = t_all - float(t_all[0])
    xlab = "Time (s)"

    # ---------- 提取 BI / BE 子视图 ----------
    bi = dvl.view_kind("BI", require_valid=True)
    be = dvl.view_kind("BE", require_valid=True)

    if bi.t_s.size == 0:
        print("[DVL-PLOT] WARNING: no valid BI samples; top panel will be empty.")
    if be.t_s.size == 0:
        print("[DVL-PLOT] WARNING: no valid BE samples; bottom panel will be empty.")

    # ---------- 创建 2x1 画布（适合论文单列） ----------
    fig_w = layout.fig_w_in
    fig_h = layout.fig_h_in * 0.8  # 两行略矮

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(fig_w, fig_h),
        dpi=layout.dpi,
        gridspec_kw={"hspace": 0.22},
    )

    for i, ax in enumerate(axes):
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
        if i == 0:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
    )

    # ------------------------------------------------------------------
    # Row 1: BI 体坐标速度 v_body (m/s)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    if bi.t_s.size > 0 and bi.v_body_mps is not None:
        t_bi = bi.t_s
        if use_rel_time:
            t_bi = t_bi - float(bi.t_s[0])

        v_body = np.asarray(bi.v_body_mps, dtype=float)
        lines = plot_xyz_lines(ax1, t_bi, v_body, linewidth=layout.lw())
        ax1.set_title(r"$v_{\mathrm{body}}$ (m/s)", fontsize=layout.title_fs())
        add_xyz_legend(ax1, lines, layout)  # 图例只在这一行
    else:
        ax1.set_title(r"$v_{\mathrm{body}}$ (no data)", fontsize=layout.title_fs())

    set_y_ticks_pretty_3(ax1, y_pad_frac=layout.y_pad_frac)

    # ------------------------------------------------------------------
    # Row 2: BE ENU 速度 v_enu (m/s)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    if be.t_s.size > 0 and be.v_enu_mps is not None:
        t_be = be.t_s
        if use_rel_time:
            t_be = t_be - float(be.t_s[0])

        v_enu = np.asarray(be.v_enu_mps, dtype=float)
        _ = plot_xyz_lines(ax2, t_be, v_enu, linewidth=layout.lw())
        ax2.set_title(r"$v_{\mathrm{ENU}}$ (m/s)", fontsize=layout.title_fs())
    else:
        ax2.set_title(r"$v_{\mathrm{ENU}}$ (no data)", fontsize=layout.title_fs())

    ax2.set_xlabel(xlab, fontsize=layout.label_fs())
    set_y_ticks_pretty_3(ax2, y_pad_frac=layout.y_pad_frac)

    fig.savefig(paths.dvl_vel_png)
    plt.close(fig)
    return paths.dvl_vel_png



# ======================================================================
# 下面是「预处理后的 DVL CSV」绘图（3 个独立图窗）
# ======================================================================
@dataclass(frozen=True)
class DvlProcPlotPaths:
    run_dir: Path
    plots_dir: Path
    combined_png: Path

def _resolve_proc_out_dirs(proc_csv: str | Path,
                           out_root: str | Path) -> DvlProcPlotPaths:
    proc_csv = Path(proc_csv)
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / proc_csv.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return DvlProcPlotPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        combined_png=plots_dir / "dvl_proc_BI_BE_BD.png",
    )

def save_dvl_proc_figures(
    proc_csv: str | Path,
    *,
    out_root: str | Path = "out/dvl_plots_proc",
    use_rel_time: bool = False,
) -> Path:
    """
    单张图包含 3 个子图：

      (1) Body velocity v_body (VelBx/VelBy/VelBz_body_mps)
      (2) Vertical velocity VelU_enu_mps（若不存在则在子图中提示）
      (3) Depth Depth_m（若不存在则在子图中提示）

    纵轴不写文字，仅用每一行简短标题说明物理量，
    底部在最后一行给统一的 Time (s)。
    """
    setup_mpl()
    paths = _resolve_proc_out_dirs(proc_csv, out_root)

    df = pd.read_csv(proc_csv)

    # ---------- 时间轴 ----------
    if "t_s" not in df.columns:
        raise ValueError(f"[DVL-PROC-PLOT] processed CSV {proc_csv} has no 't_s' column.")
    t = df["t_s"].to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("[DVL-PROC-PLOT] empty processed DVL file.")

    if use_rel_time:
        t_plot = t - float(t[0])
    else:
        t_plot = t
    # 为了简洁，统一写 Time (s)
    xlab = "Time (s)"

    layout = Imu3RowLayout()

    # ---------- 创建 3x1 画布 ----------
    fig_w = layout.fig_w_in * 1.3
    fig_h = layout.fig_h_in * 0.9

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(fig_w, fig_h),
        dpi=layout.dpi,
        gridspec_kw={"hspace": 0.28},
    )
    ax_bi, ax_be, ax_bd = axes

    # 统一 tick 风格
    for i, ax in enumerate(axes):
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
        if i < 2:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    # ==============================================================
    # 1) BI 体坐标速度：VelBx_body_mps / VelBy_body_mps / VelBz_body_mps
    # ==============================================================
    bi_cols = ("VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps")
    if not all(c in df.columns for c in bi_cols):
        msg = (
            "[DVL-PROC-PLOT] cannot find BI body velocity columns "
            f"{bi_cols} in {proc_csv}, skip BI plot."
        )
        print(msg)
        ax_bi.text(
            0.5,
            0.5,
            "Body velocity missing\n"
            f"({', '.join(bi_cols)})",
            transform=ax_bi.transAxes,
            ha="center",
            va="center",
        )
        ax_bi.set_title("Body velocity (missing)", fontsize=layout.title_fs())
    else:
        v_body = df[list(bi_cols)].to_numpy(dtype=float)

        c_x = IMU_AXIS_COLORS[0] if len(IMU_AXIS_COLORS) > 0 else None
        c_y = IMU_AXIS_COLORS[1] if len(IMU_AXIS_COLORS) > 1 else None
        c_z = IMU_AXIS_COLORS[2] if len(IMU_AXIS_COLORS) > 2 else None

        ax_bi.plot(t_plot, v_body[:, 0], label="Vx", color=c_x)
        ax_bi.plot(t_plot, v_body[:, 1], label="Vy", color=c_y)
        ax_bi.plot(t_plot, v_body[:, 2], label="Vz", color=c_z)

        ax_bi.set_title("Body velocity (m/s)", fontsize=layout.title_fs())
        ax_bi.legend(fontsize=layout.legend_fs(), loc="upper right")
        ax_bi.grid(True, which="both", alpha=0.3)

    # ==============================================================
    # 2) BE 垂向速度：VelU_enu_mps（可选）
    # ==============================================================
    be_col = "VelU_enu_mps"
    if be_col not in df.columns:
        print(
            "[DVL-PROC-PLOT] WARNING: Cannot find BE vertical velocity column "
            f"'{be_col}', skip BE vertical velocity plot."
        )
        ax_be.text(
            0.5,
            0.5,
            f"Vertical velocity missing\n('{be_col}')",
            transform=ax_be.transAxes,
            ha="center",
            va="center",
        )
        ax_be.set_title("Vertical velocity (missing)", fontsize=layout.title_fs())
    else:
        vup = df[be_col].to_numpy(dtype=float)
        ax_be.plot(t_plot, vup, label="Vz")
        ax_be.set_title("Vertical velocity (m/s)", fontsize=layout.title_fs())
        ax_be.legend(fontsize=layout.legend_fs(), loc="upper right")
        ax_be.grid(True, which="both", alpha=0.3)

    # ==============================================================
    # 3) BD 深度：Depth_m（可选）
    # ==============================================================
    depth_col = "Depth_m"
    if depth_col not in df.columns:
        print(
            "[DVL-PROC-PLOT] WARNING: Cannot find BD depth column "
            f"'{depth_col}', skip BD depth plot."
        )
        ax_bd.text(
            0.5,
            0.5,
            f"Depth missing\n('{depth_col}')",
            transform=ax_bd.transAxes,
            ha="center",
            va="center",
        )
        ax_bd.set_title("Depth (missing)", fontsize=layout.title_fs())
    else:
        depth = df[depth_col].to_numpy(dtype=float)
        ax_bd.plot(t_plot, depth)
        ax_bd.set_title("Depth (m)", fontsize=layout.title_fs())
        ax_bd.grid(True, which="both", alpha=0.3)

    # ---------- 底部统一 x 轴标签 ----------
    ax_bd.set_xlabel(xlab, fontsize=layout.label_fs())

    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        hspace=0.28,
    )

    fig.savefig(paths.combined_png)
    plt.close(fig)

    return paths.combined_png


