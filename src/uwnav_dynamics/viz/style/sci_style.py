# offline_nav/src/offnav/viz/style.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from cycler import cycler


# =========================
# 统一风格（全局）
# =========================

@dataclass(frozen=True)
class MplGlobalStyle:
    # Fonts
    font_family: str = "serif"
    font_serif: Tuple[str, ...] = ("Times New Roman", "Times", "DejaVu Serif")
    base_fontsize: int = 12
    labelsize: int = 12
    ticksize: int = 11
    legendsize: int = 11

    # Axes / lines
    axes_linewidth: float = 1.0
    tick_length: float = 5.0
    tick_width: float = 1.0
    default_linewidth: float = 1.1  # 全局默认（轨迹会再特化）
    grid: bool = False

    # Figure
    figure_dpi: int = 150
    # 单栏论文常用：宽约 85mm(3.35")；你现在习惯略大一些也没问题
    figure_size_single: Tuple[float, float] = (4.7, 3.3)   # 约 12cm x 8.5cm
    figure_size_wide: Tuple[float, float] = (5.4, 2.4)     # 两子图并排常用

    # Color cycle（方法对比时用）
    prop_cycle: Tuple[str, ...] = (
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
    )

    # Legend
    legend_frameon: bool = False
    legend_borderaxespad: float = 0.3


_GLOBAL = MplGlobalStyle()


def setup_mpl(style: MplGlobalStyle = _GLOBAL) -> None:
    """
    Set global matplotlib rcParams for paper-quality figures.
    Call once before plotting.
    """
    plt.rcParams.update(
        {
            # Font
            "font.family": style.font_family,
            "font.serif": list(style.font_serif),
            "font.size": style.base_fontsize,
            "axes.labelsize": style.labelsize,
            "axes.titlesize": style.labelsize,  # 默认不写 title；保留一致性
            "xtick.labelsize": style.ticksize,
            "ytick.labelsize": style.ticksize,
            "legend.fontsize": style.legendsize,

            # Axes
            "axes.grid": style.grid,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": True,
            "axes.spines.right": True,

            # Ticks
            "xtick.major.size": style.tick_length,
            "xtick.major.width": style.tick_width,
            "ytick.major.size": style.tick_length,
            "ytick.major.width": style.tick_width,

            # Lines
            "lines.linewidth": style.default_linewidth,

            # Legend
            "legend.frameon": style.legend_frameon,
            "legend.borderaxespad": style.legend_borderaxespad,

            # Colors
            "axes.prop_cycle": cycler(color=list(style.prop_cycle)),

            # Figure
            "figure.figsize": style.figure_size_single,
            "figure.dpi": style.figure_dpi,
            "savefig.dpi": 300,  # 导出默认 300dpi（期刊更稳）
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


# =========================
# 轨迹绘图特化（局部）
# =========================

@dataclass(frozen=True)
class TrajStyle:
    # Trajectory line
    traj_lw: float = 1.0
    traj_alpha: float = 0.95

    # Start / End markers (paper-friendly)
    start_marker: str = "o"
    end_marker: str = "o"
    start_s: float = 18.0
    end_s: float = 22.0
    marker_edge_lw: float = 0.9

    # Semantic colors (single-method default)
    traj_color: str = "#1f77b4"      # deep blue
    depth_color: str = "#2ca02c"     # green

    # Start/End styling: open vs filled (recommended)
    start_face: str = "white"
    end_face: Optional[str] = None  # None => use edgecolor (i.e., same as traj color)


_TRAJ = TrajStyle()


def apply_axes_2d(ax: plt.Axes) -> None:
    """
    Clean 2D axes: white background, consistent spine/tick styling.
    """
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(direction="out")


def plot_start_end(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    ts: TrajStyle = _TRAJ,
    *,
    label_start: Optional[str] = "Start",
    label_end: Optional[str] = "End",
) -> None:
    # Start: open circle
    ax.scatter(
        [x0], [y0],
        s=ts.start_s,
        marker=ts.start_marker,
        facecolors=ts.start_face,
        edgecolors=color,
        linewidths=ts.marker_edge_lw,
        zorder=3,
        label=label_start,
    )

    # End: filled circle
    end_face = ts.end_face if ts.end_face is not None else color
    ax.scatter(
        [x1], [y1],
        s=ts.end_s,
        marker=ts.end_marker,
        facecolors=end_face,
        edgecolors=color,
        linewidths=ts.marker_edge_lw,
        zorder=3,
        label=label_end,
    )


def plot_traj_line(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    label: Optional[str] = None,
    ts: TrajStyle = _TRAJ,
) -> None:
    ax.plot(x, y, color=color, linewidth=ts.traj_lw, alpha=ts.traj_alpha, label=label, zorder=2)


def get_figsize_two_panels(style: MplGlobalStyle = _GLOBAL) -> Tuple[float, float]:
    """
    Recommended size for two panels in one figure (E-N + U-t).
    """
    return style.figure_size_wide
