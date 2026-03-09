"""
模块名称：多行传感器图布局样式

模块职责：
为 IMU、DVL、Power 等多行或多面板传感器图提供统一的布局、
刻度密度和共享 x 轴策略，但不重新定义全局配色或视觉层级。

主要功能：
1. 提供 3 行传感器图的固定画布、边距与刻度策略。
2. 提供稳健的 y 轴 “nice ticks” 规则。
3. 提供 X/Y/Z 三轴折线与单次 legend 的布局辅助函数。

数据流：
plot 模块准备时间轴与传感器数组
    ↓
`make_imu_3rows_canvas()` 创建统一画布
    ↓
`plot_xyz_lines()` / `add_xyz_legend()` 绘图
    ↓
`finalize_imu_axes()` 收尾 y 轴刻度

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 本模块只负责多行布局；颜色、线型、preset 和 legend 极简风格来自 `sci_style.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from uwnav_dynamics.viz.style.sci_style import (
    apply_minimal_legend,
    apply_shared_xlabels,
    get_figure_size,
    get_style,
    get_xyz_styles,
)


_PAPER_STYLE = get_style("paper")
_SENSOR_3ROW_SIZE = get_figure_size("sensor_3row", preset="paper")


@dataclass(frozen=True)
class Imu3RowLayout:
    """三行共享 x 轴传感器图的布局参数集合。"""
    fig_w_in: float = _SENSOR_3ROW_SIZE[0]
    fig_h_in: float = _SENSOR_3ROW_SIZE[1]
    dpi: int = _PAPER_STYLE.figure_dpi
    left: float = 0.14
    right: float = 0.98
    bottom: float = 0.12
    top: float = 0.97
    hspace: float = 0.20
    x_nbins: int = 6
    y_nticks: int = 3
    y_pad_frac: float = 0.03
    ref_in: float = _SENSOR_3ROW_SIZE[0]

    def scale(self) -> float:
        return min(self.fig_w_in, self.fig_h_in) / self.ref_in

    def label_fs(self) -> float:
        return float(_PAPER_STYLE.labelsize) * self.scale()

    def tick_fs(self) -> float:
        return float(_PAPER_STYLE.ticksize) * self.scale()

    def legend_fs(self) -> float:
        return float(_PAPER_STYLE.legendsize) * self.scale()

    def lw(self) -> float:
        return float(_PAPER_STYLE.default_linewidth) * self.scale()

    def tick_len_major(self) -> float:
        return float(_PAPER_STYLE.tick_length) * self.scale()

    def tick_len_minor(self) -> float:
        return 0.65 * self.tick_len_major()

    def tick_w_major(self) -> float:
        return float(_PAPER_STYLE.tick_width) * self.scale()

    def tick_w_minor(self) -> float:
        return 0.75 * self.tick_w_major()


_XYZ = get_xyz_styles()
IMU_AXIS_COLORS: Tuple[str, str, str] = (
    _XYZ["x"].color,
    _XYZ["y"].color,
    _XYZ["z"].color,
)
IMU_AXIS_LABELS: Tuple[str, str, str] = ("X axis", "Y axis", "Z axis")


def apply_legend_style(leg: Optional[plt.Legend]) -> None:
    """对传感器图图例应用统一极简样式。"""
    apply_minimal_legend(leg)


def set_y_ticks_pretty_3(ax: plt.Axes, *, y_pad_frac: float = 0.03) -> None:
    """把 y 轴刻度收敛到最多 3 个较易读的主刻度。"""
    ax.relim()
    ax.autoscale(enable=True, axis="y", tight=False)

    y0, y1 = ax.get_ylim()
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return

    span = float(y1 - y0)
    if span == 0.0:
        eps = 1e-6 if y0 == 0.0 else abs(y0) * 1e-3
        y0, y1 = y0 - eps, y1 + eps
        span = float(y1 - y0)

    pad = abs(span) * float(y_pad_frac)
    ax.set_ylim(y0 - pad, y1 + pad)
    y0, y1 = ax.get_ylim()

    locator = mticker.MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    ax.yaxis.set_major_locator(locator)

    ticks = ax.get_yticks()
    ticks = ticks[np.isfinite(ticks)]
    ticks_in = ticks[(ticks >= y0 - 1e-12) & (ticks <= y1 + 1e-12)]

    if len(ticks_in) < 3:
        pos = np.array([0.25, 0.5, 0.75], dtype=float)
        ax.set_yticks(y0 + (y1 - y0) * pos)
        return

    if len(ticks_in) > 3:
        mid = 0.5 * (y0 + y1)
        order = np.argsort(np.abs(ticks_in - mid))
        ax.set_yticks(np.sort(ticks_in[order[:3]]))
        return

    ax.set_yticks(ticks_in[:3])


def make_imu_3rows_canvas(
    layout: Imu3RowLayout = Imu3RowLayout(),
) -> Tuple[plt.Figure, Sequence[plt.Axes], Imu3RowLayout]:
    """创建标准三行共享 x 轴的传感器画布。"""
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(layout.fig_w_in, layout.fig_h_in),
        dpi=layout.dpi,
        gridspec_kw={"hspace": layout.hspace},
    )

    x_locator = mticker.MaxNLocator(nbins=layout.x_nbins)
    for ax in axes:
        ax.xaxis.set_major_locator(x_locator)
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

    apply_shared_xlabels(list(axes), "Time (s)")
    fig.subplots_adjust(left=layout.left, right=layout.right, bottom=layout.bottom, top=layout.top)
    return fig, axes, layout


def finalize_imu_axes(axes: Sequence[plt.Axes], *, y_pad_frac: float = 0.03) -> None:
    """对三行传感器图统一收尾 y 轴刻度。"""
    for ax in axes:
        set_y_ticks_pretty_3(ax, y_pad_frac=y_pad_frac)


def plot_xyz_lines(
    ax: plt.Axes,
    t_s: np.ndarray,
    xyz: np.ndarray,
    *,
    linewidth: float,
    colors: Tuple[str, str, str] = IMU_AXIS_COLORS,
) -> Tuple[plt.Line2D, plt.Line2D, plt.Line2D]:
    """在单个坐标轴上绘制三轴时序曲线。"""
    x = np.asarray(xyz, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {x.shape}")
    l1, = ax.plot(t_s, x[:, 0], linewidth=linewidth, color=colors[0])
    l2, = ax.plot(t_s, x[:, 1], linewidth=linewidth, color=colors[1])
    l3, = ax.plot(t_s, x[:, 2], linewidth=linewidth, color=colors[2])
    return l1, l2, l3


def add_xyz_legend(
    ax: plt.Axes,
    lines: Tuple[plt.Line2D, plt.Line2D, plt.Line2D],
    layout: Imu3RowLayout,
    *,
    labels: Tuple[str, str, str] = IMU_AXIS_LABELS,
    loc: str = "upper right",
) -> None:
    """为三轴曲线添加一次性 legend。"""
    leg = ax.legend(list(lines), list(labels), loc=loc, frameon=False, fontsize=layout.legend_fs())
    apply_legend_style(leg)
