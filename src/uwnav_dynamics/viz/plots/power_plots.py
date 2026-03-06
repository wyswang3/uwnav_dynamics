"""
模块名称：Power 绘图

模块职责：
生成 8 路电机电流的科研风格面板图，
用于数据质量检查与动力学辅助学习的可视化审查。

主要功能：
1. 将 `PowerFrame.curr_motors` 绘制为 4×2 共享时间轴面板。
2. 移除重复 legend，通过 y 轴标签承载每个电机子图的语义。
3. 统一使用与 IMU / DVL 相同的字体、边距和坐标轴规则。

数据流：
PowerFrame
    ↓
统一 style token + 4×2 面板布局
    ↓
无标题、无重复 legend 的 figure
    ↓
out/.../plots/power_currents_8motors.png

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.style.imu_style

备注：
- 每个子图只展示一个电机电流，因此默认不放 legend。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uwnav_dynamics.io.readers.power_reader import PowerFrame
from uwnav_dynamics.viz.style.imu_style import Imu3RowLayout
from uwnav_dynamics.viz.style.sci_style import apply_axes_style, get_figure_size, setup_mpl


@dataclass(frozen=True)
class PowerPlotPaths:
    run_dir: Path
    plots_dir: Path
    currents_png: Path


def _resolve_out_dirs(power: PowerFrame, out_root: str | Path) -> PowerPlotPaths:
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / power.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return PowerPlotPaths(run_dir=run_dir, plots_dir=plots_dir, currents_png=plots_dir / "power_currents_8motors.png")


def save_power_currents_8motors(
    power: PowerFrame,
    *,
    out_root: str | Path = "out/power_plots",
    use_rel_time: bool = False,
) -> Path:
    setup_mpl()
    paths = _resolve_out_dirs(power, out_root)

    t = np.asarray(power.t_s, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("[POWER-PLOT] empty PowerFrame (no samples).")

    t_plot = t - float(t[0]) if use_rel_time else t
    curr = np.asarray(power.curr_motors, dtype=float)
    if curr.ndim != 2 or curr.shape[1] != 8:
        raise ValueError(f"[POWER-PLOT] curr_motors must have shape (N, 8), got {curr.shape}")

    layout = Imu3RowLayout()
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=get_figure_size("sensor_4x2"), dpi=layout.dpi)
    axes_flat = axes.ravel()

    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = prop_cycle.by_key().get("color", None) if prop_cycle is not None else None

    for i, ax in enumerate(axes_flat):
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
        color = colors[i % len(colors)] if colors else None
        ax.plot(t_plot, curr[:, i], linewidth=layout.lw(), color=color)
        ax.set_ylabel(f"M{i + 1} (A)")

    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98, hspace=0.22, wspace=0.18)
    fig.savefig(paths.currents_png)
    plt.close(fig)
    return paths.currents_png
