"""
模块名称：Power 绘图

模块职责：
生成面向论文与数据审查的电机功率图，
统一支持“总功率 + 8 电机功率 excerpt”主图与旧版电流 QA 面板图。

主要功能：
1. 将 `PowerFrame.power_motors` 绘制为“顶部总功率 + 下方 4×2 单电机功率”同步观测图。
2. 支持按显式时间窗或“总功率峰值窗口”自动截取代表性 excerpt。
3. 保留 `PowerFrame.curr_motors` 的 4×2 电流 QA 图，供调试而非论文主图使用。
4. 对单点/空序列跳过折线绘制，并把原因写到 sidecar 记录文件。

数据流：
PowerFrame
    ↓
时间窗选择（full / peak_total_power / explicit window）
    ↓
统一 style token + 总功率/单电机功率布局
    ↓
论文主图 `power_sync_overview_8motors.png`
以及可选 QA 图 `power_currents_8motors.png`

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.style.imu_style

备注：
- 论文主图默认使用功率（W），不再让 current-only 面板承担主文证据职责。
- 图面延续仓库统一科研风格：白底、无网格、无冗余标题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from uwnav_dynamics.io.readers.power_reader import PowerFrame
from uwnav_dynamics.viz.style.imu_style import Imu3RowLayout
from uwnav_dynamics.viz.style.sci_style import (
    align_ylabels,
    apply_axes_style,
    get_figure_size,
    plot_or_record_series,
    save_figure,
    SparsePlotRecorder,
    setup_mpl,
)


@dataclass(frozen=True)
class PowerPlotPaths:
    """Power 图产物的标准路径集合。"""
    run_dir: Path
    plots_dir: Path
    overview_png: Path
    currents_png: Path
    overview_warnings_txt: Path
    currents_warnings_txt: Path


_POWER_PANEL_COLORS: tuple[str, ...] = (
    "#2C7FB8",
    "#F28E2B",
    "#1B9E77",
    "#E76F51",
    "#5C6BC0",
    "#D65A7A",
    "#4C956C",
    "#D4A72C",
)


def _resolve_out_dirs(power: PowerFrame, out_root: str | Path) -> PowerPlotPaths:
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / power.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return PowerPlotPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        overview_png=plots_dir / "power_sync_overview_8motors.png",
        currents_png=plots_dir / "power_currents_8motors.png",
        overview_warnings_txt=plots_dir / "power_sync_overview_8motors.plot_warnings.txt",
        currents_warnings_txt=plots_dir / "power_currents_8motors.plot_warnings.txt",
    )


def resolve_power_time_window(
    power: PowerFrame,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    window_s: float = 60.0,
    window_mode: Literal["full", "peak_total_power"] = "peak_total_power",
) -> tuple[float, float]:
    """解析功率图使用的时间窗。"""
    t = np.asarray(power.t_s, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("[POWER-PLOT] empty PowerFrame (no samples).")

    if t_start is not None or t_end is not None:
        lo = float(t[0]) if t_start is None else float(t_start)
        hi = float(t[-1]) if t_end is None else float(t_end)
    elif window_mode == "full" or float(window_s) <= 0.0 or float(window_s) >= float(t[-1] - t[0]):
        lo = float(t[0])
        hi = float(t[-1])
    else:
        total_power = np.nansum(np.asarray(power.power_motors, dtype=float), axis=1)
        energy_prefix = np.concatenate([[0.0], np.cumsum(np.nan_to_num(total_power, nan=0.0), dtype=float)])
        best_i = 0
        best_j = t.size
        best_energy = float("-inf")
        for i in range(t.size):
            j = int(np.searchsorted(t, t[i] + float(window_s), side="right"))
            energy = float(energy_prefix[j] - energy_prefix[i])
            if energy > best_energy:
                best_i = i
                best_j = max(i + 1, j)
                best_energy = energy
        lo = float(t[best_i])
        hi = float(t[min(best_j - 1, t.size - 1)])

    if hi < lo:
        raise ValueError(f"[POWER-PLOT] invalid time window: start={lo}, end={hi}")
    return lo, hi


def save_power_sync_overview_8motors(
    power: PowerFrame,
    *,
    out_root: str | Path = "out/power_plots",
    use_rel_time: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
    window_s: float = 60.0,
    window_mode: Literal["full", "peak_total_power"] = "peak_total_power",
) -> Path:
    """保存“总功率 + 8 电机功率”的同步观测总览图。"""
    setup_mpl()
    paths = _resolve_out_dirs(power, out_root)

    t = np.asarray(power.t_s, dtype=float).reshape(-1)
    p_motor = np.asarray(power.power_motors, dtype=float)
    if p_motor.ndim != 2 or p_motor.shape[1] != 8:
        raise ValueError(f"[POWER-PLOT] power_motors must have shape (N, 8), got {p_motor.shape}")

    win_lo, win_hi = resolve_power_time_window(
        power,
        t_start=t_start,
        t_end=t_end,
        window_s=window_s,
        window_mode=window_mode,
    )
    keep = (t >= win_lo) & (t <= win_hi)
    if not np.any(keep):
        raise ValueError(f"[POWER-PLOT] selected time window has no samples: [{win_lo}, {win_hi}]")

    t_win = t[keep]
    p_win = p_motor[keep]
    t_plot = t_win - float(t_win[0]) if use_rel_time else t_win
    total_power = np.nansum(p_win, axis=1)
    recorder = SparsePlotRecorder()

    layout = Imu3RowLayout()
    fig_w, fig_h = get_figure_size("sensor_4x2")
    fig = plt.figure(figsize=(fig_w, fig_h * 1.34), dpi=layout.dpi)
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.48, 1.0, 1.0, 1.0, 1.0],
        hspace=0.28,
        wspace=0.18,
    )
    ax_total = fig.add_subplot(gs[0, :])
    motor_axes = [fig.add_subplot(gs[row, col], sharex=ax_total) for row in range(1, 5) for col in range(2)]

    total_lines = plot_or_record_series(
        ax_total,
        t_plot,
        total_power,
        panel="Power overview / total",
        series="total power",
        color="#2F3B52",
        recorder=recorder,
        skip_message="Total power skipped: fewer than 2 samples",
        linewidth=1.55,
        zorder=4,
    )
    if len(total_lines) > 0:
        ax_total.fill_between(t_plot, 0.0, total_power, color="#B8C4CF", alpha=0.28, zorder=2)
    ax_total.set_ylabel("Total power (W)")
    apply_axes_style(ax_total, grid=False)

    for i, ax in enumerate(motor_axes):
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
        color = _POWER_PANEL_COLORS[i % len(_POWER_PANEL_COLORS)]
        motor_lines = plot_or_record_series(
            ax,
            t_plot,
            p_win[:, i],
            panel=f"Power overview / T{i + 1}",
            series=f"thruster {i + 1} power",
            color=color,
            recorder=recorder,
            skip_message=f"T{i + 1} skipped: fewer than 2 samples",
            linewidth=layout.lw(),
            zorder=4,
        )
        if len(motor_lines) > 0:
            ax.fill_between(t_plot, 0.0, p_win[:, i], color=color, alpha=0.34, zorder=2)
        ax.set_ylabel(f"T{i + 1} (W)")
        ax.text(
            0.02,
            0.84,
            f"T{i + 1}",
            transform=ax.transAxes,
            color=color,
            fontsize=layout.label_fs(),
            fontweight="bold",
            ha="left",
            va="center",
        )

    for ax in [ax_total, *motor_axes[:-2]]:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in motor_axes[-2:]:
        ax.set_xlabel("Time (s)")

    align_ylabels([ax_total, *motor_axes])
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.07, top=0.985)
    save_figure(fig, paths.overview_png.with_suffix(""), fmt="png")
    plt.close(fig)
    recorder.write_text(paths.overview_warnings_txt)
    return paths.overview_png


def save_power_currents_8motors(
    power: PowerFrame,
    *,
    out_root: str | Path = "out/power_plots",
    use_rel_time: bool = False,
) -> Path:
    """保存 8 路电机电流的 4×2 QA 面板图。"""
    setup_mpl()
    paths = _resolve_out_dirs(power, out_root)

    t = np.asarray(power.t_s, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("[POWER-PLOT] empty PowerFrame (no samples).")

    t_plot = t - float(t[0]) if use_rel_time else t
    curr = np.asarray(power.curr_motors, dtype=float)
    if curr.ndim != 2 or curr.shape[1] != 8:
        raise ValueError(f"[POWER-PLOT] curr_motors must have shape (N, 8), got {curr.shape}")
    recorder = SparsePlotRecorder()

    layout = Imu3RowLayout()
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=get_figure_size("sensor_4x2"), dpi=layout.dpi)
    axes_flat = axes.ravel()

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
        color = _POWER_PANEL_COLORS[i % len(_POWER_PANEL_COLORS)]
        plot_or_record_series(
            ax,
            t_plot,
            curr[:, i],
            panel=f"Power currents / M{i + 1}",
            series=f"motor {i + 1} current",
            color=color,
            recorder=recorder,
            skip_message=f"M{i + 1} skipped: fewer than 2 samples",
            linewidth=layout.lw(),
        )
        ax.set_ylabel(f"M{i + 1} (A)")

    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98, hspace=0.22, wspace=0.18)
    fig.savefig(paths.currents_png)
    plt.close(fig)
    recorder.write_text(paths.currents_warnings_txt)
    return paths.currents_png
