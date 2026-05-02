"""
模块名称：Power 绘图

模块职责：
生成面向论文与数据审查的电机功率图，
统一支持参考 `Push the machine data excerpt.png` 视觉语言的
“总功率 + 8 电机功率 excerpt”主图与旧版电流 QA 面板图。

主要功能：
1. 将 `PowerFrame.power_motors` 绘制为“顶部总功率 + 下方 4×2 单电机功率”同步观测图。
2. 支持按显式时间窗或“总功率峰值窗口”自动截取代表性 excerpt。
3. 主图采用“淡原始轨迹 + 平滑主线 + 面积填充 + 面板统计”的分析型视觉层级。
4. 保留 `PowerFrame.curr_motors` 的 4×2 电流 QA 图，供调试而非论文主图使用。
5. 对单点/空序列跳过折线绘制，并把原因写到 sidecar 记录文件。

数据流：
PowerFrame
    ↓
时间窗选择（full / peak_total_power / explicit window）
    ↓
Push-machine excerpt style token + 总功率/单电机功率布局
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
- 论文主图采用白底、粗黑轴线、浅色面积填充和面板内标签，
  以匹配参考图的“机器功率 excerpt”视觉语言。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from uwnav_dynamics.io.readers.power_reader import PowerFrame
from uwnav_dynamics.viz.style.imu_style import Imu3RowLayout
from uwnav_dynamics.viz.style.sci_style import (
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
    "#80D1C4",
    "#E8A000",
    "#B9B6D8",
    "#FF7D73",
    "#7EB4D8",
    "#FFB55E",
    "#A9DC5B",
    "#F7B9D7",
)
_POWER_TOTAL_LINE = "#303234"
_POWER_TOTAL_FILL = "#ECECEC"
_POWER_AXIS = "#000000"
_POWER_TEXT = "#303234"


@dataclass(frozen=True)
class PowerExcerptStyle:
    """参考 Push-machine excerpt 的功率主图视觉参数。"""
    fig_w_in: float = 13.06
    fig_h_in: float = 9.05
    dpi: int = 150
    axis_lw: float = 1.45
    major_tick_w: float = 1.45
    major_tick_len: float = 5.6
    total_lw: float = 2.25
    motor_lw: float = 1.70
    raw_lw: float = 0.45
    raw_alpha: float = 0.08
    total_fill_alpha: float = 0.72
    motor_fill_alpha: float = 0.56
    label_fs: float = 15.0
    tick_fs: float = 12.5
    panel_fs: float = 15.5
    annotate_fs: float = 13.5
    stat_fs: float = 7.2
    marker_size: float = 18.0
    marker_lw: float = 1.05
    x_nbins: int = 7


_POWER_EXCERPT_STYLE = PowerExcerptStyle()


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
    window_s: float = 600.0,
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


def _apply_machine_excerpt_axes(
    ax: plt.Axes,
    *,
    show_bottom: bool,
    show_left: bool = True,
    style: PowerExcerptStyle = _POWER_EXCERPT_STYLE,
) -> None:
    """应用参考图的粗黑轴线、外向刻度和无网格风格。"""
    ax.set_facecolor("#FFFFFF")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_left)
    ax.spines["bottom"].set_visible(show_bottom)
    for spine_name in ("left", "bottom"):
        spine = ax.spines[spine_name]
        if spine.get_visible():
            spine.set_color(_POWER_AXIS)
            spine.set_linewidth(style.axis_lw)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=style.major_tick_w,
        length=style.major_tick_len,
        color=_POWER_AXIS,
        labelcolor=_POWER_AXIS,
        labelsize=style.tick_fs,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=style.x_nbins, prune=None))


def _nice_upper_limit(y: np.ndarray, *, floor: float, step: float) -> float:
    """为功率面板生成接近参考图的整齐上界。"""
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    ymax = max(float(np.nanmax(finite)), float(floor))
    return float(np.ceil(ymax / float(step)) * float(step))


def _smooth_series_by_time(t: np.ndarray, y: np.ndarray, *, smooth_s: float) -> np.ndarray:
    """按近似时间窗做有限值感知的移动平均，仅用于绘图视觉层级。"""
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0 or float(smooth_s) <= 0.0:
        return values

    t_arr = np.asarray(t, dtype=float).reshape(-1)
    if t_arr.size != values.size or t_arr.size < 3:
        return values

    dt = np.diff(t_arr[np.isfinite(t_arr)])
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return values

    win = int(round(float(smooth_s) / float(np.median(dt))))
    win = max(1, min(win, max(1, values.size // 3)))
    if win <= 1:
        return values
    if win % 2 == 0:
        win += 1

    kernel = np.ones(win, dtype=float)
    finite = np.isfinite(values)
    weighted = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.divide(weighted, counts, out=np.full_like(values, np.nan), where=counts > 0.0)
    return out


def _peak_marker_indices(y: np.ndarray, *, max_markers: int = 3) -> np.ndarray:
    """选择少量峰值点，用空心圆提示局部高功率片段。"""
    values = np.asarray(y, dtype=float).reshape(-1)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return np.asarray([], dtype=int)

    positive = finite[values[finite] > 0.0]
    candidates = positive if positive.size > 0 else finite
    order = candidates[np.argsort(values[candidates])[::-1]]
    min_spacing = max(1, values.size // 18)
    chosen: list[int] = []
    for idx in order:
        if all(abs(int(idx) - int(prev)) >= min_spacing for prev in chosen):
            chosen.append(int(idx))
        if len(chosen) >= int(max_markers):
            break
    return np.asarray(sorted(chosen), dtype=int)


def _active_fraction(y: np.ndarray, *, threshold_w: float) -> float:
    """估计窗口内电机处于有效功率输出状态的比例。"""
    values = np.asarray(y, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.count_nonzero(finite > float(threshold_w)) / finite.size)


def _format_power_stats(y: np.ndarray, *, threshold_w: float) -> str:
    """生成面板内简短统计文本。"""
    values = np.asarray(y, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite data"
    peak = float(np.nanmax(finite))
    mean = float(np.nanmean(finite))
    active = 100.0 * _active_fraction(finite, threshold_w=threshold_w)
    return f"p {peak:.0f} W   μ {mean:.0f} W   on {active:.0f}%"


def save_power_sync_overview_8motors(
    power: PowerFrame,
    *,
    out_root: str | Path = "out/power_plots",
    use_rel_time: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
    window_s: float = 600.0,
    window_mode: Literal["full", "peak_total_power"] = "peak_total_power",
    smooth_s: float = 12.0,
) -> Path:
    """保存“总功率 + 8 电机功率”的同步观测总览图。"""
    setup_mpl()
    paths = _resolve_out_dirs(power, out_root)
    excerpt_style = _POWER_EXCERPT_STYLE

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
    total_power_smooth = _smooth_series_by_time(t_win, total_power, smooth_s=float(smooth_s))
    p_smooth = np.column_stack(
        [_smooth_series_by_time(t_win, p_win[:, i], smooth_s=float(smooth_s)) for i in range(8)]
    )
    recorder = SparsePlotRecorder()

    fig = plt.figure(
        figsize=(excerpt_style.fig_w_in, excerpt_style.fig_h_in),
        dpi=excerpt_style.dpi,
    )
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.70, 0.90, 0.90, 0.90, 0.90],
        hspace=0.40,
        wspace=0.20,
    )
    ax_total = fig.add_subplot(gs[0, :])
    motor_axes = [fig.add_subplot(gs[row, col], sharex=ax_total) for row in range(1, 5) for col in range(2)]

    total_lines = plot_or_record_series(
        ax_total,
        t_plot,
        total_power_smooth,
        panel="Power overview / total",
        series="smoothed total power",
        color=_POWER_TOTAL_LINE,
        recorder=recorder,
        skip_message="Total power skipped: fewer than 2 samples",
        linewidth=excerpt_style.total_lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )
    if len(total_lines) > 0:
        ax_total.plot(
            t_plot,
            total_power,
            color=_POWER_TOTAL_LINE,
            linewidth=excerpt_style.raw_lw,
            alpha=excerpt_style.raw_alpha,
            zorder=3,
        )
        ax_total.fill_between(
            t_plot,
            0.0,
            total_power_smooth,
            color=_POWER_TOTAL_FILL,
            alpha=excerpt_style.total_fill_alpha,
            zorder=2,
        )
    ax_total.set_ylabel("Total Power (W)", fontsize=excerpt_style.label_fs, labelpad=16.0)
    total_ymax = _nice_upper_limit(total_power, floor=50.0, step=50.0)
    ax_total.set_ylim(0.0, total_ymax)
    ax_total.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    _apply_machine_excerpt_axes(ax_total, show_bottom=True, style=excerpt_style)
    ax_total.text(
        0.86,
        0.70,
        "Total Power",
        transform=ax_total.transAxes,
        color=_POWER_TEXT,
        fontsize=excerpt_style.annotate_fs,
        ha="left",
        va="center",
        bbox={
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 1.0,
        },
    )
    ax_total.text(
        0.02,
        0.90,
        _format_power_stats(total_power, threshold_w=max(50.0, 0.05 * float(np.nanmax(total_power)))),
        transform=ax_total.transAxes,
        color="#5A5A5A",
        fontsize=excerpt_style.stat_fs,
        ha="left",
        va="top",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.76, "pad": 0.8},
    )

    for i, ax in enumerate(motor_axes):
        _apply_machine_excerpt_axes(
            ax,
            show_bottom=i >= 6,
            style=excerpt_style,
        )
        color = _POWER_PANEL_COLORS[i % len(_POWER_PANEL_COLORS)]
        motor_lines = plot_or_record_series(
            ax,
            t_plot,
            p_smooth[:, i],
            panel=f"Power overview / T{i + 1}",
            series=f"thruster {i + 1} smoothed power",
            color=color,
            recorder=recorder,
            skip_message=f"T{i + 1} skipped: fewer than 2 samples",
            linewidth=excerpt_style.motor_lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
        if len(motor_lines) > 0:
            ax.plot(
                t_plot,
                p_win[:, i],
                color=color,
                linewidth=excerpt_style.raw_lw,
                alpha=excerpt_style.raw_alpha,
                zorder=3,
            )
            ax.fill_between(t_plot, 0.0, p_smooth[:, i], color=color, alpha=excerpt_style.motor_fill_alpha, zorder=2)
            peak_idx = _peak_marker_indices(p_smooth[:, i], max_markers=3)
            if peak_idx.size > 0:
                ax.scatter(
                    t_plot[peak_idx],
                    p_smooth[peak_idx, i],
                    s=excerpt_style.marker_size,
                    facecolors="#FFFFFF",
                    edgecolors=color,
                    linewidths=excerpt_style.marker_lw,
                    zorder=5,
                )
        motor_ymax = _nice_upper_limit(p_win[:, i], floor=100.0, step=100.0)
        ax.set_ylim(0.0, motor_ymax)
        ax.set_yticks([0.0, motor_ymax])
        ax.text(
            0.02,
            0.89,
            f"T{i + 1}",
            transform=ax.transAxes,
            color=color,
            fontsize=excerpt_style.panel_fs,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax.text(
            0.02,
            0.10,
            _format_power_stats(p_win[:, i], threshold_w=max(20.0, 0.05 * motor_ymax)),
            transform=ax.transAxes,
            color="#5A5A5A",
            fontsize=excerpt_style.stat_fs,
            ha="left",
            va="bottom",
            bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.68, "pad": 0.6},
        )

    for ax in [ax_total, *motor_axes[:-2]]:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in motor_axes[-2:]:
        ax.set_xlabel("")
    fig.supxlabel("Time (s)", fontsize=excerpt_style.label_fs, y=0.028)
    fig.supylabel("Power (W)", fontsize=excerpt_style.label_fs, x=0.025)

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.120, top=0.965)
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
