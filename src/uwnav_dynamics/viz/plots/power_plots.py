# src/uwnav_dynamics/viz/plots/power_plots.py
from __future__ import annotations

"""
uwnav_dynamics.viz.plots.power_plots

电机电流可视化（Volt32 / motor power logger）。

设计目标：
  - 一张图包含 8 个电机的电流曲线；
  - 使用与 IMU / DVL 相同的绘图风格（sci_style + Imu3RowLayout）；
  - 适合作为“动力学辅助学习数据”的 sanity check 图。

输入：
  - PowerFrame（来自 uwnav_dynamics.io.readers.power_reader.read_power_csv）

输出：
  - out_root/<csv_stem>/plots/power_currents_8motors.png
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from uwnav_dynamics.io.readers.power_reader import PowerFrame
from uwnav_dynamics.viz.style.sci_style import setup_mpl
from uwnav_dynamics.viz.style.imu_style import Imu3RowLayout


# ----------------------------------------------------------------------
# 路径管理
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PowerPlotPaths:
    run_dir: Path
    plots_dir: Path
    currents_png: Path


def _resolve_out_dirs(power: PowerFrame, out_root: str | Path) -> PowerPlotPaths:
    """
    根据 PowerFrame.path 决定输出目录结构：

      out_root/
        <csv_stem>/
          plots/
            power_currents_8motors.png
    """
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / power.path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return PowerPlotPaths(
        run_dir=run_dir,
        plots_dir=plots_dir,
        currents_png=plots_dir / "power_currents_8motors.png",
    )


# ----------------------------------------------------------------------
# 主绘图函数
# ----------------------------------------------------------------------
def save_power_currents_8motors(
    power: PowerFrame,
    *,
    out_root: str | Path = "out/power_plots",
    use_rel_time: bool = False,
) -> Path:
    """
    绘制 8 个电机电流，4×2 子图，风格与其他传感器图统一。
    """
    setup_mpl()
    paths = _resolve_out_dirs(power, out_root)

    t = np.asarray(power.t_s, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("[POWER-PLOT] empty PowerFrame (no samples).")

    if use_rel_time:
        t_plot = t - float(t[0])
        xlab = "Time (s)"
    else:
        t_plot = t
        xlab = f"Time (s) [{power.time_col}]"

    curr = np.asarray(power.curr_motors, dtype=float)
    if curr.ndim != 2 or curr.shape[1] != 8:
        raise ValueError(
            f"[POWER-PLOT] curr_motors must have shape (N, 8), got {curr.shape}"
        )

    layout = Imu3RowLayout()

    # 长宽比：略宽、略扁一点，适合 4×2 面板
    fig_w = layout.fig_w_in * 1.2
    fig_h = layout.fig_h_in * 0.9

    fig, axes = plt.subplots(
        4,
        2,
        sharex=True,
        figsize=(fig_w, fig_h),
        dpi=layout.dpi,
        gridspec_kw={"hspace": 0.25, "wspace": 0.20},
    )
    axes_flat = axes.ravel()

    # 读取 sci_style 配置好的全局颜色循环
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = None
    if prop_cycle is not None:
        try:
            colors = prop_cycle.by_key().get("color", None)
        except Exception:
            colors = None

    # 统一 tick 样式
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
        if i < 6:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    # 每个电机一个子图：不写轴标题和子图标题，颜色手动从全局 color cycle 里取
    for motor_idx in range(8):
        ax = axes_flat[motor_idx]
        y = curr[:, motor_idx]

        if colors and len(colors) > 0:
            color = colors[motor_idx % len(colors)]
        else:
            color = None  # fallback 给 Matplotlib 自己选

        ax.plot(
            t_plot,
            y,
            linewidth=layout.lw(),
            label=f"Motor {motor_idx + 1} (A)",  # 这里从 1 开始编号
            color=color,
        )

        # 图例中写 motor+单位，占位很小
        ax.legend(
            fontsize=layout.legend_fs(),
            loc="upper right",
        )

        ax.grid(True, which="both", alpha=0.3)

    # 统一 x 轴标签：只在最底下一行显示
    for ax in axes[-1, :]:
        ax.set_xlabel(xlab, fontsize=layout.label_fs())

    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.97])
    fig.savefig(paths.currents_png)
    plt.close(fig)

    return paths.currents_png

