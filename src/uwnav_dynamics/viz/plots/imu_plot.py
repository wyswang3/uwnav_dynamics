"""
模块名称：IMU 绘图

模块职责：
负责生成原始 IMU 与预处理后 IMU 的科研风格图像，
统一输出 3 行共享时间轴的波形图与采样间隔图。

主要功能：
1. 读取 `ImuFrame` 中的原始 9 轴数据并输出三行图。
2. 从预处理后的 IMU CSV 中提取体坐标加速度、角速度与姿态并输出三行图。
3. 输出 IMU 采样间隔 `Δt` 图，用于诊断时间戳稳定性。

数据流：
ImuFrame 或 *_proc.csv
    ↓
统一 style token + IMU 三行布局
    ↓
无标题、共享 x 轴、最小 legend 的 figure
    ↓
out/.../plots/imu_raw_9axis.png / imu_dt.png / imu_proc_3rows.png

依赖模块：
- pandas
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.style.imu_style

备注：
- 行语义由 y 轴标签承担，不再使用标题。
- 多轴语义的 legend 仅保留一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from uwnav_dynamics.io.readers.imu_reader import ImuFrame
from uwnav_dynamics.viz.style.imu_style import (
    Imu3RowLayout,
    add_xyz_legend,
    finalize_imu_axes,
    make_imu_3rows_canvas,
    plot_xyz_lines,
    set_y_ticks_pretty_3,
)
from uwnav_dynamics.viz.style.sci_style import apply_axes_style, get_figure_size, setup_mpl


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
    setup_mpl()
    paths = _resolve_out_dirs(imu, out_root)

    t = imu.t_rel_s if use_rel_time else imu.t_s
    fig, axes, layout = make_imu_3rows_canvas(layout)
    lw = layout.lw()

    lines_acc = plot_xyz_lines(axes[0], t, imu.acc_g, linewidth=lw)
    axes[0].set_ylabel("Acc (g)")
    add_xyz_legend(axes[0], lines_acc, layout)

    plot_xyz_lines(axes[1], t, imu.gyro_deg_s, linewidth=lw)
    axes[1].set_ylabel("Gyro (deg/s)")

    plot_xyz_lines(axes[2], t, imu.ang_deg, linewidth=lw)
    axes[2].set_ylabel("Att (deg)")

    for ax in axes:
        apply_axes_style(ax, grid=False)

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
    setup_mpl()
    paths = _resolve_out_dirs(imu, out_root)

    t = imu.t_rel_s if use_rel_time else imu.t_s
    xlab = "Time (s)"
    t_mid = 0.5 * (t[1:] + t[:-1])
    dt = imu.dt_s

    fig, ax = plt.subplots(1, 1, figsize=get_figure_size("single"))
    ax.plot(t_mid, dt, linewidth=1.0, color="#30343A")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\Delta t$ (s)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    apply_axes_style(ax, grid=False)
    set_y_ticks_pretty_3(ax, y_pad_frac=0.05)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.98)
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
    p1 = save_imu_raw_9axis(imu, out_root=out_root, layout=layout, use_rel_time=use_rel_time)
    p2 = save_imu_dt(imu, out_root=out_root, use_rel_time=True)
    return p1, p2


def _pick_vec3(
    df: pd.DataFrame,
    cand_triplets: Sequence[Tuple[str, str, str]],
    desc: str,
) -> np.ndarray:
    for cols in cand_triplets:
        if all(c in df.columns for c in cols):
            arr = df[list(cols)].to_numpy(dtype=float)
            if np.isfinite(arr).any():
                print(f"[IMU-PROC-PLOT] use {desc} from columns {cols}")
                return arr
            print(f"[IMU-PROC-PLOT] columns {cols} exist for {desc} but all-NaN, trying next...")
    print(f"[IMU-PROC-PLOT] WARNING: no valid columns found for {desc}, using empty array.")
    return np.full((df.shape[0], 3), np.nan, dtype=float)


def save_imu_proc_3rows_from_csv(
    proc_csv: str | Path,
    *,
    out_root: str | Path = "out/imu_plots_proc",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Path:
    setup_mpl()

    proc_path = Path(proc_csv).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    run_dir = out_root / proc_path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "imu_proc_3rows.png"

    df = pd.read_csv(proc_path)
    if "t_s" not in df.columns:
        raise ValueError(f"{proc_path} 缺少列 't_s'，请确认是否为 pipeline 输出的 *_proc.csv")

    t = df["t_s"].to_numpy(dtype=float)
    t_plot = t - t[0] if use_rel_time else t

    acc_body = _pick_vec3(
        df,
        cand_triplets=[
            ("AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2"),
            ("AccE_enu_mps2", "AccN_enu_mps2", "AccU_enu_mps2"),
            ("AccX", "AccY", "AccZ"),
        ],
        desc="body linear acceleration",
    )
    gyro_body = _pick_vec3(
        df,
        cand_triplets=[
            ("GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"),
            ("GyroX", "GyroY", "GyroZ"),
        ],
        desc="body angular rate",
    )

    if all(c in df.columns for c in ("roll_rad", "pitch_rad", "yaw_rad")):
        att_deg = np.column_stack(
            [
                np.rad2deg(df["roll_rad"].to_numpy(dtype=float)),
                np.rad2deg(df["pitch_rad"].to_numpy(dtype=float)),
                np.rad2deg(df["yaw_rad"].to_numpy(dtype=float)),
            ]
        )
        print("[IMU-PROC-PLOT] use attitude from roll_rad/pitch_rad/yaw_rad")
    elif all(c in df.columns for c in ("AngX", "AngY", "AngZ")):
        att_deg = df[["AngX", "AngY", "AngZ"]].to_numpy(dtype=float)
        print("[IMU-PROC-PLOT] use attitude from AngX/AngY/AngZ (deg)")
    else:
        att_deg = np.full((df.shape[0], 3), np.nan, dtype=float)
        print("[IMU-PROC-PLOT] WARNING: no columns for attitude, using NaN")

    fig, axes, layout = make_imu_3rows_canvas(layout)
    lw = layout.lw()

    lines_acc = plot_xyz_lines(axes[0], t_plot, acc_body, linewidth=lw)
    axes[0].set_ylabel(r"Acc (m/s$^2$)")
    add_xyz_legend(axes[0], lines_acc, layout)

    plot_xyz_lines(axes[1], t_plot, gyro_body, linewidth=lw)
    axes[1].set_ylabel("Gyro (rad/s)")

    plot_xyz_lines(axes[2], t_plot, att_deg, linewidth=lw)
    axes[2].set_ylabel("Att (deg)")

    for ax in axes:
        apply_axes_style(ax, grid=False)

    finalize_imu_axes(axes, y_pad_frac=layout.y_pad_frac)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
