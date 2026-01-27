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
from typing import Optional, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd  # 新增

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

    Row 1: Acceleration (g)
    Row 2: Angular rate (deg/s)
    Row 3: Attitude (deg)

    纵轴不写任何文字，仅通过每一行的标题说明物理量，
    底部只在最后一行给一个统一的 Time (s)。
    """
    setup_mpl()

    paths = _resolve_out_dirs(imu, out_root)

    t = imu.t_rel_s if use_rel_time else imu.t_s
    # 为了简洁，绝对时间也统一写 Time (s)
    xlab = "Time (s)"

    fig, axes, layout = make_imu_3rows_canvas(layout)
    # 适合作为论文单列图的比例
    fig.set_size_inches(5.0, 4.0)

    lw = layout.lw()

    # Row 1: Acceleration (g) —— 带图例，无 y label
    ax = axes[0]
    lines_acc = plot_xyz_lines(ax, t, imu.acc_g, linewidth=lw)
    ax.set_title("Acceleration (g)", fontsize=layout.title_fs())
    add_xyz_legend(ax, lines_acc, layout)

    # Row 2: Angular rate (deg/s) —— 无图例、无 y label
    ax = axes[1]
    plot_xyz_lines(ax, t, imu.gyro_deg_s, linewidth=lw)
    ax.set_title("Angular rate (deg/s)", fontsize=layout.title_fs())

    # Row 3: Attitude (deg) —— 无 y label，仅 x 轴标签
    ax = axes[2]
    plot_xyz_lines(ax, t, imu.ang_deg, linewidth=lw)
    ax.set_title("Attitude (deg)", fontsize=layout.title_fs())
    ax.set_xlabel(xlab, fontsize=layout.label_fs())

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

def _pick_vec3(
    df: pd.DataFrame,
    cand_triplets: Sequence[Tuple[str, str, str]],
    desc: str,
) -> np.ndarray:
    """
    在若干候选列三元组中，选择第一个“存在且不全为 NaN”的 3 维向量列。

    cand_triplets: [("AccX_body_mps2","AccY_body_mps2","AccZ_body_mps2"), ("AccX","AccY","AccZ"), ...]
    desc: 用于打印诊断信息的文字说明。
    """
    for cols in cand_triplets:
        if all(c in df.columns for c in cols):
            arr = df[list(cols)].to_numpy(dtype=float)
            if np.isfinite(arr).any():
                # 找到可用数据
                print(f"[IMU-PROC-PLOT] use {desc} from columns {cols}")
                return arr
            else:
                print(f"[IMU-PROC-PLOT] columns {cols} exist for {desc} but all-NaN, trying next...")
    # 如果所有候选都失败，返回全 NaN（长度=0 由调用方处理）
    print(f"[IMU-PROC-PLOT] WARNING: no valid columns found for {desc}, using empty array.")
    return np.full((df.shape[0], 3), np.nan, dtype=float)

def save_imu_proc_3rows_from_csv(
    proc_csv: str | Path,
    *,
    out_root: str | Path = "out/imu_plots_proc",
    layout: Imu3RowLayout = Imu3RowLayout(),
    use_rel_time: bool = False,
) -> Path:
    """
    预处理后的 IMU 三行图：

      Row 1: Acceleration (m/s²)
      Row 2: Angular rate (rad/s)
      Row 3: Attitude (deg)

    不使用任何 y 轴文字，只在每一行上方给简短标题，
    底部仅在最后一行给统一的 Time (s)。
    """
    setup_mpl()

    proc_path = Path(proc_csv).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()

    run_dir = out_root / proc_path.stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "imu_proc_3rows.png"

    df = pd.read_csv(proc_path)

    if "t_s" not in df.columns:
        raise ValueError(
            f"{proc_path} 缺少列 't_s'，请确认是否为 pipeline 输出的 *_proc.csv"
        )

    t = df["t_s"].to_numpy(dtype=float)
    if use_rel_time:
        t_plot = t - t[0]
        xlab = "Time (s)"
    else:
        t_plot = t
        # 为了简洁，绝对时间也统一写 Time (s)
        xlab = "Time (s)"

    # ---------- 1) 体坐标线加速度 ----------
    acc_body = _pick_vec3(
        df,
        cand_triplets=[
            ("AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2"),
            ("AccE_enu_mps2", "AccN_enu_mps2", "AccU_enu_mps2"),
            ("AccX", "AccY", "AccZ"),
        ],
        desc="body linear acceleration",
    )

    # ---------- 2) 体坐标角速度 ----------
    gyro_body = _pick_vec3(
        df,
        cand_triplets=[
            ("GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"),
            ("GyroX", "GyroY", "GyroZ"),
        ],
        desc="body angular rate",
    )

    # ---------- 3) 姿态角（绘图用 deg） ----------
    if all(c in df.columns for c in ("roll_rad", "pitch_rad", "yaw_rad")):
        roll_rad = df["roll_rad"].to_numpy(dtype=float)
        pitch_rad = df["pitch_rad"].to_numpy(dtype=float)
        yaw_rad = df["yaw_rad"].to_numpy(dtype=float)
        att_deg = np.column_stack(
            [np.rad2deg(roll_rad), np.rad2deg(pitch_rad), np.rad2deg(yaw_rad)]
        )
        print("[IMU-PROC-PLOT] use attitude from roll_rad/pitch_rad/yaw_rad")
    elif all(c in df.columns for c in ("AngX", "AngY", "AngZ")):
        att_deg = df[["AngX", "AngY", "AngZ"]].to_numpy(dtype=float)
        print("[IMU-PROC-PLOT] use attitude from AngX/AngY/AngZ (deg)")
    else:
        att_deg = np.full((df.shape[0], 3), np.nan, dtype=float)
        print("[IMU-PROC-PLOT] WARNING: no columns for attitude, using NaN")

    fig, axes, layout = make_imu_3rows_canvas(layout)
    # 适合论文单列的比例：比之前略扁一些
    fig.set_size_inches(5.0, 4.0)
    lw = layout.lw()

    # Row 1: Acceleration —— 带图例，无 y label
    ax = axes[0]
    lines_acc = plot_xyz_lines(ax, t_plot, acc_body, linewidth=lw)
    ax.set_title("Acceleration (m/s$^2$)", fontsize=layout.title_fs())
    add_xyz_legend(ax, lines_acc, layout)

    # Row 2: Angular rate —— 无图例，无 y label
    ax = axes[1]
    plot_xyz_lines(ax, t_plot, gyro_body, linewidth=lw)
    ax.set_title("Angular rate (rad/s)", fontsize=layout.title_fs())

    # Row 3: Attitude —— 无图例，只有 x 轴标签
    ax = axes[2]
    plot_xyz_lines(ax, t_plot, att_deg, linewidth=lw)
    ax.set_title("Attitude (deg)", fontsize=layout.title_fs())
    ax.set_xlabel(xlab, fontsize=layout.label_fs())

    # 不再设置任何 ax.set_ylabel(...)
    finalize_imu_axes(axes, y_pad_frac=layout.y_pad_frac)

    fig.savefig(out_png)
    plt.close(fig)
    return out_png