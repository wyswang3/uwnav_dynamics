"""
模块名称：论文用状态代理量预处理图

模块职责：
为 `docs/math/` 理论文档生成可复现的论文风格图像，
展示原始传感器统计量与因果 KF 状态代理量之间的关系。

主要功能：
1. 从训练基础表读取原始 IMU / DVL 与 KF 代理状态列。
2. 生成 Acc / Gyro / Vel 三行对照图，说明噪声抑制与速度稠密化效果。
3. 将图像输出到论文文档可直接引用的路径。
4. 对单点/空序列跳过折线绘制，并输出 sidecar 记录。

数据流：
train_base_kf_v2.csv
    ↓
提取原始体坐标加速度、角速度、DVL 速度与 KF 代理状态
    ↓
统一科研绘图风格绘制三行对照图
    ↓
docs/math/figures/*.png

依赖模块：
- pandas
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.style.imu_style

备注：
- 本图服务于论文方法章节，因此默认使用紧凑、无标题、共享时间轴的学术风格。
- 速度行使用稀疏 DVL 观测点与 KF 稠密速度代理量叠加，强调“校正 + 传播”的状态构造语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from uwnav_dynamics.viz.style.imu_style import Imu3RowLayout, finalize_imu_axes, make_imu_3rows_canvas
from uwnav_dynamics.viz.style.sci_style import (
    add_axes_legend,
    align_ylabels,
    apply_axes_style,
    plot_or_record_series,
    setup_mpl,
    SparsePlotRecorder,
)


def _safe_norm3(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    arr = df[list(cols)].to_numpy(dtype=float)
    out = np.linalg.norm(arr, axis=1)
    out[~np.isfinite(arr).all(axis=1)] = np.nan
    return out


def _select_window(df: pd.DataFrame, t_start: float | None, t_end: float | None) -> pd.DataFrame:
    if t_start is None or t_end is None:
        dvl_hits = df.loc[df["HasDvlUpdate"].fillna(0).to_numpy(dtype=float) > 0.0, "t_s"]
        if len(dvl_hits) > 0:
            center = float(dvl_hits.iloc[0])
            t_start = center - 2.0 if t_start is None else t_start
            t_end = center + 8.0 if t_end is None else t_end
        else:
            head_t = float(df["t_s"].iloc[0])
            t_start = head_t if t_start is None else t_start
            t_end = head_t + 10.0 if t_end is None else t_end
    sel = df[(df["t_s"] >= float(t_start)) & (df["t_s"] <= float(t_end))].copy()
    if sel.empty:
        raise ValueError(f"selected window is empty: t_start={t_start}, t_end={t_end}")
    return sel


def save_proxy_state_comparison_figure(
    csv_path: str | Path,
    out_png: str | Path,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    use_rel_time: bool = True,
) -> Path:
    """
    生成论文方法章节用的原始观测与 KF 状态代理量对照图。

    图像三行分别展示：
    1. 原始与 KF 体坐标加速度模长
    2. 原始与 KF 体坐标角速度模长
    3. 稀疏 DVL 体速度模长与 KF 稠密速度代理量
    """
    setup_mpl()

    csv_path = Path(csv_path).expanduser().resolve()
    out_png = Path(out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    warnings_txt = out_png.with_suffix(".plot_warnings.txt")
    recorder = SparsePlotRecorder()

    df = pd.read_csv(csv_path)
    required = [
        "t_s",
        "AccX_body_mps2",
        "AccY_body_mps2",
        "AccZ_body_mps2",
        "GyroX_body_rad_s",
        "GyroY_body_rad_s",
        "GyroZ_body_rad_s",
        "VelBx_body_mps",
        "VelBy_body_mps",
        "VelBz_body_mps",
        "AccKfX_body_mps2",
        "AccKfY_body_mps2",
        "AccKfZ_body_mps2",
        "GyroKfX_body_rad_s",
        "GyroKfY_body_rad_s",
        "GyroKfZ_body_rad_s",
        "VelKfX_body_mps",
        "VelKfY_body_mps",
        "VelKfZ_body_mps",
        "HasDvlUpdate",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    win = _select_window(df, t_start=t_start, t_end=t_end)
    t = win["t_s"].to_numpy(dtype=float)
    t_plot = t - float(t[0]) if use_rel_time else t

    acc_raw = _safe_norm3(win, ("AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2"))
    acc_kf = _safe_norm3(win, ("AccKfX_body_mps2", "AccKfY_body_mps2", "AccKfZ_body_mps2"))
    gyro_raw = _safe_norm3(win, ("GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"))
    gyro_kf = _safe_norm3(win, ("GyroKfX_body_rad_s", "GyroKfY_body_rad_s", "GyroKfZ_body_rad_s"))
    vel_dvl = _safe_norm3(win, ("VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"))
    vel_kf = _safe_norm3(win, ("VelKfX_body_mps", "VelKfY_body_mps", "VelKfZ_body_mps"))
    has_dvl = win["HasDvlUpdate"].fillna(0.0).to_numpy(dtype=float) > 0.0

    layout = Imu3RowLayout()
    fig, axes, layout = make_imu_3rows_canvas(layout)

    lw_raw = 1.05
    lw_kf = 1.55

    plot_or_record_series(
        axes[0],
        t_plot,
        acc_raw,
        panel="Proxy state excerpt / acc",
        series="raw imu norm",
        color="#2F3B52",
        recorder=recorder,
        skip_message="Acc raw skipped: fewer than 2 samples",
        linewidth=lw_raw,
        alpha=0.92,
        label="Raw IMU norm",
    )
    plot_or_record_series(
        axes[0],
        t_plot,
        acc_kf,
        panel="Proxy state excerpt / acc",
        series="kf proxy norm",
        color="#2C7FB8",
        recorder=recorder,
        skip_message="Acc KF skipped: fewer than 2 samples",
        linewidth=lw_kf,
        alpha=0.98,
        linestyle="--",
        label="KF proxy norm",
    )
    axes[0].set_ylabel(r"$||a_b||$ (m/s$^2$)")

    plot_or_record_series(
        axes[1],
        t_plot,
        gyro_raw,
        panel="Proxy state excerpt / gyro",
        series="raw imu norm",
        color="#2F3B52",
        recorder=recorder,
        skip_message="Gyro raw skipped: fewer than 2 samples",
        linewidth=lw_raw,
        alpha=0.92,
        label="Raw IMU norm",
    )
    plot_or_record_series(
        axes[1],
        t_plot,
        gyro_kf,
        panel="Proxy state excerpt / gyro",
        series="kf proxy norm",
        color="#E76F51",
        recorder=recorder,
        skip_message="Gyro KF skipped: fewer than 2 samples",
        linewidth=lw_kf,
        alpha=0.98,
        linestyle="--",
        label="KF proxy norm",
    )
    axes[1].set_ylabel(r"$||\omega_b||$ (rad/s)")

    plot_or_record_series(
        axes[2],
        t_plot,
        vel_kf,
        panel="Proxy state excerpt / velocity",
        series="kf dense velocity",
        color="#1B9E77",
        recorder=recorder,
        skip_message="KF dense velocity skipped: fewer than 2 samples",
        linewidth=lw_kf,
        alpha=0.98,
        label="KF dense velocity",
    )
    axes[2].scatter(
        t_plot[has_dvl],
        vel_dvl[has_dvl],
        s=11.0,
        color="#D4A72C",
        alpha=0.88,
        edgecolors="none",
        label="DVL updates",
        zorder=5,
    )
    axes[2].set_ylabel(r"$||v_b||$ (m/s)")

    for ax in axes:
        apply_axes_style(ax, grid=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    add_axes_legend(axes[0], loc="upper right", ncol=1)
    add_axes_legend(axes[2], loc="upper right", ncol=1)
    finalize_imu_axes(axes, y_pad_frac=layout.y_pad_frac)

    fig.savefig(out_png)
    plt.close(fig)
    recorder.write_text(warnings_txt)
    return out_png


def save_proxy_state_fullrun_figure(
    csv_path: str | Path,
    out_png: str | Path,
    *,
    use_rel_time: bool = True,
) -> Path:
    """
    生成覆盖全程的预处理评估图。

    图像四行分别展示：
    1. 原始与 KF 加速度模长；
    2. 原始与 KF 角速度模长；
    3. DVL 稀疏速度与 KF 稠密速度；
    4. DVL freshness 与速度方差代理量。
    """
    setup_mpl()

    csv_path = Path(csv_path).expanduser().resolve()
    out_png = Path(out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    warnings_txt = out_png.with_suffix(".plot_warnings.txt")
    recorder = SparsePlotRecorder()

    df = pd.read_csv(csv_path)
    required = [
        "t_s",
        "AccX_body_mps2",
        "AccY_body_mps2",
        "AccZ_body_mps2",
        "GyroX_body_rad_s",
        "GyroY_body_rad_s",
        "GyroZ_body_rad_s",
        "VelBx_body_mps",
        "VelBy_body_mps",
        "VelBz_body_mps",
        "AccKfX_body_mps2",
        "AccKfY_body_mps2",
        "AccKfZ_body_mps2",
        "GyroKfX_body_rad_s",
        "GyroKfY_body_rad_s",
        "GyroKfZ_body_rad_s",
        "VelKfX_body_mps",
        "VelKfY_body_mps",
        "VelKfZ_body_mps",
        "DtSinceDvl_s",
        "VelKfVarX_body_mps2",
        "VelKfVarY_body_mps2",
        "VelKfVarZ_body_mps2",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    t = df["t_s"].to_numpy(dtype=float)
    t_plot = t - float(t[0]) if use_rel_time else t

    acc_raw = _safe_norm3(df, ("AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2"))
    acc_kf = _safe_norm3(df, ("AccKfX_body_mps2", "AccKfY_body_mps2", "AccKfZ_body_mps2"))
    gyro_raw = _safe_norm3(df, ("GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"))
    gyro_kf = _safe_norm3(df, ("GyroKfX_body_rad_s", "GyroKfY_body_rad_s", "GyroKfZ_body_rad_s"))
    vel_dvl = _safe_norm3(df, ("VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"))
    vel_kf = _safe_norm3(df, ("VelKfX_body_mps", "VelKfY_body_mps", "VelKfZ_body_mps"))
    vel_var = _safe_norm3(df, ("VelKfVarX_body_mps2", "VelKfVarY_body_mps2", "VelKfVarZ_body_mps2"))

    has_meas = (
        df["HasDvlMeasurement"].fillna(0.0).to_numpy(dtype=float) > 0.0
        if "HasDvlMeasurement" in df.columns
        else np.isfinite(df[["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]].to_numpy(dtype=float)).all(axis=1)
    )
    has_update = (
        df["HasDvlUpdate"].fillna(0.0).to_numpy(dtype=float) > 0.0
        if "HasDvlUpdate" in df.columns
        else has_meas
    )
    dt_since_dvl = df["DtSinceDvl_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(6.8, 5.5),
        dpi=150,
        gridspec_kw={"hspace": 0.18},
    )

    plot_or_record_series(
        axes[0],
        t_plot,
        acc_raw,
        panel="Proxy state fullrun / acc",
        series="raw imu norm",
        color="#2F3B52",
        recorder=recorder,
        skip_message="Acc raw skipped: fewer than 2 samples",
        linewidth=0.95,
        alpha=0.80,
        label="Raw IMU norm",
    )
    plot_or_record_series(
        axes[0],
        t_plot,
        acc_kf,
        panel="Proxy state fullrun / acc",
        series="kf proxy norm",
        color="#2C7FB8",
        recorder=recorder,
        skip_message="Acc KF skipped: fewer than 2 samples",
        linewidth=1.25,
        alpha=0.98,
        linestyle="--",
        label="KF proxy norm",
    )
    axes[0].set_ylabel(r"$||a_b||$")

    plot_or_record_series(
        axes[1],
        t_plot,
        gyro_raw,
        panel="Proxy state fullrun / gyro",
        series="raw imu norm",
        color="#2F3B52",
        recorder=recorder,
        skip_message="Gyro raw skipped: fewer than 2 samples",
        linewidth=0.95,
        alpha=0.80,
        label="Raw IMU norm",
    )
    plot_or_record_series(
        axes[1],
        t_plot,
        gyro_kf,
        panel="Proxy state fullrun / gyro",
        series="kf proxy norm",
        color="#E76F51",
        recorder=recorder,
        skip_message="Gyro KF skipped: fewer than 2 samples",
        linewidth=1.25,
        alpha=0.98,
        linestyle="--",
        label="KF proxy norm",
    )
    axes[1].set_ylabel(r"$||\omega_b||$")

    plot_or_record_series(
        axes[2],
        t_plot,
        vel_kf,
        panel="Proxy state fullrun / velocity",
        series="kf dense velocity",
        color="#1B9E77",
        recorder=recorder,
        skip_message="KF dense velocity skipped: fewer than 2 samples",
        linewidth=1.25,
        alpha=0.98,
        label="KF dense velocity",
    )
    axes[2].scatter(t_plot[has_meas], vel_dvl[has_meas], s=5.0, color="#D4A72C", alpha=0.55, edgecolors="none", label="DVL measurement")
    axes[2].scatter(t_plot[has_update], vel_dvl[has_update], s=7.0, color="#C8553D", alpha=0.80, edgecolors="none", label="Accepted update")
    axes[2].set_ylabel(r"$||v_b||$")

    plot_or_record_series(
        axes[3],
        t_plot,
        dt_since_dvl,
        panel="Proxy state fullrun / quality",
        series="dt since dvl",
        color="#5C6BC0",
        recorder=recorder,
        skip_message="DVL freshness skipped: fewer than 2 samples",
        linewidth=1.05,
        alpha=0.96,
        label=r"$\Delta t$ since DVL",
    )
    plot_or_record_series(
        axes[3],
        t_plot,
        vel_var,
        panel="Proxy state fullrun / quality",
        series="velocity variance proxy",
        color="#7A8CA3",
        recorder=recorder,
        skip_message="Velocity variance skipped: fewer than 2 samples",
        linewidth=1.00,
        alpha=0.90,
        linestyle="--",
        label="Velocity variance proxy",
    )
    axes[3].set_ylabel("Quality")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        apply_axes_style(ax, grid=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))

    add_axes_legend(axes[0], loc="upper right", ncol=1)
    add_axes_legend(axes[2], loc="upper right", ncol=1)
    add_axes_legend(axes[3], loc="upper right", ncol=1)
    align_ylabels(axes)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    fig.savefig(out_png)
    plt.close(fig)
    recorder.write_text(warnings_txt)
    return out_png
