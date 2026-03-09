"""
模块名称：科研绘图风格 smoke test

模块职责：
验证传感器图与 rollout 图已经遵循新的科研绘图规范，
重点检查无标题、共享 x 轴标签最小化与 legend 不重复等契约。

主要功能：
1. 通过 monkeypatch `Figure.savefig` 捕获 figure 对象进行结构断言。
2. 验证 IMU / DVL / Power 图默认无标题。
3. 验证共享 x 轴时仅底部保留 xlabel，单变量子图默认无 legend。

数据流：
synthetic sensor frame / processed csv
    ↓
viz.plots.* / viz.eval.plot_rollout_samples
    ↓
matplotlib Figure
    ↓
style contract assertions

依赖模块：
- matplotlib
- numpy
- pandas
- uwnav_dynamics.viz.*
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from uwnav_dynamics.io.readers.dvl_reader import DvlFrame
from uwnav_dynamics.io.readers.imu_reader import ImuFrame
from uwnav_dynamics.io.readers.power_reader import PowerFrame
from uwnav_dynamics.viz.eval.plot_rollout_samples import build_rollout_sample_figure
from uwnav_dynamics.viz.plots.dvl_plots import save_dvl_proc_figures
from uwnav_dynamics.viz.plots.imu_plot import save_imu_raw_9axis
from uwnav_dynamics.viz.plots.power_plots import save_power_currents_8motors


def _capture_savefig(monkeypatch):
    captured: list[tuple[Figure, Path]] = []

    def fake_savefig(self, fname, *args, **kwargs):
        path = Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        captured.append((self, path))

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    return captured


def _make_imu_frame(tmp_path: Path) -> ImuFrame:
    n = 32
    t = np.linspace(0.0, 3.1, n)
    dt = np.diff(t)
    df = pd.DataFrame({"EstS": t})
    return ImuFrame(
        path=tmp_path / "imu.csv",
        kind="test",
        df=df,
        time_col="EstS",
        t_s=t,
        t_rel_s=t - t[0],
        dt_s=dt,
        acc_g=np.column_stack([np.sin(t), np.cos(t), np.sin(0.5 * t)]),
        gyro_deg_s=np.column_stack([2.0 * np.sin(t), 1.5 * np.cos(t), np.sin(0.3 * t)]),
        ang_deg=np.column_stack([10.0 * np.sin(t), 5.0 * np.cos(t), 3.0 * np.sin(0.2 * t)]),
        acc_mps2=np.column_stack([np.sin(t), np.cos(t), np.sin(0.5 * t)]) * 9.78,
        gyro_rad_s=np.column_stack([2.0 * np.sin(t), 1.5 * np.cos(t), np.sin(0.3 * t)]) * np.pi / 180.0,
        ang_rad=np.column_stack([10.0 * np.sin(t), 5.0 * np.cos(t), 3.0 * np.sin(0.2 * t)]) * np.pi / 180.0,
        yaw_deg=None,
    )


def _make_power_frame(tmp_path: Path) -> PowerFrame:
    n = 40
    t = np.linspace(0.0, 4.0, n)
    currents = np.column_stack([0.5 + 0.1 * i + np.sin(t + 0.2 * i) for i in range(8)]).astype(np.float32)
    volts = np.full_like(currents, 12.0, dtype=np.float32)
    return PowerFrame(
        path=tmp_path / "power.csv",
        time_col="EstS",
        t_s=t,
        time_cols_raw={"EstS": t},
        est_ns=None,
        mono_ns=None,
        volt_motors=volts,
        curr_motors=currents,
        power_motors=volts * currents,
    )


def test_imu_raw_figure_has_no_titles_and_single_legend(tmp_path, monkeypatch):
    captured = _capture_savefig(monkeypatch)
    imu = _make_imu_frame(tmp_path)

    save_imu_raw_9axis(imu, out_root=tmp_path / "out")

    fig, _ = captured[0]
    axes = fig.axes
    fig.canvas.draw()
    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Time (s)"]
    assert [ax.get_ylabel() for ax in axes] == ["Acc (g)", "Gyro (deg/s)", "Att (deg)"]
    assert sum(ax.get_legend() is not None for ax in axes) == 1
    renderer = fig.canvas.get_renderer()
    ylabel_x0 = [ax.yaxis.label.get_window_extent(renderer).x0 for ax in axes]
    assert max(ylabel_x0) - min(ylabel_x0) < 1.0


def test_dvl_proc_figure_uses_bottom_xlabel_and_no_single_var_legends(tmp_path, monkeypatch):
    captured = _capture_savefig(monkeypatch)
    proc_csv = tmp_path / "dvl_proc.csv"
    t = np.linspace(0.0, 3.0, 24)
    pd.DataFrame(
        {
            "t_s": t,
            "VelBx_body_mps": np.sin(t),
            "VelBy_body_mps": np.cos(t),
            "VelBz_body_mps": 0.5 * np.sin(t),
            "VelU_enu_mps": 0.1 * np.cos(t),
            "Depth_m": 2.0 + 0.2 * np.sin(t),
        }
    ).to_csv(proc_csv, index=False)

    save_dvl_proc_figures(proc_csv, out_root=tmp_path / "out")

    fig, _ = captured[0]
    axes = fig.axes
    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Time (s)"]
    assert axes[0].get_legend() is not None
    assert axes[1].get_legend() is None
    assert axes[2].get_legend() is None


def test_power_figure_has_no_titles_or_legends_and_only_bottom_xlabels(tmp_path, monkeypatch):
    captured = _capture_savefig(monkeypatch)
    power = _make_power_frame(tmp_path)

    save_power_currents_8motors(power, out_root=tmp_path / "out")

    fig, _ = captured[0]
    axes = fig.axes
    assert all(ax.get_title() == "" for ax in axes)
    assert all(ax.get_legend() is None for ax in axes)
    assert [ax.get_xlabel() for ax in axes[:6]] == [""] * 6
    assert [ax.get_xlabel() for ax in axes[6:]] == ["Time (s)", "Time (s)"]


def test_rollout_sample_figure_has_no_titles_bottom_xlabel_and_single_legend():
    H = 12
    y_true = np.zeros((H, 9), dtype=float)
    y_hat = np.ones((H, 9), dtype=float) * 0.1
    fig, axes = build_rollout_sample_figure(y_hat=y_hat, y_true=y_true, dt_s=0.05)
    fig.canvas.draw()

    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Prediction horizon (s)"]
    assert len(fig.legends) == 1
    assert mpl.rcParams["font.family"][0] == "Times New Roman"
    assert axes[2].xaxis.label.get_fontfamily()[0] == "Times New Roman"
    assert fig.get_facecolor() == to_rgba("#FFFFFF")
    assert all(ax.get_facecolor() == to_rgba("#FFFFFF") for ax in axes)
    assert not any(line.get_visible() for ax in axes for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert to_rgba(axes[2].xaxis.label.get_color()) == to_rgba("#000000")
    assert to_rgba(axes[2].get_xticklabels()[0].get_color()) == to_rgba("#000000")
    legend = fig.legends[0]
    assert legend.get_frame().get_alpha() == 0.0
    assert legend.get_texts()[0].get_fontfamily()[0] == "Times New Roman"
    assert to_rgba(legend.get_texts()[0].get_color()) == to_rgba("#000000")
