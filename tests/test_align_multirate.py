from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uwnav_dynamics.preprocess.align.aligner import AlignConfig, build_training_table_imu_main


def _write_imu_csv(path: Path) -> None:
    """
    写一个最小 IMU 预处理结果文件（100 Hz）。
    """
    t = np.arange(0.0, 1.0, 0.01, dtype=float)  # 100 samples
    df = pd.DataFrame(
        {
            "t_s": t,
            "AccX_body_mps2": np.zeros_like(t),
            "AccY_body_mps2": np.zeros_like(t),
            "AccZ_body_mps2": np.zeros_like(t),
            "GyroX_body_rad_s": np.zeros_like(t),
            "GyroY_body_rad_s": np.zeros_like(t),
            "GyroZ_body_rad_s": np.zeros_like(t),
        }
    )
    df.to_csv(path, index=False)


def _write_pwm_csv(path: Path) -> None:
    """
    写一个最小 PWM 对齐文件（100 Hz）。
    """
    t = np.arange(0.0, 1.0, 0.01, dtype=float)
    data = {"t_s": t}
    for i in range(1, 9):
        data[f"ch{i}_cmd"] = np.full_like(t, 7.5)
    pd.DataFrame(data).to_csv(path, index=False)


def _write_dvl_csv(path: Path) -> None:
    """
    写一个最小 DVL 文件（10 Hz），仅在 0.0, 0.1, ... 0.9 有观测。
    """
    t = np.arange(0.0, 1.0, 0.1, dtype=float)
    df = pd.DataFrame(
        {
            "t_s": t,
            "VelBx_body_mps": np.ones_like(t) * 0.1,
            "VelBy_body_mps": np.ones_like(t) * 0.2,
            "VelBz_body_mps": np.ones_like(t) * -0.1,
            "Speed_body_mps": np.sqrt(0.1**2 + 0.2**2 + (-0.1) ** 2) * np.ones_like(t),
        }
    )
    df.to_csv(path, index=False)


def _write_power_csv(path: Path) -> None:
    """
    写一个最小 Power 文件（5 Hz），仅在 0.0, 0.2, ... 0.8 有观测。
    """
    t = np.arange(0.0, 1.0, 0.2, dtype=float)
    data = {"t_s": t}
    for i in range(8):
        data[f"P{i}_W"] = np.full_like(t, 10.0 + i)
    pd.DataFrame(data).to_csv(path, index=False)


def test_align_multirate_masks(tmp_path):
    """
    验证多频对齐后的 mask 行为：
      - DVL 10 Hz 对齐到 100 Hz 主轴时，仅对应采样点 dvl_mask=1；
      - Power 5 Hz 对齐到 100 Hz 主轴时，仅对应采样点 power_mask=1（max_dt=0）。
      - mask=0 的行，相关列应为 NaN。

    这里故意把 power_max_dt_s 设为 0：
      - 边界点（例如 0.6s）理论上应命中；
      - 非边界点（例如 0.59s / 0.61s）绝不能因为修复浮点误差而被误放宽。
    """
    imu_csv = tmp_path / "imu_proc.csv"
    pwm_csv = tmp_path / "pwm.csv"
    dvl_csv = tmp_path / "dvl_proc.csv"
    power_csv = tmp_path / "power.csv"

    _write_imu_csv(imu_csv)
    _write_pwm_csv(pwm_csv)
    _write_dvl_csv(dvl_csv)
    _write_power_csv(power_csv)

    cfg = AlignConfig(
        dt_main_s=0.01,
        t_margin_s=0.0,
        dvl_max_dt_s=0.003,   # 仅命中精确对应点（0.01 网格）
        power_max_dt_s=0.0,   # power 仅在采样点有效
    )

    df = build_training_table_imu_main(
        imu_proc_csv=imu_csv,
        pwm_csv=pwm_csv,
        dvl_proc_csv=dvl_csv,
        power_csv=power_csv,
        cfg=cfg,
    )

    dvl_mask = df["dvl_mask"].to_numpy(dtype=int)
    power_mask = df["power_mask"].to_numpy(dtype=int)

    dvl_hit = np.where(dvl_mask == 1)[0]
    power_hit = np.where(power_mask == 1)[0]

    # DVL: 0.0, 0.1, ..., 0.9 -> indices 0,10,...,90
    assert np.array_equal(dvl_hit, np.arange(0, 100, 10, dtype=int))
    # Power: 0.0, 0.2, ..., 0.8 -> indices 0,20,...,80
    assert np.array_equal(power_hit, np.arange(0, 100, 20, dtype=int))
    # 修复浮点边界后，理论重合点 0.60s 应命中，但相邻非边界点不能被误放宽。
    assert power_mask[60] == 1
    assert power_mask[59] == 0
    assert power_mask[61] == 0

    # 兼容列 has_dvl 与 dvl_mask 一致
    assert np.array_equal(df["has_dvl"].to_numpy(dtype=int), dvl_mask)

    # mask=0 时 DVL 监督列应为 NaN
    dvl_cols = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps", "Speed_body_mps"]
    dvl_non_hit = np.where(dvl_mask == 0)[0]
    assert np.isnan(df.loc[dvl_non_hit, dvl_cols].to_numpy(dtype=float)).all()

    # mask=0 时 Power 列应为 NaN
    power_cols = [f"P{i}_W" for i in range(8)]
    power_non_hit = np.where(power_mask == 0)[0]
    assert np.isnan(df.loc[power_non_hit, power_cols].to_numpy(dtype=float)).all()
