"""
模块名称：KF / ESKF 融合预处理测试

模块职责：
验证新的融合预处理模块能够把对齐基础表转换为训练可用的低噪声状态代理量表，
并在落盘前通过有限值审查，避免 NaN 进入后续训练主链路。

主要功能：
1. 验证融合输出包含训练所需的 KF 列、姿态上下文列和诊断列。
2. 验证 power 初始缺测会被修复为 finite。
3. 验证 DVL 观测可压低明显的加速度尖峰。

数据流：
synthetic train_base DataFrame + imu_proc DataFrame
    ↓
build_fused_train_base_from_frames()
    ↓
finite-column assertions / spike suppression assertion

依赖模块：
- numpy
- pandas
- uwnav_dynamics.preprocess.fusion.kf_eskf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uwnav_dynamics.preprocess.fusion.kf_eskf import (
    KfEskfFusionConfig,
    build_fused_train_base_from_frames,
)


def _make_base_df() -> pd.DataFrame:
    n = 12
    t = np.arange(n, dtype=float) * 0.01
    acc = np.zeros((n, 3), dtype=float)
    gyro = np.zeros((n, 3), dtype=float)
    acc[5, 0] = 8.0  # 人工高频尖峰
    dvl = np.zeros((n, 3), dtype=float)
    dvl_mask = np.ones(n, dtype=int)

    df = pd.DataFrame(
        {
            "t_s": t,
            "AccX_body_mps2": acc[:, 0],
            "AccY_body_mps2": acc[:, 1],
            "AccZ_body_mps2": acc[:, 2],
            "GyroX_body_rad_s": gyro[:, 0],
            "GyroY_body_rad_s": gyro[:, 1],
            "GyroZ_body_rad_s": gyro[:, 2],
            "VelBx_body_mps": dvl[:, 0],
            "VelBy_body_mps": dvl[:, 1],
            "VelBz_body_mps": dvl[:, 2],
            "dvl_mask": dvl_mask,
            "has_dvl": dvl_mask,
            "power_mask": np.asarray([0, 0] + [1] * (n - 2), dtype=int),
        }
    )
    for i in range(1, 9):
        df[f"ch{i}_cmd"] = 7.5
    for i in range(8):
        vals = np.linspace(10.0 + i, 12.0 + i, n)
        vals[:2] = np.nan
        df[f"P{i}_W"] = vals
    return df


def _make_imu_df() -> pd.DataFrame:
    n = 12
    t = np.arange(n, dtype=float) * 0.01
    return pd.DataFrame(
        {
            "t_s": t,
            "roll_rad": np.zeros(n, dtype=float),
            "pitch_rad": np.zeros(n, dtype=float),
            "yaw_rad": np.zeros(n, dtype=float),
        }
    )


def test_build_fused_train_base_from_frames_outputs_finite_training_columns() -> None:
    out = build_fused_train_base_from_frames(
        _make_base_df(),
        _make_imu_df(),
        cfg=KfEskfFusionConfig(mode="eskf"),
    )

    required = [
        "AccKfX_body_mps2",
        "GyroKfX_body_rad_s",
        "VelKfX_body_mps",
        "RollKf_rad",
        "PitchKf_rad",
        "YawKf_rad",
        "SinYawKf",
        "CosYawKf",
        "DtSinceDvl_s",
    ] + [f"P{i}_W" for i in range(8)]

    for col in required:
        assert col in out.columns
        assert np.isfinite(out[col].to_numpy(dtype=float)).all(), col


def test_fusion_uses_dvl_to_suppress_acceleration_spike() -> None:
    base_df = _make_base_df()
    out = build_fused_train_base_from_frames(
        base_df,
        _make_imu_df(),
        cfg=KfEskfFusionConfig(
            mode="eskf",
            accel_lpf_alpha=0.35,
            vel_accel_blend=0.90,
            bias_correction_gain_acc=0.05,
        ),
    )

    raw_peak = abs(float(base_df.loc[5, "AccX_body_mps2"]))
    fused_peak = abs(float(out.loc[5, "AccKfX_body_mps2"]))
    assert fused_peak < raw_peak
