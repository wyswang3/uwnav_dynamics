"""
模块名称：dense target 数据集构建回归测试

模块职责：
验证 dataset build 在面对 dense target 数据质量问题时会及时 fail-fast，
并确认修复后的 IMU dense 对齐结果能够稳定进入 `labels.npz`，
而不会在上游生成 NaN 后继续扩散到训练标签窗口。

主要功能：
1. 构造含单个 dense target NaN 的最小 `train_base.csv`，验证 build_dataset 明确报错。
2. 构造不规则 IMU 时间轴的最小端到端样例，验证 `labels.npz` 中 dense target 全 finite。

数据流：
synthetic IMU proc / PWM / train_base.csv
    ↓
build_training_table_imu_main()
    ↓
build_dataset_from_config()
    ↓
labels.npz finite / fail-fast assertions

依赖模块：
- numpy
- pandas
- pytest
- uwnav_dynamics.preprocess.align.aligner
- uwnav_dynamics.preprocess.build_dataset
- uwnav_dynamics.preprocess.sliding_window

备注：
- 本测试只覆盖上游对齐与 dataset build 的质量契约；
- 不修改训练侧 dense supervision 语义。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uwnav_dynamics.preprocess.align.aligner import AlignConfig, build_training_table_imu_main
from uwnav_dynamics.preprocess.build_dataset import (
    DatasetConfig,
    DatasetOutputConfig,
    build_dataset_from_config,
)
from uwnav_dynamics.preprocess.sliding_window import SlidingWindowConfig


def _make_dataset_cfg(
    *,
    base_csv: Path,
    out_dir: Path,
    input_cols: list[str],
    target_cols: list[str],
) -> DatasetConfig:
    return DatasetConfig(
        name="dense_target_smoke",
        base_csv=base_csv,
        time_col="t_s",
        sliding_cfg=SlidingWindowConfig(
            input_cols=input_cols,
            target_cols=target_cols,
            hist_len=4,
            pred_len=2,
            stride=1,
            valid_mask_col=None,
            min_valid_ratio=1.0,
            drop_incomplete=True,
        ),
        output=DatasetOutputConfig(dir=out_dir, normalize="none"),
    )


def _write_train_base_csv_with_dense_nan(path: Path) -> None:
    t = np.arange(12, dtype=float) * 0.01
    df = pd.DataFrame(
        {
            "t_s": t,
            "ch1_cmd": np.full_like(t, 7.5),
            "AccX_body_mps2": np.sin(t),
            "AccY_body_mps2": np.cos(t),
            "AccZ_body_mps2": 0.5 * np.sin(2.0 * t),
            "GyroX_body_rad_s": 0.1 * np.cos(t),
            "GyroY_body_rad_s": 0.1 * np.sin(t),
            "GyroZ_body_rad_s": 0.05 * np.cos(2.0 * t),
        }
    )
    df.loc[3, "AccX_body_mps2"] = np.nan
    df.to_csv(path, index=False)


def _write_irregular_imu_proc_csv(path: Path) -> np.ndarray:
    n = 40
    dt = np.full(n - 1, 0.01, dtype=float)
    dt[::7] += 0.0010
    dt[3::9] -= 0.0007
    t = np.concatenate([[0.0], np.cumsum(dt)])
    df = pd.DataFrame(
        {
            "t_s": t,
            "AccX_body_mps2": np.sin(t),
            "AccY_body_mps2": np.cos(t),
            "AccZ_body_mps2": 0.25 * np.sin(2.0 * t),
            "GyroX_body_rad_s": 0.1 * np.cos(t),
            "GyroY_body_rad_s": 0.1 * np.sin(t),
            "GyroZ_body_rad_s": 0.05 * np.cos(2.0 * t),
        }
    )
    df.to_csv(path, index=False)
    return t


def _write_pwm_csv(path: Path, t_end: float) -> None:
    t = np.arange(0.0, t_end + 0.02, 0.01, dtype=float)
    data = {"t_s": t}
    for i in range(1, 9):
        data[f"ch{i}_cmd"] = np.full_like(t, 7.5)
    pd.DataFrame(data).to_csv(path, index=False)


def test_build_dataset_rejects_nonfinite_dense_targets_in_base_csv(tmp_path):
    base_csv = tmp_path / "train_base_bad.csv"
    out_dir = tmp_path / "processed_bad"
    _write_train_base_csv_with_dense_nan(base_csv)

    imu_cols = [
        "AccX_body_mps2",
        "AccY_body_mps2",
        "AccZ_body_mps2",
        "GyroX_body_rad_s",
        "GyroY_body_rad_s",
        "GyroZ_body_rad_s",
    ]
    cfg = _make_dataset_cfg(
        base_csv=base_csv,
        out_dir=out_dir,
        input_cols=["ch1_cmd", *imu_cols],
        target_cols=imu_cols,
    )

    with pytest.raises(ValueError, match="non-finite target cols in base_csv"):
        build_dataset_from_config(cfg)

    assert not (out_dir / "labels.npz").exists()
    assert not (out_dir / "features.npz").exists()


def test_build_dataset_labels_dense_targets_are_finite_after_align(tmp_path):
    imu_csv = tmp_path / "imu_irregular_proc.csv"
    pwm_csv = tmp_path / "pwm.csv"
    base_csv = tmp_path / "train_base_good.csv"
    out_dir = tmp_path / "processed_good"

    t_imu = _write_irregular_imu_proc_csv(imu_csv)
    _write_pwm_csv(pwm_csv, float(t_imu[-1]))

    df_base = build_training_table_imu_main(
        imu_proc_csv=imu_csv,
        pwm_csv=pwm_csv,
        dvl_proc_csv=None,
        power_csv=None,
        cfg=AlignConfig(
            dt_main_s=0.01,
            t_margin_s=0.0,
        ),
    )
    df_base.to_csv(base_csv, index=False)

    imu_cols = [
        "AccX_body_mps2",
        "AccY_body_mps2",
        "AccZ_body_mps2",
        "GyroX_body_rad_s",
        "GyroY_body_rad_s",
        "GyroZ_body_rad_s",
    ]
    cfg = _make_dataset_cfg(
        base_csv=base_csv,
        out_dir=out_dir,
        input_cols=["ch1_cmd", *imu_cols],
        target_cols=imu_cols,
    )

    build_dataset_from_config(cfg)

    labels = np.load(out_dir / "labels.npz", allow_pickle=True)
    y = labels["Y"]
    assert np.isfinite(y).all()
