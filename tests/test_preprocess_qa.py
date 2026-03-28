"""
模块名称：预处理 QA 回归测试

模块职责：
验证 `uwnav_dynamics.preprocess.qa` 提供的 hard / soft 检查能够正确识别
时间轴问题、窗口可构造性问题与稀疏监督覆盖率 warning，
并生成统一的 PASS / FAIL 摘要输出。

主要功能：
1. 验证合法 base table 的 QA 结果为 PASS，且包含覆盖率与统计摘要。
2. 验证时间列非严格递增时会 fail-fast。
3. 验证行数不足时会 fail-fast。
4. 验证 DVL 覆盖率为 0 时会产生 warning 但不改变 hard pass/fail 语义。

数据流：
synthetic train_base DataFrame
    ↓
run_train_base_qa()
    ↓
render_train_base_qa() / assert_train_base_qa_pass()

依赖模块：
- numpy
- pandas
- pytest
- uwnav_dynamics.preprocess.qa

备注：
- 本测试只覆盖 QA 模块本身，不涉及训练或评估阶段。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uwnav_dynamics.preprocess.qa import (
    assert_train_base_qa_pass,
    render_train_base_qa,
    run_train_base_qa,
)


IMU_COLS = [
    "AccX_body_mps2",
    "AccY_body_mps2",
    "AccZ_body_mps2",
    "GyroX_body_rad_s",
    "GyroY_body_rad_s",
    "GyroZ_body_rad_s",
]


def _make_base_df(n: int = 16) -> pd.DataFrame:
    t = np.arange(n, dtype=float) * 0.01
    df = pd.DataFrame(
        {
            "t_s": t,
            "AccX_body_mps2": np.sin(t),
            "AccY_body_mps2": np.cos(t),
            "AccZ_body_mps2": 0.5 * np.sin(2.0 * t),
            "GyroX_body_rad_s": 0.1 * np.cos(t),
            "GyroY_body_rad_s": 0.1 * np.sin(t),
            "GyroZ_body_rad_s": 0.05 * np.cos(2.0 * t),
            "dvl_mask": (np.arange(n) % 4 == 0).astype(int),
            "power_mask": (np.arange(n) % 5 == 0).astype(int),
        }
    )
    return df


def test_run_train_base_qa_passes_and_collects_summary():
    df = _make_base_df()
    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col="t_s",
        dense_target_cols=IMU_COLS,
        hist_len=4,
        pred_len=2,
        key_stat_cols=IMU_COLS,
    )

    text = render_train_base_qa(report)
    assert report.passed
    assert report.hard_error_count == 0
    assert "[QA][base_csv][SUMMARY] status=PASS" in text
    assert "[QA][base_csv][COVERAGE] name=dvl_mask" in text
    assert "[QA][base_csv][STAT] col=AccX_body_mps2" in text


def test_run_train_base_qa_rejects_non_increasing_time():
    df = _make_base_df()
    df.loc[5, "t_s"] = df.loc[4, "t_s"]

    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col="t_s",
        dense_target_cols=IMU_COLS,
        hist_len=4,
        pred_len=2,
        key_stat_cols=IMU_COLS,
    )

    assert not report.passed
    with pytest.raises(ValueError, match="strictly increasing"):
        assert_train_base_qa_pass(report)


def test_run_train_base_qa_rejects_insufficient_rows_for_windows():
    df = _make_base_df(n=5)

    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col="t_s",
        dense_target_cols=IMU_COLS,
        hist_len=4,
        pred_len=2,
        key_stat_cols=IMU_COLS,
    )

    assert not report.passed
    with pytest.raises(ValueError, match="rows not enough for sliding window"):
        assert_train_base_qa_pass(report)


def test_run_train_base_qa_warns_on_zero_dvl_coverage():
    df = _make_base_df()
    df["dvl_mask"] = 0

    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col="t_s",
        dense_target_cols=IMU_COLS,
        hist_len=4,
        pred_len=2,
        key_stat_cols=IMU_COLS,
    )

    text = render_train_base_qa(report)
    assert report.passed
    assert report.warning_count >= 1
    assert "zero_dvl_mask_coverage" in text
    assert "ratio=0.000000" in text


def test_run_train_base_qa_rejects_nonfinite_required_columns():
    df = _make_base_df()
    df.loc[3, "P0_W"] = np.nan

    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col="t_s",
        dense_target_cols=IMU_COLS,
        hist_len=4,
        pred_len=2,
        key_stat_cols=IMU_COLS,
        required_finite_cols=("P0_W",),
    )

    assert not report.passed
    with pytest.raises(ValueError, match="non-finite required cols"):
        assert_train_base_qa_pass(report)
