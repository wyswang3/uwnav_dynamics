"""
模块名称：预处理 QA CLI 回归测试

模块职责：
验证 `uwnav_dynamics.preprocess.cli_qa` 作为现有 QA 模块的薄命令行外壳，
能够正确读取 dataset yaml、可选覆盖 base_csv 路径，并用退出码表达 hard checks 结果。

主要功能：
1. 验证合法 base_csv 经 CLI 审查后返回 0，并打印 PASS 摘要。
2. 验证 `--base_csv` 覆盖路径在 hard check 失败时返回非 0，并打印 FAIL 摘要。

数据流：
dataset yaml + base_csv
    ↓
cli_qa.main()
    ↓
load_dataset_config() + run_train_base_qa()
    ↓
stdout summary + return code

依赖模块：
- numpy
- pandas
- yaml
- uwnav_dynamics.preprocess.cli_qa

备注：
- 本测试只验证 CLI 外壳，不修改 QA 核心规则。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.preprocess.cli_qa import main


IMU_COLS = [
    "AccX_body_mps2",
    "AccY_body_mps2",
    "AccZ_body_mps2",
    "GyroX_body_rad_s",
    "GyroY_body_rad_s",
    "GyroZ_body_rad_s",
]


def _write_base_csv(path: Path, *, with_nan: bool) -> None:
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
    if with_nan:
        df.loc[2, "AccX_body_mps2"] = np.nan
    df.to_csv(path, index=False)


def _write_dataset_yaml(path: Path, *, base_csv: Path, out_dir: Path) -> None:
    payload = {
        "dataset": {
            "name": "qa_cli_smoke",
            "base_table": {
                "csv": str(base_csv),
                "time_col": "t_s",
            },
            "sliding_window": {
                "input_cols": ["ch1_cmd", *IMU_COLS],
                "target_cols": list(IMU_COLS),
                "hist_len": 4,
                "pred_len": 2,
                "stride": 1,
                "valid_mask_col": None,
                "min_valid_ratio": 1.0,
                "drop_incomplete": True,
            },
            "output": {
                "dir": str(out_dir),
                "normalize": "none",
            },
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_preprocess_qa_cli_returns_zero_on_pass(tmp_path, capsys):
    base_csv = tmp_path / "train_base_ok.csv"
    dataset_yaml = tmp_path / "dataset.yaml"
    out_dir = tmp_path / "processed"

    _write_base_csv(base_csv, with_nan=False)
    _write_dataset_yaml(dataset_yaml, base_csv=base_csv, out_dir=out_dir)

    ret = main(["--dataset-yaml", str(dataset_yaml)])
    out = capsys.readouterr().out

    assert ret == 0
    assert "[QA][base_csv][SUMMARY] status=PASS" in out


def test_preprocess_qa_cli_returns_nonzero_on_hard_fail_with_override(tmp_path, capsys):
    base_csv_ok = tmp_path / "train_base_ok.csv"
    base_csv_bad = tmp_path / "train_base_bad.csv"
    dataset_yaml = tmp_path / "dataset.yaml"
    out_dir = tmp_path / "processed"

    _write_base_csv(base_csv_ok, with_nan=False)
    _write_base_csv(base_csv_bad, with_nan=True)
    _write_dataset_yaml(dataset_yaml, base_csv=base_csv_ok, out_dir=out_dir)

    ret = main(["--dataset-yaml", str(dataset_yaml), "--base_csv", str(base_csv_bad)])
    out = capsys.readouterr().out

    assert ret == 1
    assert "[QA][base_csv][SUMMARY] status=FAIL" in out
    assert "non-finite target cols in base_csv" in out
