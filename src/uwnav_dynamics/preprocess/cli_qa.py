"""
模块名称：train_base QA 命令行入口

模块职责：
为已有的 `train_base.csv` 提供一个独立的命令行审查入口，
复用现有 `preprocess.qa` 的 hard / soft checks 与摘要渲染逻辑，
避免必须通过 align 或 build_dataset 主流程才能查看 QA 结果。

主要功能：
1. 读取 dataset yaml，解析 time_col / target_cols / hist_len / pred_len。
2. 允许用显式 `--base_csv` 覆盖 yaml 中的 `base_table.csv`。
3. 复用 `_add_state_velocity_cols()`、`run_train_base_qa()` 与 summary render。
4. 根据 hard checks 结果返回 0 / 非 0 退出码。

数据流：
dataset yaml + base_csv
    ↓
load_dataset_config()
    ↓
_add_state_velocity_cols()
    ↓
run_train_base_qa()
    ↓
stdout summary + process exit code

依赖模块：
- pandas
- uwnav_dynamics.preprocess.build_dataset
- uwnav_dynamics.preprocess.qa

备注：
- 本模块不新增任何 QA 规则；
- 不做自动修复、改写或额外数据处理；
- 仅作为已有 QA 逻辑的薄 CLI 外壳。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from uwnav_dynamics.preprocess.build_dataset import (
    _add_state_velocity_cols,
    load_dataset_config,
)
from uwnav_dynamics.preprocess.qa import (
    assert_train_base_qa_pass,
    render_train_base_qa,
    run_train_base_qa,
)


def main(argv: Sequence[str] | None = None) -> int:
    """train_base QA 报告的命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.preprocess.cli_qa")
    ap.add_argument(
        "-y",
        "--dataset-yaml",
        type=str,
        required=True,
        help="dataset yaml used to resolve base_table / time_col / target_cols / window span",
    )
    ap.add_argument(
        "--base_csv",
        type=str,
        default=None,
        help="optional override for dataset.base_table.csv",
    )
    args = ap.parse_args(argv)

    cfg = load_dataset_config(Path(args.dataset_yaml))
    base_csv = Path(args.base_csv).expanduser().resolve() if args.base_csv else cfg.base_csv

    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {base_csv}")

    df = pd.read_csv(base_csv)
    if df.empty:
        raise RuntimeError(f"Base CSV is empty: {base_csv}")

    df = _add_state_velocity_cols(df)
    report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col=cfg.time_col,
        dense_target_cols=cfg.sliding_cfg.target_cols,
        hist_len=int(cfg.sliding_cfg.hist_len),
        pred_len=int(cfg.sliding_cfg.pred_len),
        key_stat_cols=cfg.sliding_cfg.target_cols,
    )
    print(render_train_base_qa(report))

    try:
        assert_train_base_qa_pass(report)
    except ValueError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
