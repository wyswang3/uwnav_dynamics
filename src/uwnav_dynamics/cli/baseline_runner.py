"""
模块名称：baseline runner CLI

模块职责：
提供 trivial 与 classical baseline 的正式命令行入口，
把 baseline fit、评估与汇总统一编排为可复现实验流程。

主要功能：
1. 解析 train yaml、baseline 列表与运行目录参数。
2. 调用 `uwnav_dynamics.baselines.runner.run_baseline_suite()` 执行整套流程。
3. 输出 `manifest.yaml`、`summary.csv` 与可选 compare 图目录提示。

数据流：
train yaml + CLI args
    ↓
run_baseline_suite()
    ↓
baseline run dirs / eval artifact / summary.csv

依赖模块：
- uwnav_dynamics.baselines.runner

备注：
- baseline runner 默认输出到 `run.out_dir/baseline_runs/`。
- 各 baseline 的评估 artifact 与神经网络评估保持同一目录契约。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uwnav_dynamics.baselines.runner import run_baseline_suite


def main() -> int:
    """baseline runner 命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.baseline_runner")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml")
    ap.add_argument(
        "--baseline",
        type=str,
        nargs="+",
        default=["trivial_last", "classical_ridge"],
        choices=["trivial_last", "classical_ridge"],
        help="baseline kinds to run",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--work_dir", type=str, default=None, help="override baseline work dir")
    ap.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-4, 1.0e-2, 1.0],
        help="candidate ridge alphas for classical_ridge",
    )
    ap.add_argument("--save_samples", type=int, default=256)
    ap.add_argument("--compare", action="store_true", help="write compare plots across baselines")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    args = ap.parse_args()

    summary_path = run_baseline_suite(
        train_yaml=Path(args.yaml),
        baselines=tuple(args.baseline),
        split=str(args.split),
        work_dir=Path(args.work_dir) if args.work_dir is not None else None,
        alpha_grid=tuple(float(v) for v in args.alpha_grid),
        save_samples=int(args.save_samples),
        compare=bool(args.compare),
        dt=float(args.dt),
        x_axis=str(args.x_axis),
        plot_fmt=str(args.plot_fmt),
    )
    print(f"[BASELINE] summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
