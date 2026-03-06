"""
模块名称：评估 CLI 编排入口

模块职责：
作为正式用户入口，负责将“数值评估”和“可选绘图”串联起来，
但不在本模块内重新实现评估数值逻辑或绘图逻辑。

主要功能：
1. 解析用户传入的 train yaml、checkpoint 与评估运行参数。
2. 先调用纯数值的 `uwnav_dynamics.eval.evaluate` 生成评估 artifact。
3. 若请求绘图，再调用 `uwnav_dynamics.viz.eval.*` 从 artifact 读盘出图。

数据流：
train yaml + ckpt/run_dir + CLI runtime args
    ↓
cli.utils 解析 ckpt / eval_dir / plots_dir
    ↓
eval.evaluate 写出 metrics.yaml / CSV / pred_samples.npz
    ↓
viz.eval.plot_horizon_metrics + viz.eval.plot_rollout_samples
    ↓
plots/*

依赖模块：
- uwnav_dynamics.cli.utils
- uwnav_dynamics.eval.evaluate
- uwnav_dynamics.viz.eval.plot_horizon_metrics
- uwnav_dynamics.viz.eval.plot_rollout_samples

备注：
- 若数值评估成功但绘图失败，本模块会返回非零码，同时保留已生成的数值 artifact。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import (
    pick_ckpt,
    resolve_eval_out_dir,
    resolve_eval_plots_dir,
    resolve_run_out_dir,
)


def _run_stage(stage_name: str, cmd: list[str]) -> int:
    print(f"[CLI][EVAL][{stage_name}] cmd:", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.eval")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml")
    ap.add_argument("--ckpt", type=str, default=None, help="ckpt path OR run_dir (contains best/last)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="override eval output dir")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--n_plot_samples", type=int, default=8)
    args = ap.parse_args()

    y = Path(args.yaml)
    out_dir, variant = resolve_run_out_dir(y)

    # 默认 ckpt：用 out_dir/variant 下的 best/last
    if args.ckpt is None:
        run_dir = out_dir / variant
        ckpt = pick_ckpt(run_dir)
    else:
        ckpt = pick_ckpt(Path(args.ckpt))

    eval_out_dir = resolve_eval_out_dir(
        y,
        split=args.split,
        out_dir_override=Path(args.out_dir) if args.out_dir is not None else None,
    )
    plots_dir = resolve_eval_plots_dir(eval_out_dir)

    cmd_eval = [
        sys.executable, "-m", "uwnav_dynamics.eval.evaluate",
        "--yaml", str(y),
        "--ckpt", str(ckpt),
        "--split", args.split,
        "--save_samples", "256",
    ]
    if args.device is not None:
        cmd_eval += ["--device", args.device]
    if args.batch_size is not None:
        cmd_eval += ["--batch_size", str(args.batch_size)]
    if args.out_dir is not None:
        cmd_eval += ["--out_dir", args.out_dir]

    ret = _run_stage("NUMERIC", cmd_eval)
    if ret != 0:
        return ret

    print(f"[CLI][EVAL] numeric artifacts saved under: {eval_out_dir}")

    if not args.plots:
        return 0

    viz_cmds = [
        (
            "PLOT_RMSE",
            [
                sys.executable,
                "-m",
                "uwnav_dynamics.viz.eval.plot_horizon_metrics",
                "--eval_dir",
                str(eval_out_dir),
                "--out_dir",
                str(plots_dir),
                "--metric",
                "rmse",
                "--dt",
                str(args.dt),
                "--x",
                args.x_axis,
                "--fmt",
                args.plot_fmt,
            ],
        ),
        (
            "PLOT_MAE",
            [
                sys.executable,
                "-m",
                "uwnav_dynamics.viz.eval.plot_horizon_metrics",
                "--eval_dir",
                str(eval_out_dir),
                "--out_dir",
                str(plots_dir),
                "--metric",
                "mae",
                "--dt",
                str(args.dt),
                "--x",
                args.x_axis,
                "--fmt",
                args.plot_fmt,
            ],
        ),
        (
            "PLOT_SAMPLES",
            [
                sys.executable,
                "-m",
                "uwnav_dynamics.viz.eval.plot_rollout_samples",
                "--eval_dir",
                str(eval_out_dir),
                "--out_dir",
                str(plots_dir),
                "--n",
                str(args.n_plot_samples),
                "--dt",
                str(args.dt),
                "--fmt",
                args.plot_fmt,
            ],
        ),
    ]

    print(f"[CLI][EVAL] start visualization stage under: {plots_dir}")
    for stage_name, cmd in viz_cmds:
        ret = _run_stage(stage_name, cmd)
        if ret != 0:
            print(
                f"[CLI][EVAL] visualization failed at stage={stage_name}; "
                f"numeric artifacts were kept under: {eval_out_dir}"
            )
            return ret

    print(f"[CLI][EVAL] plots saved under: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
