"""
模块名称：训练-评估-绘图流水线入口

模块职责：
提供正式的一键流水线入口，按顺序串联训练、checkpoint 选择、数值评估与可选绘图，
并复用各子模块已有的 CLI 契约，而不是在本模块内重写业务逻辑。

主要功能：
1. 调用训练入口生成 run-scoped artifact 与 checkpoint。
2. 复用统一 helper 解析 run_dir，并选择本次评估使用的 checkpoint。
3. 调用 `cli/eval.py` 完成 eval -> viz 编排，形成 train -> eval -> viz 全链路。

数据流：
train yaml + CLI runtime args
    ↓
train.run_train
    ↓
run_dir + checkpoint
    ↓
cli.eval
    ↓
eval artifact + plots artifact

依赖模块：
- uwnav_dynamics.cli.utils
- uwnav_dynamics.train.run_train
- uwnav_dynamics.cli.eval

备注：
- 本模块保持外部 CLI 参数语义稳定，不直接拼接 eval/plots 路径规则。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import resolve_run_out_dir, pick_ckpt


def _run(cmd: list[str]) -> None:
    print("[CLI][PIPE] cmd:", " ".join(cmd))
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise SystemExit(ret)


def main() -> int:
    """训练后评估与绘图联动的命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.pipeline")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml")
    ap.add_argument("--device", type=str, default=None, help="override training device")
    ap.add_argument("--epochs", type=int, default=None, help="override train epochs")
    ap.add_argument("--batch_size", type=int, default=None, help="override batch size")
    ap.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--eval_device", type=str, default=None)
    ap.add_argument("--plots", action="store_true", help="generate plots in evaluation stage")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--n_plot_samples", type=int, default=8)
    args = ap.parse_args()

    y = Path(args.yaml)

    # 1) TRAIN
    cmd_train = [sys.executable, "-m", "uwnav_dynamics.train.run_train", "-y", str(y)]
    if args.device is not None:
        cmd_train += ["--device", args.device]
    if args.epochs is not None:
        cmd_train += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd_train += ["--batch_size", str(args.batch_size)]
    _run(cmd_train)

    # 2) PICK CKPT
    out_dir, variant = resolve_run_out_dir(y)
    run_dir = out_dir / variant
    ckpt = pick_ckpt(run_dir)

    # 3) EVAL (+ optional plots)
    cmd_eval = [
        sys.executable, "-m", "uwnav_dynamics.cli.eval",
        "--yaml", str(y),
        "--ckpt", str(ckpt),
        "--split", args.eval_split,
    ]
    if args.eval_device is not None:
        cmd_eval += ["--device", args.eval_device]
    if args.plots:
        cmd_eval += ["--plots", "--plot_fmt", args.plot_fmt, "--dt", str(args.dt),
                    "--x_axis", args.x_axis, "--n_plot_samples", str(args.n_plot_samples)]
    _run(cmd_eval)

    print("[CLI][PIPE] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
