"""
模块名称：训练 CLI 入口

模块职责：
作为最小训练入口，负责解析少量常用命令行覆盖项，
并把实际训练工作转交给 `uwnav_dynamics.train.run_train`。

主要功能：
1. 解析 train yaml 与少量高频 override，如 `device/epochs/batch_size`。
2. 复用子进程调用 `train.run_train`，避免在 CLI 层重复实现训练逻辑。
3. 根据 train yaml 推导默认 run 目录，便于用户快速定位训练产物。

数据流：
train yaml + CLI override
    ↓
cli.train
    ↓
uwnav_dynamics.train.run_train
    ↓
run.out_dir/run.variant/

依赖模块：
- uwnav_dynamics.cli.utils
- uwnav_dynamics.train.run_train

备注：
- 本入口刻意只暴露少量覆盖项；需要更完整控制时应直接使用 `run_train.py`。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import resolve_run_out_dir


def main() -> int:
    """解析最小训练参数并转发给正式训练入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.train")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml under configs/train/*.yaml")
    ap.add_argument("--data_dir", type=str, default=None, help="override data.data_dir")
    ap.add_argument("--device", type=str, default=None, help="override run.device (cpu/cuda)")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--batch_size", type=int, default=None, help="override data.batch_size")
    args = ap.parse_args()

    y = Path(args.yaml)

    cmd = [sys.executable, "-m", "uwnav_dynamics.train.run_train", "--yaml", str(y)]
    if args.data_dir is not None:
        cmd += ["--data_dir", args.data_dir]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]

    print("[CLI][TRAIN] cmd:", " ".join(cmd))
    ret = subprocess.run(cmd).returncode

    # 给个提示：训练输出在哪
    out_dir, variant = resolve_run_out_dir(y)
    print(f"[CLI][TRAIN] expected run dir: {out_dir}/{variant}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
