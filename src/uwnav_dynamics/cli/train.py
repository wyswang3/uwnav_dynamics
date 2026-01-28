# src/uwnav_dynamics/cli/train.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import resolve_run_out_dir


def main() -> int:
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
