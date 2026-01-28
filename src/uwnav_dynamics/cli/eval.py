# src/uwnav_dynamics/cli/eval.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .utils import resolve_run_out_dir, pick_ckpt


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

    cmd = [
        sys.executable, "-m", "uwnav_dynamics.eval.evaluate",
        "--yaml", str(y),
        "--ckpt", str(ckpt),
        "--split", args.split,
        "--save_samples", "256",
    ]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.out_dir is not None:
        cmd += ["--out_dir", args.out_dir]
    if args.plots:
        cmd += ["--plots", "--plot_fmt", args.plot_fmt, "--dt", str(args.dt), "--x_axis", args.x_axis,
                "--n_plot_samples", str(args.n_plot_samples)]

    print("[CLI][EVAL] cmd:", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
