# src/uwnav_dynamics/viz/eval/plot_rollout_samples.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from uwnav_dynamics.viz.style.sci_style import setup_mpl


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm3(x: np.ndarray) -> np.ndarray:
    # x: (...,3)
    return np.sqrt(np.sum(x * x, axis=-1))


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_rollout_samples")
    ap.add_argument("--eval_dir", type=str, required=True, help="evaluation dir containing pred_samples.npz")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--n", type=int, default=8, help="number of sample windows to plot")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "plots")
    _ensure_dir(out_dir)

    z = np.load(eval_dir / "pred_samples.npz")
    y_hat = z["y_hat"]   # (N,H,9)
    y_true = z["y_true"] # (N,H,9)

    setup_mpl()

    N, H, D = y_hat.shape
    nplot = min(int(args.n), N)
    t = np.arange(1, H + 1) * float(args.dt)

    for i in range(nplot):
        yh = y_hat[i]
        yt = y_true[i]

        acc_h = _norm3(yh[:, 0:3]); acc_t = _norm3(yt[:, 0:3])
        gyr_h = _norm3(yh[:, 3:6]); gyr_t = _norm3(yt[:, 3:6])
        vel_h = _norm3(yh[:, 6:9]); vel_t = _norm3(yt[:, 6:9])

        fig, ax = plt.subplots(1, 1)
        ax.plot(t, acc_t, label="||Acc|| true")
        ax.plot(t, acc_h, label="||Acc|| pred")
        ax.plot(t, gyr_t, label="||Gyro|| true")
        ax.plot(t, gyr_h, label="||Gyro|| pred")
        ax.plot(t, vel_t, label="||Vel|| true")
        ax.plot(t, vel_h, label="||Vel|| pred")

        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel("Magnitude")
        ax.legend(loc="best")
        ax.grid(False)

        stem = f"rollout_sample_{i:03d}"
        if args.fmt in ("png", "both"):
            fig.savefig(out_dir / f"{stem}.png")
        if args.fmt in ("pdf", "both"):
            fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)

    print(f"[VIZ] wrote {nplot} sample plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
