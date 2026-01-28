# src/uwnav_dynamics/viz/eval/plot_rollout_samples.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from uwnav_dynamics.viz.style.sci_style import setup_mpl


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm3(x: np.ndarray) -> np.ndarray:
    """
    x: (...,3) -> (...,)
    用于将 Acc/Gyro/Vel 三轴合成为一个 magnitude 曲线，便于快速判断：
      - 相位滞后
      - 漂移/发散
      - 高/低频拟合偏差
    """
    return np.sqrt(np.sum(x * x, axis=-1))


def plot_rollout_samples_from_npz(
    pred_npz: Path,
    out_dir: Path,
    *,
    dt_s: float = 0.01,
    n: int = 8,
    fmt: str = "png",      # "png" | "pdf" | "both"
) -> None:
    """
    从 pred_samples.npz 生成时域对比图（magnitude 版）

    输入文件约定（由 eval/evaluate.py 产出）：
      - y_hat:  (N,H,9)
      - y_true: (N,H,9)
      - logvar: (N,H,9)  (当前不画，但保留供未来画置信区间)

    输出：
      out_dir/rollout_sample_000.png 等

    业务逻辑：
      - 我们把 9 维输出按 [Acc3, Gyro3, Vel3] 分成三段；
      - 每段用 ||·|| 合成一条曲线（更适合 sanity-check）；
      - 后续如果需要画每个维度的对比，可在此函数基础上扩展。
    """
    pred_npz = Path(pred_npz)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    z = np.load(pred_npz)
    if "y_hat" not in z or "y_true" not in z:
        raise ValueError(f"{pred_npz} must contain keys: y_hat, y_true (and optionally logvar)")

    y_hat = z["y_hat"]    # (N,H,9)
    y_true = z["y_true"]  # (N,H,9)

    if y_hat.ndim != 3 or y_true.ndim != 3:
        raise ValueError(f"Expect y_hat/y_true to be 3D, got {y_hat.shape}, {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat={y_hat.shape}, y_true={y_true.shape}")
    if y_hat.shape[-1] != 9:
        raise ValueError(f"Expect last dim = 9 (Acc3+Gyro3+Vel3), got {y_hat.shape[-1]}")

    setup_mpl()

    N, H, _ = y_hat.shape
    nplot = min(int(n), int(N))
    t = np.arange(1, H + 1, dtype=float) * float(dt_s)

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
        if fmt in ("png", "both"):
            fig.savefig(out_dir / f"{stem}.png")
        if fmt in ("pdf", "both"):
            fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)


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

    plot_rollout_samples_from_npz(
        pred_npz=eval_dir / "pred_samples.npz",
        out_dir=out_dir,
        dt_s=float(args.dt),
        n=int(args.n),
        fmt=str(args.fmt),
    )
    print(f"[VIZ] wrote rollout sample plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
