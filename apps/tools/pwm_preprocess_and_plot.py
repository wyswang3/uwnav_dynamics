#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- make src/ importable when running as a script ----
ROOT = Path(__file__).resolve().parents[2]  # apps/tools -> repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uwnav_dynamics.io.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.pwm_reader import read_pwm_csv

# 复用统一绘图风格（你需要在 uwnav_dynamics/viz/style.py 里提供同名函数）
from uwnav_dynamics.viz.style.sci_style import setup_mpl, apply_axes_2d


_CMD_COLS = [f"ch{i}_cmd" for i in range(1, 9)]


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    """Uniform downsample indices so that len(idx) <= max_points."""
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    step = int(np.ceil(n / max_points))
    return np.arange(0, n, step, dtype=int)


def _palette_8() -> List[str]:
    """
    8-color palette (paper-friendly). At least 2 colors as requested.
    """
    return [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]

def plot_pwm_8ch_cmd_only(
    est_s: np.ndarray,
    cmd: np.ndarray,
    out_png: Path,
    *,
    max_points: int = 40000,
) -> None:
    """
    Plot 8 channels cmd only, 4 rows x 2 cols.
    Minimal figure:
      - no legend
      - no suptitle
      - each panel title: CH1..CH8 only
      - no y-axis label (avoid repetitive "PWM cmd")
    """
    if cmd.ndim != 2 or cmd.shape[1] != 8:
        raise ValueError(f"cmd shape must be (N,8), got {cmd.shape}")

    setup_mpl()
    colors = _palette_8()

    n = est_s.size
    idx = _downsample_idx(n, max_points)
    t = est_s[idx]
    cmd_d = cmd[idx, :]

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.6), sharex=True)
    axes = axes.reshape(-1)

    for i in range(8):
        ax = axes[i]
        apply_axes_2d(ax)
        ax.plot(t, cmd_d[:, i], color=colors[i])
        ax.set_title(f"CH{i+1}")

        # No y-axis label to avoid repetitive info
        ax.set_ylabel("")

        # Only show x label on bottom row for cleanliness
        if i >= 6:
            ax.set_xlabel("EstS (s)")
        else:
            ax.set_xlabel("")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="PWM preprocess: add EstS/EstNS, export cmd-only csv, plot 8ch cmd (4x2)."
    )
    ap.add_argument("--dataset_yaml", required=True, help="configs/dataset/*.yaml")
    ap.add_argument(
        "--out_dir",
        default="out/pwm",
        help="Output base dir (relative to repo root). Default: out/pwm",
    )
    ap.add_argument("--max_points", type=int, default=40000, help="Downsample for plotting (0=disable)")
    args = ap.parse_args()

    spec = DatasetSpec.load(args.dataset_yaml)
    spec.ensure_paths_exist(require_imu=True, require_pwm=True)

    kw = spec.pwm_reader_kwargs()
    pwm = read_pwm_csv(**kw)  # returns df with EstS/EstNS

    # ---- export csv: keep only cmd + time columns ----
    df = pwm.df.copy()

    keep_cols = ["t_s", "EstS", "EstNS"] + _CMD_COLS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Aligned PWM df missing columns: {missing}")

    out_base = (ROOT / args.out_dir / spec.meta.dataset_id).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    src_name = Path(kw["csv_path"]).stem
    out_csv = out_base / f"{src_name}_cmd_aligned.csv"
    out_png = out_base / f"{src_name}_cmd_8ch.png"

    df_out = df[keep_cols]
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[PWM] wrote cmd-only aligned csv: {out_csv}")

    # ---- plot (cmd only, no legend) ----
    plot_pwm_8ch_cmd_only(
        df_out["EstS"].to_numpy(dtype=float),
        df_out[_CMD_COLS].to_numpy(dtype=float),
        out_png,
        max_points=int(args.max_points),
    )

    print(f"[PWM] wrote plot: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
