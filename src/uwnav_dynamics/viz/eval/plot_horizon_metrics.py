# src/uwnav_dynamics/viz/eval/plot_horizon_metrics.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import yaml

from uwnav_dynamics.viz.style.sci_style import setup_mpl


# -----------------------------
# IO
# -----------------------------

def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_csv_hd(p: Path) -> np.ndarray:
    """
    读取 evaluate.py 保存的 (H,D) CSV（第一列是 h 序号，后续列 d0..d8）
    """
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"CSV too short: {p}")
    data = []
    for row in lines[1:]:
        parts = row.split(",")
        vals = [float(x) for x in parts[1:]]  # skip 'h'
        data.append(vals)
    return np.asarray(data, dtype=float)  # (H,D)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plotting
# -----------------------------

@dataclass(frozen=True)
class HorizonPlotCfg:
    dt_s: float = 0.01
    use_seconds: bool = True
    metric: str = "rmse"   # "rmse" | "mae"
    out_name: str = "rmse_horizon_groups"
    fmt: str = "png"       # "png" | "pdf" | "both"


def _group_slices(dout: int = 9) -> Dict[str, slice]:
    # S1 约定：0:3 acc, 3:6 gyro, 6:9 vel
    if dout != 9:
        raise ValueError("This plot assumes dout=9")
    return {"Acc": slice(0, 3), "Gyro": slice(3, 6), "Vel": slice(6, 9)}


def _metric_from_dir(eval_dir: Path, metric: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    返回：
      hd: (H,D) 指标矩阵
      meta: metrics.yaml 解析结果
    """
    meta = _load_yaml(eval_dir / "metrics.yaml")
    if metric == "rmse":
        hd = _load_csv_hd(eval_dir / "rmse_by_horizon.csv")
    elif metric == "mae":
        hd = _load_csv_hd(eval_dir / "mae_by_horizon.csv")
    else:
        raise ValueError(metric)
    return hd, meta


def plot_groups_vs_horizon(
    eval_dirs: List[Path],
    labels: List[str],
    out_dir: Path,
    cfg: HorizonPlotCfg,
) -> None:
    setup_mpl()

    # x 轴：步数 or 秒
    # 读取第一个的 H
    hd0, _ = _metric_from_dir(eval_dirs[0], cfg.metric)
    H = hd0.shape[0]
    x_steps = np.arange(1, H + 1)
    x = x_steps * cfg.dt_s if cfg.use_seconds else x_steps

    groups = _group_slices(9)

    # --- 画三条曲线：Acc/Gyro/Vel，每个 eval_dir 一组（多模型对比）
    fig, ax = plt.subplots(1, 1)

    for eval_dir, lab in zip(eval_dirs, labels):
        hd, meta = _metric_from_dir(eval_dir, cfg.metric)

        # group 平均：对各自维度取 mean
        y_acc = hd[:, groups["Acc"]].mean(axis=1)
        y_gyro = hd[:, groups["Gyro"]].mean(axis=1)
        y_vel = hd[:, groups["Vel"]].mean(axis=1)

        # 业务逻辑：同一个模型三条线放同一张图会显得拥挤；
        # 这里采用“每个模型画一条线”的策略（默认画 Vel），更适合多模型对比。
        # 但你现在 B0 单模型阶段，建议画三条线。
        # 因此：如果只有一个 eval_dir -> 画三条；多个 -> 默认只画 vel（更清晰）。
        if len(eval_dirs) == 1:
            ax.plot(x, y_acc, label="Acc")
            ax.plot(x, y_gyro, label="Gyro")
            ax.plot(x, y_vel, label="Vel")
        else:
            ax.plot(x, y_vel, label=lab)

    # 坐标轴标签
    if cfg.use_seconds:
        ax.set_xlabel("Prediction horizon (s)")
    else:
        ax.set_xlabel("Prediction step (k)")
    ax.set_ylabel(cfg.metric.upper())

    ax.grid(False)
    ax.legend(loc="best")

    _ensure_dir(out_dir)
    if cfg.fmt in ("png", "both"):
        fig.savefig(out_dir / f"{cfg.out_name}.png")
    if cfg.fmt in ("pdf", "both"):
        fig.savefig(out_dir / f"{cfg.out_name}.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_horizon_metrics")
    ap.add_argument("--eval_dir", type=str, nargs="+", required=True,
                    help="One or more evaluation output dirs (containing metrics.yaml, rmse_by_horizon.csv, ...)")
    ap.add_argument("--label", type=str, nargs="*", default=None,
                    help="Labels for each eval_dir (optional). If not given, use folder names.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for plots. Default: <first_eval_dir>/plots")
    ap.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dirs = [Path(p) for p in args.eval_dir]
    labels = args.label
    if labels is None or len(labels) == 0:
        labels = [p.name for p in eval_dirs]
    if len(labels) != len(eval_dirs):
        raise SystemExit("len(--label) must match len(--eval_dir)")

    out_dir = Path(args.out_dir) if args.out_dir else (eval_dirs[0] / "plots")
    cfg = HorizonPlotCfg(
        dt_s=float(args.dt),
        use_seconds=(args.x == "sec"),
        metric=args.metric,
        out_name=f"{args.metric}_horizon_groups" if len(eval_dirs) == 1 else f"{args.metric}_horizon_compare",
        fmt=args.fmt,
    )
    plot_groups_vs_horizon(eval_dirs=eval_dirs, labels=labels, out_dir=out_dir, cfg=cfg)
    print(f"[VIZ] wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
