"""
模块名称：评估 horizon 指标绘图

模块职责：
从评估目录中读取数值评估阶段已经落盘的指标文件，
并生成 horizon 维度上的 RMSE / MAE 曲线图。

主要功能：
1. 读取 `metrics.yaml` 与 horizon CSV artifact。
2. 优先根据 `metrics.yaml.layout.semantic` 对输出分组聚合并绘图。
3. 支持 dense / masked horizon artifact 并行存在，且不覆盖现有 dense 输出命名。
4. 支持单评估目录出图与多评估目录对比出图。

数据流：
eval_<split>/metrics.yaml + rmse_by_horizon.csv / mae_by_horizon.csv
    ↓
load + group aggregation
    ↓
matplotlib figure
    ↓
plots/rmse_horizon_*.png|pdf / mae_horizon_*.png|pdf
以及可选 *_masked.png|pdf

依赖模块：
- matplotlib
- yaml
- numpy
- uwnav_dynamics.viz.style.sci_style

备注：
- 当前默认消费 PR3 保持稳定的 horizon CSV artifact，
  并在旧 artifact 缺少 layout metadata 时统一 fallback 到 canonical `acc/gyro/vel` 分组。
- PR5 通过平行的 masked CSV 扩展，不替换现有 dense CSV 契约。
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import yaml

from uwnav_dynamics.models.utils.semantic_output_layout import (
    SemanticOutputLayout,
    load_semantic_layout_from_metrics_dict,
)
from uwnav_dynamics.viz.style.sci_style import (
    apply_axes_style,
    apply_minimal_legend,
    get_figure_size,
    get_group_styles,
    get_model_role_style,
    infer_model_role,
    save_figure,
    setup_mpl,
)


_GROUP_DISPLAY_NAMES = {
    "acc": "Acc",
    "gyro": "Gyro",
    "vel": "Vel",
}


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


def _metric_csv_name(metric: str, artifact_variant: str) -> str:
    suffix = "" if artifact_variant == "dense" else "_masked"
    if metric == "rmse":
        return f"rmse_by_horizon{suffix}.csv"
    if metric == "mae":
        return f"mae_by_horizon{suffix}.csv"
    raise ValueError(metric)


def _metric_from_dir(
    eval_dir: Path,
    metric: str,
    artifact_variant: str = "dense",
) -> Tuple[np.ndarray, Dict[str, Any], SemanticOutputLayout]:
    """
    返回：
      hd: (H,D) 指标矩阵
      meta: metrics.yaml 解析结果
    """
    meta = _load_yaml(eval_dir / "metrics.yaml")
    hd = _load_csv_hd(eval_dir / _metric_csv_name(metric, artifact_variant))
    semantic_layout = load_semantic_layout_from_metrics_dict(meta, dout=hd.shape[1])
    return hd, meta, semantic_layout


def _group_curves(hd: np.ndarray, semantic_layout: SemanticOutputLayout) -> Dict[str, np.ndarray]:
    curves: Dict[str, np.ndarray] = {}
    for key, indices in semantic_layout.group_indices.items():
        vals = hd[:, list(indices)]
        valid_count = np.sum(~np.isnan(vals), axis=1)
        curve = np.full(vals.shape[0], np.nan, dtype=float)
        valid = valid_count > 0
        if np.any(valid):
            curve[valid] = np.nansum(vals[valid], axis=1) / valid_count[valid]
        curves[key] = curve
    return curves


def build_groups_vs_horizon_figure(
    eval_dirs: List[Path],
    labels: List[str],
    cfg: HorizonPlotCfg,
    artifact_variant: str = "dense",
) -> Tuple[plt.Figure, plt.Axes]:
    setup_mpl()

    hd0, _, semantic_layout0 = _metric_from_dir(eval_dirs[0], cfg.metric, artifact_variant=artifact_variant)
    H = hd0.shape[0]
    x_steps = np.arange(1, H + 1)
    x = x_steps * cfg.dt_s if cfg.use_seconds else x_steps
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size("single"))

    group_styles = get_group_styles()
    for eval_dir, lab in zip(eval_dirs, labels):
        hd, _, semantic_layout = _metric_from_dir(eval_dir, cfg.metric, artifact_variant=artifact_variant)
        curves = _group_curves(hd, semantic_layout)

        if len(eval_dirs) == 1:
            for key in semantic_layout0.group_indices:
                display_name = _GROUP_DISPLAY_NAMES[key]
                sty = group_styles[display_name]
                ax.plot(
                    x,
                    curves[key],
                    label=display_name,
                    color=sty.color,
                    linestyle=sty.linestyle,
                    linewidth=sty.linewidth,
                    alpha=sty.alpha,
                    zorder=sty.zorder,
                )
        else:
            role = infer_model_role(lab)
            sty = get_model_role_style(role)
            ax.plot(
                x,
                curves["vel"],
                label=lab,
                color=sty.color,
                linestyle=sty.linestyle,
                linewidth=sty.linewidth,
                alpha=sty.alpha,
                zorder=sty.zorder,
            )

    ax.set_xlabel("Prediction horizon (s)" if cfg.use_seconds else "Prediction step (k)")
    ax.set_ylabel(cfg.metric.upper())
    apply_axes_style(ax, grid=False)
    apply_minimal_legend(ax.legend(loc="best"))
    return fig, ax


def plot_groups_vs_horizon(
    eval_dirs: List[Path],
    labels: List[str],
    out_dir: Path,
    cfg: HorizonPlotCfg,
) -> None:
    _ensure_dir(out_dir)
    fig, _ = build_groups_vs_horizon_figure(
        eval_dirs=eval_dirs,
        labels=labels,
        cfg=cfg,
        artifact_variant="dense",
    )
    save_figure(fig, out_dir / cfg.out_name, fmt=cfg.fmt)
    plt.close(fig)

    masked_csv_name = _metric_csv_name(cfg.metric, "masked")
    has_masked = [((Path(eval_dir) / masked_csv_name).exists()) for eval_dir in eval_dirs]
    if any(has_masked):
        if not all(has_masked):
            warnings.warn(
                f"masked horizon artifact missing for part of eval_dirs; skip masked plot for {cfg.metric}",
                UserWarning,
            )
            return
        fig_masked, _ = build_groups_vs_horizon_figure(
            eval_dirs=eval_dirs,
            labels=labels,
            cfg=cfg,
            artifact_variant="masked",
        )
        save_figure(fig_masked, out_dir / f"{cfg.out_name}_masked", fmt=cfg.fmt)
        plt.close(fig_masked)


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
