"""
模块名称：多模型评估比较图

模块职责：
从多个评估目录中读取稳定的 horizon 指标 artifact，
生成论文友好的多模型比较图，并固化 primary / baseline / ablation 的视觉层级。

主要功能：
1. 第一版优先实现 horizon compare 主路径。
2. 将 semantic layout 驱动的 Acc / Gyro / Vel 三组误差拆成 3×1 共享 x 轴布局。
3. 自动根据标签或显式 role 推断视觉层级，突出 proposed / primary 方法。

数据流：
多个 eval_dir 下的 metrics.yaml + rmse/mae_by_horizon.csv
    ↓
group aggregation
    ↓
role-aware style mapping
    ↓
plots/rmse_model_compare_horizon.png|pdf

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.eval.plot_horizon_metrics
- uwnav_dynamics.viz.style.sci_style

备注：
- 第一版不实现 global summary 主路径，只在此处保留 TODO。
- rollout compare 若后续接入，也应继续复用同一套 role-aware style token。
- 若评估目录缺少 semantic layout metadata，则统一 warning 并回退到 canonical `acc/gyro/vel` 分组。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from uwnav_dynamics.viz.eval.plot_horizon_metrics import _metric_from_dir
from uwnav_dynamics.viz.style.sci_style import (
    apply_axes_style,
    apply_minimal_legend,
    apply_shared_xlabels,
    get_figure_size,
    get_model_role_style,
    infer_model_role,
    save_figure,
    setup_mpl,
)


@dataclass(frozen=True)
class ModelCompareCfg:
    dt_s: float = 0.01
    use_seconds: bool = True
    metric: str = "rmse"
    fmt: str = "png"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_roles(labels: Sequence[str], explicit_roles: Optional[Sequence[str]] = None) -> List[str]:
    roles: List[str] = []
    for idx, lab in enumerate(labels):
        role_hint = None if explicit_roles is None else explicit_roles[idx]
        roles.append(infer_model_role(lab, explicit_role=role_hint))
    return roles


def build_horizon_compare_figure(
    *,
    eval_dirs: Sequence[Path],
    labels: Sequence[str],
    cfg: ModelCompareCfg,
    roles: Optional[Sequence[str]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    setup_mpl()

    if len(eval_dirs) == 0:
        raise ValueError("At least one eval_dir is required")
    if len(eval_dirs) != len(labels):
        raise ValueError("len(eval_dirs) must equal len(labels)")

    resolved_roles = _resolve_roles(labels, roles)
    order = {"primary": 0, "ablation": 1, "baseline": 2}
    zipped = sorted(zip(eval_dirs, labels, resolved_roles), key=lambda item: order[item[2]])

    hd0, _, semantic_layout0 = _metric_from_dir(Path(zipped[0][0]), cfg.metric)
    H = hd0.shape[0]
    x_steps = np.arange(1, H + 1)
    x = x_steps * cfg.dt_s if cfg.use_seconds else x_steps

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("compare_3row"))
    group_specs = (
        ("acc", f"Acc {cfg.metric.upper()}"),
        ("gyro", f"Gyro {cfg.metric.upper()}"),
        ("vel", f"Vel {cfg.metric.upper()}"),
    )

    for group_ax, (group_key, ylabel) in zip(axes, group_specs):
        for eval_dir, label, role in zipped:
            hd, _, semantic_layout = _metric_from_dir(Path(eval_dir), cfg.metric)
            curve = hd[:, list(semantic_layout.group_indices[group_key])].mean(axis=1)
            sty = get_model_role_style(role)
            group_ax.plot(
                x,
                curve,
                label=label,
                color=sty.color,
                linestyle=sty.linestyle,
                linewidth=sty.linewidth,
                alpha=sty.alpha,
                zorder=sty.zorder,
            )
        group_ax.set_ylabel(ylabel)
        apply_axes_style(group_ax, grid=False)

    apply_shared_xlabels(list(axes), "Prediction horizon (s)" if cfg.use_seconds else "Prediction step (k)")
    apply_minimal_legend(axes[0].legend(loc="upper right"))
    return fig, (axes[0], axes[1], axes[2])


def plot_horizon_compare(
    *,
    eval_dirs: Sequence[Path],
    labels: Sequence[str],
    out_dir: Path,
    cfg: ModelCompareCfg,
    roles: Optional[Sequence[str]] = None,
) -> None:
    _ensure_dir(out_dir)
    fig, _ = build_horizon_compare_figure(eval_dirs=eval_dirs, labels=labels, cfg=cfg, roles=roles)
    save_figure(fig, out_dir / f"{cfg.metric}_model_compare_horizon", fmt=cfg.fmt)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_model_compare")
    ap.add_argument("--eval_dir", type=str, nargs="+", required=True, help="evaluation dirs to compare")
    ap.add_argument("--label", type=str, nargs="*", default=None, help="display labels for each eval dir")
    ap.add_argument(
        "--role",
        type=str,
        nargs="*",
        default=None,
        choices=["primary", "baseline", "ablation"],
        help="optional explicit role list; otherwise infer from labels",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dirs = [Path(p) for p in args.eval_dir]
    labels = list(args.label) if args.label else [p.name for p in eval_dirs]
    if len(labels) != len(eval_dirs):
        raise SystemExit("len(--label) must match len(--eval_dir)")
    if args.role is not None and len(args.role) != len(eval_dirs):
        raise SystemExit("len(--role) must match len(--eval_dir)")

    out_dir = Path(args.out_dir) if args.out_dir else (eval_dirs[0] / "plots")
    cfg = ModelCompareCfg(
        dt_s=float(args.dt),
        use_seconds=(args.x == "sec"),
        metric=str(args.metric),
        fmt=str(args.fmt),
    )
    plot_horizon_compare(eval_dirs=eval_dirs, labels=labels, roles=args.role, out_dir=out_dir, cfg=cfg)
    print(f"[VIZ] wrote model-compare plots to: {out_dir}")
    # TODO(PR-compare-next): rollout compare 与 global summary 可在后续 patch 中补充。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
