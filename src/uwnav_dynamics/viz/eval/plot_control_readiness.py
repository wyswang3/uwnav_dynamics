"""
模块名称：控制前诊断绘图

模块职责：
从评估阶段落盘的 `metrics.yaml["control_readiness"]` 中读取离线控制前诊断摘要，
生成单模型 summary 图或多模型 compare 图，
用于判断模型是否值得进入后续 controller-in-the-loop / 闭环验证。

主要功能：
1. 读取 `metrics.yaml["control_readiness"]["physical"]` 中的 dense / masked 诊断摘要。
2. 以 2×2 布局展示末步误差、rollout 增长率、尾部误差与最坏偏差。
3. 单评估目录输出 `control_readiness_summary.*`，多评估目录输出 `control_readiness_compare.*`。
4. 复用仓库统一科研绘图 token，保持与 horizon / compare 图一致的视觉语言。

数据流：
metrics.yaml
    ↓
control_readiness 摘要读取
    ↓
2×2 bar figure
    ↓
plots/control_readiness_summary.png|pdf
或 compare_test/control_readiness_compare.png|pdf

依赖模块：
- yaml
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 当前图只消费 `metrics.yaml` 中已经固化的控制前诊断摘要，不重新计算数值评估。
- `control_readiness` 只用于离线筛查，不等同于闭环控制可用性的最终证明。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uwnav_dynamics.viz.style.sci_style import (
    apply_axes_style,
    get_figure_size,
    get_model_role_styles,
    infer_model_role,
    normalize_model_label,
    save_figure,
    setup_mpl,
)


@dataclass(frozen=True)
class ControlReadinessPlotCfg:
    """控制前诊断图的导出格式与指标空间配置。"""
    fmt: str = "png"
    metric_space: str = "physical"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_roles(labels: Sequence[str], explicit_roles: Optional[Sequence[str]] = None) -> list[str]:
    roles: list[str] = []
    for idx, label in enumerate(labels):
        role_hint = None if explicit_roles is None else explicit_roles[idx]
        roles.append(infer_model_role(label, explicit_role=role_hint))
    return roles


def _load_control_readiness_summary(
    eval_dir: Path,
    *,
    metric_space: str,
    artifact_variant: str,
) -> dict[str, Any]:
    metrics = _load_yaml(eval_dir / "metrics.yaml")
    control_readiness = metrics.get("control_readiness")
    if not isinstance(control_readiness, dict):
        raise ValueError(f"{eval_dir / 'metrics.yaml'} missing control_readiness metadata")

    metric_block = control_readiness.get(metric_space)
    if not isinstance(metric_block, dict):
        raise ValueError(
            f"{eval_dir / 'metrics.yaml'} missing control_readiness.{metric_space!s} metadata"
        )

    summary = metric_block.get(artifact_variant)
    if not isinstance(summary, dict):
        raise ValueError(
            f"{eval_dir / 'metrics.yaml'} missing "
            f"control_readiness.{metric_space}.{artifact_variant} metadata"
        )
    return summary


def _annotation_text(summary: dict[str, Any]) -> str:
    bias = summary.get("bias", {})
    component = bias.get("worst_component", "")
    if not component:
        return ""
    return str(component)


def _extract_scalar(summary: dict[str, Any], path: tuple[str, ...]) -> float:
    cur: Any = summary
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float("nan")
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def build_control_readiness_figure(
    *,
    eval_dirs: Sequence[Path],
    labels: Sequence[str],
    cfg: ControlReadinessPlotCfg,
    roles: Optional[Sequence[str]] = None,
    artifact_variant: str = "dense",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """构建控制前诊断的 2×2 比较图。"""
    setup_mpl()

    if len(eval_dirs) == 0:
        raise ValueError("At least one eval_dir is required")
    if len(eval_dirs) != len(labels):
        raise ValueError("len(eval_dirs) must equal len(labels)")

    display_labels = [normalize_model_label(label) for label in labels]
    resolved_roles = _resolve_roles(display_labels, roles)
    order = {"primary": 0, "ablation": 1, "baseline": 2}
    zipped = sorted(zip(eval_dirs, display_labels, resolved_roles), key=lambda item: order[item[2]])
    role_styles = get_model_role_styles([item[2] for item in zipped])

    summaries = [
        _load_control_readiness_summary(
            Path(eval_dir),
            metric_space=cfg.metric_space,
            artifact_variant=artifact_variant,
        )
        for eval_dir, _, _ in zipped
    ]

    x = np.arange(len(zipped), dtype=float)
    width = 0.62
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("sensor_4x2"))
    flat_axes = tuple(axes.reshape(-1))

    panel_specs = (
        (("final_step", "rmse_global"), "Final-step RMSE"),
        (("rollout_growth", "rmse_last_over_first"), "RMSE growth"),
        (("tail_error", "final_step_abs_p95_global"), "Final-step abs P95"),
        (("bias", "worst_abs_bias"), "Worst |bias|"),
    )

    tick_labels = [label for _, label, _ in zipped]
    rotate = 20 if any(len(label) > 12 for label in tick_labels) else 0

    for idx, (ax, (path, ylabel)) in enumerate(zip(flat_axes, panel_specs)):
        values = [_extract_scalar(summary, path) for summary in summaries]
        bars = ax.bar(
            x,
            values,
            width=width,
            color=[style.color for style in role_styles],
            alpha=0.9,
            zorder=4,
        )
        for bar, value, summary in zip(bars, values, summaries):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#000000",
                    zorder=6,
                )
            if path == ("bias", "worst_abs_bias"):
                annotation = _annotation_text(summary)
                if annotation:
                    ax.text(
                        bar.get_x() + bar.get_width() * 0.5,
                        value * 0.5 if np.isfinite(value) else 0.0,
                        annotation,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#000000",
                        rotation=90,
                        zorder=6,
                    )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x, tick_labels)
        if idx < 2:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        else:
            ax.set_xlabel("Model variant")
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate)
                tick.set_ha("right" if rotate else "center")
        apply_axes_style(ax, grid=False)

    fig.subplots_adjust(hspace=0.28, wspace=0.20, top=0.97)
    return fig, (flat_axes[0], flat_axes[1], flat_axes[2], flat_axes[3])


def plot_control_readiness(
    *,
    eval_dirs: Sequence[Path],
    labels: Sequence[str],
    out_dir: Path,
    cfg: ControlReadinessPlotCfg,
    roles: Optional[Sequence[str]] = None,
) -> None:
    """从评估目录读盘并导出控制前诊断图。"""
    _ensure_dir(out_dir)
    out_name = "control_readiness_summary" if len(eval_dirs) == 1 else "control_readiness_compare"

    fig, _ = build_control_readiness_figure(
        eval_dirs=eval_dirs,
        labels=labels,
        cfg=cfg,
        roles=roles,
        artifact_variant="dense",
    )
    save_figure(fig, out_dir / out_name, fmt=cfg.fmt)
    plt.close(fig)

    fig_masked, _ = build_control_readiness_figure(
        eval_dirs=eval_dirs,
        labels=labels,
        cfg=cfg,
        roles=roles,
        artifact_variant="masked",
    )
    save_figure(fig_masked, out_dir / f"{out_name}_masked", fmt=cfg.fmt)
    plt.close(fig_masked)


def main() -> int:
    """控制前诊断图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_control_readiness")
    ap.add_argument("--eval_dir", type=str, nargs="+", required=True, help="one or more evaluation dirs")
    ap.add_argument("--label", type=str, nargs="*", default=None, help="display labels for each eval_dir")
    ap.add_argument(
        "--role",
        type=str,
        nargs="*",
        default=None,
        choices=["primary", "baseline", "ablation"],
        help="optional explicit role list; otherwise infer from labels",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dirs = [Path(p) for p in args.eval_dir]
    labels = list(args.label) if args.label else [p.name for p in eval_dirs]
    if len(labels) != len(eval_dirs):
        raise SystemExit("len(--label) must match len(--eval_dir)")
    if args.role is not None and len(args.role) != len(eval_dirs):
        raise SystemExit("len(--role) must match len(--eval_dir)")

    out_dir = Path(args.out_dir) if args.out_dir else (eval_dirs[0] / "plots")
    plot_control_readiness(
        eval_dirs=eval_dirs,
        labels=labels,
        roles=args.role,
        out_dir=out_dir,
        cfg=ControlReadinessPlotCfg(fmt=str(args.fmt)),
    )
    print(f"[VIZ] wrote control-readiness plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
