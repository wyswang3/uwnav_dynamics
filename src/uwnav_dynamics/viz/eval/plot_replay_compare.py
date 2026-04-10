"""
模块名称：replay 模型筛选比较图

模块职责：
面向“多个候选网络按统一长序列 replay 协议比较”的场景，
从各 run 的 `metrics.yaml` 与 `step_metrics.csv` 读取稳定 artifact，
生成可直接用于模型筛选与技术汇报的 summary / 长线曲线图。

主要功能：
1. 汇总 replay 主指标，生成多模型 2x3 summary compare 图。
2. 读取逐步 `step_metrics.csv`，生成误差包络与生存率曲线图。
3. 复用仓库已有 role-aware 风格，突出 primary / ablation / baseline 层级。

数据流：
replay_run_dir/metrics.yaml + step_metrics.csv
    ↓
summary compare / long-horizon curve compare
    ↓
plots/replay_model_compare.png|pdf
plots/replay_long_horizon_curves.png|pdf

依赖模块：
- numpy
- matplotlib
- yaml
- uwnav_dynamics.viz.style.sci_style

备注：
- 当前图服务“离线状态转移模型筛选”，不是闭环控制最终结论。
- 曲线横轴默认使用相对 step，而不是绝对秒数，避免强依赖数据频率元信息。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uwnav_dynamics.viz.style.sci_style import (
    add_figure_legend,
    apply_axes_style,
    get_figure_size,
    get_model_role_styles,
    infer_model_role,
    normalize_model_label,
    save_figure,
    setup_mpl,
)


@dataclass(frozen=True)
class ReplayComparePlotCfg:
    """replay compare 图的导出格式配置。"""

    fmt: str = "png"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"yaml must be a mapping: {path}")
    return data


def _load_step_rows(path: Path) -> list[dict[str, float]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def _nested_get(mapping: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_roles(labels: Sequence[str], explicit_roles: Optional[Sequence[str]] = None) -> list[str]:
    roles: list[str] = []
    for idx, label in enumerate(labels):
        role_hint = None if explicit_roles is None else explicit_roles[idx]
        roles.append(infer_model_role(label, explicit_role=role_hint))
    return roles


def build_replay_summary_figure(
    *,
    run_dirs: Sequence[Path],
    labels: Sequence[str],
    cfg: ReplayComparePlotCfg,
    roles: Optional[Sequence[str]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """构建 replay 主指标 summary compare 图。"""

    del cfg
    setup_mpl()
    if len(run_dirs) == 0:
        raise ValueError("At least one replay run dir is required")
    if len(run_dirs) != len(labels):
        raise ValueError("len(run_dirs) must equal len(labels)")

    display_labels = [normalize_model_label(label) for label in labels]
    resolved_roles = _resolve_roles(display_labels, roles)
    order = {"primary": 0, "ablation": 1, "baseline": 2}
    zipped = sorted(zip(run_dirs, display_labels, resolved_roles), key=lambda item: order[item[2]])
    role_styles = get_model_role_styles([item[2] for item in zipped])
    metrics_list = [_load_yaml(Path(run_dir) / "metrics.yaml") for run_dir, _, _ in zipped]

    fig, axes = plt.subplots(2, 3, figsize=get_figure_size("sensor_4x2"))
    flat_axes = tuple(axes.reshape(-1))
    x = np.arange(len(zipped), dtype=float)
    width = 0.62
    tick_labels = [label for _, label, _ in zipped]
    rotate = 20 if any(len(label) > 10 for label in tick_labels) else 0

    panel_specs = (
        (("rmse_global",), "Global RMSE"),
        (("final_step", "rmse_global_mean"), "Final-Step RMSE"),
        (("rollout_growth", "rmse_last_over_first_p95"), "Growth P95"),
        (("tail_error", "abs_p95_global"), "Tail Abs P95"),
        (("long_horizon", "time_to_threshold", "rmse", "failure_rate"), "RMSE Threshold Fail"),
        (("bias", "worst_abs_bias"), "Worst |Bias|"),
    )

    for idx, (ax, (path, title)) in enumerate(zip(flat_axes, panel_specs)):
        values = [float(_nested_get(metrics, *path)) for metrics in metrics_list]
        bars = ax.bar(
            x,
            values,
            width=width,
            color=[style.color for style in role_styles],
            alpha=0.9,
            zorder=4,
        )
        for bar, value in zip(bars, values):
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
        ax.set_title(title)
        ax.set_xticks(x, tick_labels)
        if idx < 3:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        else:
            ax.set_xlabel("Model variant")
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate)
                tick.set_ha("right" if rotate else "center")
        apply_axes_style(ax, grid=False)

    fig.subplots_adjust(hspace=0.30, wspace=0.22, top=0.95, bottom=0.16)
    return fig, axes


def build_replay_long_horizon_figure(
    *,
    run_dirs: Sequence[Path],
    labels: Sequence[str],
    cfg: ReplayComparePlotCfg,
    roles: Optional[Sequence[str]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """构建 replay 逐步误差与生存率曲线图。"""

    del cfg
    setup_mpl()
    if len(run_dirs) == 0:
        raise ValueError("At least one replay run dir is required")
    if len(run_dirs) != len(labels):
        raise ValueError("len(run_dirs) must equal len(labels)")

    display_labels = [normalize_model_label(label) for label in labels]
    resolved_roles = _resolve_roles(display_labels, roles)
    order = {"primary": 0, "ablation": 1, "baseline": 2}
    zipped = sorted(zip(run_dirs, display_labels, resolved_roles), key=lambda item: order[item[2]])
    role_styles = get_model_role_styles([item[2] for item in zipped])
    step_tables = [_load_step_rows(Path(run_dir) / "step_metrics.csv") for run_dir, _, _ in zipped]

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("sensor_4x2"))
    flat_axes = tuple(axes.reshape(-1))
    panel_specs = (
        ("rmse_global", "Step RMSE"),
        ("abs_p95_global", "Step Abs P95"),
        ("rmse_survival_rate", "RMSE Survival"),
        ("abs_survival_rate", "Abs Survival"),
    )

    for ax, (metric_name, ylabel) in zip(flat_axes, panel_specs):
        for idx, (_run_dir, label, _role) in enumerate(zipped):
            rows = step_tables[idx]
            x = np.asarray([row["step"] for row in rows], dtype=np.float64)
            y = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
            style = role_styles[idx]
            ax.plot(
                x,
                y,
                label=label,
                color=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                alpha=style.alpha,
                zorder=style.zorder,
            )
        ax.set_ylabel(ylabel)
        apply_axes_style(ax, grid=False)

    flat_axes[0].set_xlabel("")
    flat_axes[1].set_xlabel("")
    flat_axes[2].set_xlabel("Replay step (k)")
    flat_axes[3].set_xlabel("Replay step (k)")

    handles, legend_labels = flat_axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.28, wspace=0.20)
    add_figure_legend(fig, handles, legend_labels, ncol=min(len(legend_labels), 4), y=0.985)
    return fig, axes


def plot_replay_compare(
    *,
    run_dirs: Sequence[Path],
    labels: Sequence[str],
    out_dir: Path,
    cfg: ReplayComparePlotCfg,
    roles: Optional[Sequence[str]] = None,
) -> None:
    """读取 replay artifact 并导出 summary / long-horizon compare 图。"""
    _ensure_dir(out_dir)

    fig_summary, _ = build_replay_summary_figure(
        run_dirs=run_dirs,
        labels=labels,
        cfg=cfg,
        roles=roles,
    )
    save_figure(fig_summary, out_dir / "replay_model_compare", fmt=cfg.fmt)
    plt.close(fig_summary)

    fig_curves, _ = build_replay_long_horizon_figure(
        run_dirs=run_dirs,
        labels=labels,
        cfg=cfg,
        roles=roles,
    )
    save_figure(fig_curves, out_dir / "replay_long_horizon_curves", fmt=cfg.fmt)
    plt.close(fig_curves)


def main() -> int:
    """replay compare 图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.eval.plot_replay_compare")
    ap.add_argument("--run_dir", type=str, nargs="+", required=True, help="replay run dirs to compare")
    ap.add_argument("--label", type=str, nargs="*", default=None, help="display labels for each replay run dir")
    ap.add_argument(
        "--role",
        type=str,
        nargs="*",
        default=None,
        choices=["primary", "baseline", "ablation"],
        help="optional explicit role list; otherwise infer from labels",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    labels = list(args.label) if args.label else [p.name for p in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("len(--label) must match len(--run_dir)")
    if args.role is not None and len(args.role) != len(run_dirs):
        raise SystemExit("len(--role) must match len(--run_dir)")

    plot_replay_compare(
        run_dirs=run_dirs,
        labels=labels,
        roles=args.role,
        out_dir=Path(args.out_dir),
        cfg=ReplayComparePlotCfg(fmt=str(args.fmt)),
    )
    print(f"[VIZ] wrote replay compare plots to: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
