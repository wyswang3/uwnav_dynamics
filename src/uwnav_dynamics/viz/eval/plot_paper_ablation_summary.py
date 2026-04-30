"""
模块名称：论文用消融/路线汇总图

模块职责：
从训练矩阵、replay 排名或最终选模表中读取稳定汇总字段，
生成适合论文结果章节使用的紧凑对比图，
把控制相关指标从“表格读数”升级为“可快速比较的证据图”。

主要功能：
1. 支持读取 `summary.csv`、`ranking.csv` 与 `final_selection.csv` 三类汇总表。
2. 提取 MAE、末步误差、尾部误差、误差增长与最坏偏差等控制相关指标。
3. 将不同指标归一化到“best = 1.0”坐标系，生成横向 dot-plot。
4. 复用仓库统一的 primary / ablation / baseline 视觉层级。
5. 支持按技术路线导出“路线内最优模块图 + 跨路线赢家对比图”套图。

数据流：
summary.csv / ranking.csv / final_selection.csv
    ↓
字段抽取 + lower-is-better 归一化
    ↓
role-aware dot-plot
    ↓
plots/paper_ablation_summary_*.png|pdf

依赖模块：
- csv
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 本图只做离线论文证据汇总，不替代原始 `metrics.yaml` 与 compare 曲线图。
- 所有指标默认按 “越低越好” 解释；若后续加入 success-rate 类指标，应单独扩展。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

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
class MetricSpec:
    """单个论文汇总指标的读取与展示定义。"""

    field: str
    label: str


@dataclass(frozen=True)
class RouteComparisonSuitePaths:
    """技术路线对比套图的产物路径集合。"""
    route_plot_paths: tuple[Path, ...]
    winner_compare_path: Path


_METRIC_MODES: dict[str, tuple[MetricSpec, ...]] = {
    "train_masked": (
        MetricSpec("mae_global_masked", "Masked MAE"),
        MetricSpec("final_step_mae_global_masked", "Final-step MAE"),
        MetricSpec("tail_abs_p95_masked", "Tail abs P95"),
        MetricSpec("rmse_growth_masked", "RMSE growth"),
        MetricSpec("worst_abs_bias_masked", "Worst |bias|"),
    ),
    "replay": (
        MetricSpec("mae_global", "Replay MAE"),
        MetricSpec("final_step_mae_global_mean", "Replay final-step MAE"),
        MetricSpec("tail_abs_p95_global", "Replay tail abs P95"),
        MetricSpec("rmse_growth_p95", "Replay growth P95"),
        MetricSpec("worst_abs_bias", "Replay worst |bias|"),
    ),
    "final_selection_train_masked": (
        MetricSpec("train_mae_global_masked", "Masked MAE"),
        MetricSpec("train_final_step_rmse_global_masked", "Final-step RMSE"),
        MetricSpec("train_tail_abs_p95_masked", "Tail abs P95"),
        MetricSpec("train_rmse_growth_masked", "RMSE growth"),
        MetricSpec("train_worst_abs_bias_masked", "Worst |bias|"),
    ),
    "final_selection_replay": (
        MetricSpec("replay_mae_global", "Replay MAE"),
        MetricSpec("replay_final_step_rmse_global_mean", "Replay final-step RMSE"),
        MetricSpec("replay_tail_abs_p95_global", "Replay tail abs P95"),
        MetricSpec("replay_rmse_growth_p95", "Replay growth P95"),
        MetricSpec("replay_worst_abs_bias", "Replay worst |bias|"),
    ),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_float(value: object) -> float:
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _pick_display_label(row: dict[str, str]) -> str:
    for key in ("label", "name"):
        val = str(row.get(key, "")).strip()
        if val != "":
            return normalize_model_label(val)
    return "unknown"


def _pick_role(row: dict[str, str], label: str) -> str:
    role = str(row.get("role", "")).strip()
    return infer_model_role(label, explicit_role=role if role != "" else None)


def _filter_rows(
    rows: Sequence[dict[str, str]],
    *,
    include_labels: Sequence[str] | None,
    include_names: Sequence[str] | None,
    include_scopes: Sequence[str] | None = None,
    winner_only: bool = False,
) -> list[dict[str, str]]:
    scope_set = {item.strip() for item in (include_scopes or []) if item.strip()}
    if not include_labels and not include_names and not scope_set and not winner_only:
        return list(rows)

    label_set = {item.strip() for item in (include_labels or []) if item.strip()}
    name_set = {item.strip() for item in (include_names or []) if item.strip()}
    kept: list[dict[str, str]] = []
    for row in rows:
        if winner_only and str(row.get("is_scope_winner", "")).strip().lower() not in {"1", "true", "yes"}:
            continue
        if scope_set:
            scope = str(row.get("selection_scope", "")).strip()
            if scope not in scope_set:
                continue
        raw_label = str(row.get("label", "")).strip()
        raw_name = str(row.get("name", "")).strip()
        norm_label = normalize_model_label(raw_label) if raw_label else ""
        if not label_set and not name_set:
            kept.append(row)
            continue
        if raw_label in label_set or norm_label in label_set or raw_name in name_set:
            kept.append(row)
    return kept


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    best = float(np.min(finite))
    if best == 0.0:
        out = np.full_like(values, np.nan, dtype=float)
        out[np.isfinite(values) & (values == 0.0)] = 1.0
        return out
    return values / best


def _slugify_name(text: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_") or "unnamed"


def _truthy_flag(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _scope_alias_map(
    *,
    rows: Sequence[dict[str, str]],
    route_scopes: Sequence[str] | None,
    route_names: Sequence[str] | None,
) -> list[tuple[str, str]]:
    if route_scopes and len(route_scopes) > 0:
        scopes = [str(item).strip() for item in route_scopes if str(item).strip()]
    else:
        scopes = []
        for row in rows:
            scope = str(row.get("selection_scope", "")).strip()
            if scope and scope not in scopes:
                scopes.append(scope)
    names = [str(item).strip() for item in (route_names or []) if str(item).strip()]
    pairs: list[tuple[str, str]] = []
    for idx, scope in enumerate(scopes):
        alias = names[idx] if idx < len(names) else scope.replace("_", " ")
        pairs.append((scope, alias))
    return pairs


def _clone_rows_for_route_scope(
    rows: Sequence[dict[str, str]],
    *,
    scope: str,
) -> list[dict[str, str]]:
    selected = _filter_rows(
        rows,
        include_labels=None,
        include_names=None,
        include_scopes=[scope],
        winner_only=False,
    )
    cloned: list[dict[str, str]] = []
    for row in selected:
        item = dict(row)
        label = _pick_display_label(item)
        if _truthy_flag(item.get("is_scope_winner", "")):
            item["label"] = f"{label} (winner)"
        else:
            item["label"] = label
        cloned.append(item)
    return cloned


def _clone_rows_for_route_winners(
    rows: Sequence[dict[str, str]],
    *,
    scope_alias_pairs: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    alias_by_scope = {scope: alias for scope, alias in scope_alias_pairs}
    winner_rows = _filter_rows(
        rows,
        include_labels=None,
        include_names=None,
        include_scopes=list(alias_by_scope.keys()),
        winner_only=True,
    )
    cloned: list[dict[str, str]] = []
    for row in winner_rows:
        item = dict(row)
        label = _pick_display_label(item)
        scope = str(item.get("selection_scope", "")).strip()
        alias = alias_by_scope.get(scope, scope)
        item["label"] = f"{alias}: {label}"
        cloned.append(item)
    return cloned


def build_paper_ablation_summary_figure(
    *,
    rows: Sequence[dict[str, str]],
    mode: str,
) -> tuple[plt.Figure, plt.Axes]:
    """构建论文用控制相关指标 dot-plot。"""
    if mode not in _METRIC_MODES:
        raise KeyError(f"Unknown mode: {mode!r}")
    if len(rows) == 0:
        raise ValueError("No rows available for plotting")

    setup_mpl()

    enriched = []
    for row in rows:
        label = _pick_display_label(row)
        role = _pick_role(row, label)
        enriched.append((row, label, role))
    order = {"primary": 0, "ablation": 1, "baseline": 2}
    enriched = sorted(enriched, key=lambda item: (order[item[2]], item[1]))
    role_styles = get_model_role_styles([item[2] for item in enriched])

    metric_specs = _METRIC_MODES[mode]
    metric_labels = [spec.label for spec in metric_specs]
    values = np.asarray(
        [[_safe_float(row.get(spec.field, "")) for spec in metric_specs] for row, _label, _role in enriched],
        dtype=float,
    )
    norm_values = np.vstack([_normalize_metric(values[:, j]) for j in range(values.shape[1])]).T

    fig, ax = plt.subplots(figsize=get_figure_size("single"))
    y_base = np.arange(len(metric_specs), dtype=float)[::-1]
    offsets = np.linspace(-0.24, 0.24, num=max(len(enriched), 2), dtype=float)
    if len(enriched) == 1:
        offsets = np.asarray([0.0], dtype=float)

    for idx, ((_row, label, _role), style) in enumerate(zip(enriched, role_styles)):
        y = y_base + offsets[idx]
        x = norm_values[idx]
        valid = np.isfinite(x)
        if not np.any(valid):
            continue
        for xv, yv in zip(x[valid], y[valid]):
            ax.hlines(yv, xmin=min(1.0, float(xv)), xmax=max(1.0, float(xv)), color=style.color, alpha=0.24, lw=1.1)
        ax.scatter(
            x[valid],
            y[valid],
            s=34.0,
            color=style.color,
            alpha=style.alpha,
            zorder=style.zorder,
            label=label,
        )

    ax.axvline(1.0, color="#B8C4CF", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_base, metric_labels)
    ax.set_xlabel("Relative metric to best (=1.0, lower is better)")
    ax.set_xlim(left=0.92, right=max(1.05, float(np.nanmax(norm_values[np.isfinite(norm_values)]) * 1.06)))
    apply_axes_style(ax, grid=False)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)

    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.31, right=0.98)
    add_figure_legend(fig, uniq_handles, uniq_labels, ncol=min(len(uniq_labels), 4), y=0.98)
    return fig, ax


def plot_paper_ablation_summary(
    *,
    csv_path: Path,
    out_dir: Path,
    mode: str,
    include_labels: Sequence[str] | None = None,
    include_names: Sequence[str] | None = None,
    include_scopes: Sequence[str] | None = None,
    winner_only: bool = False,
    fmt: str = "png",
) -> Path:
    """读取汇总表并导出论文用紧凑对比图。"""
    rows = _read_rows(csv_path)
    rows = _filter_rows(
        rows,
        include_labels=include_labels,
        include_names=include_names,
        include_scopes=include_scopes,
        winner_only=winner_only,
    )
    if len(rows) == 0:
        raise ValueError(f"No rows selected from {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = build_paper_ablation_summary_figure(rows=rows, mode=mode)
    out_stem = out_dir / f"paper_ablation_summary_{mode}"
    save_figure(fig, out_stem, fmt=fmt)
    plt.close(fig)
    return out_stem.with_suffix(".png")


def plot_route_comparison_suite(
    *,
    csv_path: Path,
    out_dir: Path,
    mode: str,
    route_scopes: Sequence[str] | None = None,
    route_names: Sequence[str] | None = None,
    fmt: str = "png",
) -> RouteComparisonSuitePaths:
    """
    导出技术路线对比套图：
    1. 每条路线内部的模块对比图；
    2. 各路线 scope winner 的跨路线对比图。
    """
    rows = _read_rows(csv_path)
    scope_alias_pairs = _scope_alias_map(
        rows=rows,
        route_scopes=route_scopes,
        route_names=route_names,
    )
    if len(scope_alias_pairs) == 0:
        raise ValueError(f"No route scopes resolved from {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    route_paths: list[Path] = []
    for scope, alias in scope_alias_pairs:
        route_rows = _clone_rows_for_route_scope(rows, scope=scope)
        if len(route_rows) == 0:
            raise ValueError(f"No rows selected for route scope: {scope}")
        fig, _ = build_paper_ablation_summary_figure(rows=route_rows, mode=mode)
        out_stem = out_dir / f"route_{_slugify_name(alias)}_module_compare_{mode}"
        save_figure(fig, out_stem, fmt=fmt)
        plt.close(fig)
        route_paths.append(out_stem.with_suffix(".png"))

    winner_rows = _clone_rows_for_route_winners(rows, scope_alias_pairs=scope_alias_pairs)
    if len(winner_rows) == 0:
        raise ValueError("No scope winners found for cross-route comparison")
    fig, _ = build_paper_ablation_summary_figure(rows=winner_rows, mode=mode)
    winner_out_stem = out_dir / f"route_winner_compare_{mode}"
    save_figure(fig, winner_out_stem, fmt=fmt)
    plt.close(fig)

    return RouteComparisonSuitePaths(
        route_plot_paths=tuple(route_paths),
        winner_compare_path=winner_out_stem.with_suffix(".png"),
    )


def main() -> int:
    """论文用消融/路线汇总图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.eval.plot_paper_ablation_summary")
    ap.add_argument("--csv", type=str, required=True, help="summary.csv / ranking.csv / final_selection.csv path")
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=sorted(_METRIC_MODES.keys()),
        help="which metric bundle to draw",
    )
    ap.add_argument("--out_dir", type=str, default=None, help="output directory; default uses csv parent / plots")
    ap.add_argument("--label", type=str, nargs="*", default=None, help="optional label filter")
    ap.add_argument("--name", type=str, nargs="*", default=None, help="optional name filter")
    ap.add_argument("--scope", type=str, nargs="*", default=None, help="optional selection_scope filter")
    ap.add_argument("--winner_only", action="store_true", help="only keep final_selection scope winners")
    ap.add_argument(
        "--route-suite",
        action="store_true",
        help="export one within-route plot per selection_scope plus one cross-route winner compare plot",
    )
    ap.add_argument("--route-name", type=str, nargs="*", default=None, help="optional display names for route scopes")
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (csv_path.parent / "plots").resolve()
    if bool(args.route_suite):
        suite = plot_route_comparison_suite(
            csv_path=csv_path,
            out_dir=out_dir,
            mode=str(args.mode),
            route_scopes=args.scope,
            route_names=args.route_name,
            fmt=str(args.fmt),
        )
        print(f"[VIZ] wrote route plots to: {[str(p) for p in suite.route_plot_paths]}")
        print(f"[VIZ] wrote cross-route winner compare to: {suite.winner_compare_path}")
    else:
        out_path = plot_paper_ablation_summary(
            csv_path=csv_path,
            out_dir=out_dir,
            mode=str(args.mode),
            include_labels=args.label,
            include_names=args.name,
            include_scopes=args.scope,
            winner_only=bool(args.winner_only),
            fmt=str(args.fmt),
        )
        print(f"[VIZ] wrote paper ablation summary to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
