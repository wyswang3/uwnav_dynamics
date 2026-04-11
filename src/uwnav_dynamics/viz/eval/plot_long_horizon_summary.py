"""
模块名称：长期拟合摘要图

模块职责：
从 `metrics.yaml["long_horizon_fit"]` 读取长期 rollout 摘要指标，
生成可直接用于论文与技术汇报的长期拟合总览图。

主要功能：
1. 展示 dense / masked 的 horizon AUC 指标。
2. 展示 late-horizon 区间的 RMSE / MAE 均值。
3. 展示随 horizon 增长的误差斜率。
4. 展示 final-step 误差，帮助与全局均值区分。
5. 支持多评估目录长期拟合 compare 图，比较不同方案的长时 rollout 质量。

数据流：
eval_dir/metrics.yaml
    ↓
long_horizon_fit 读取
    ↓
2×2 summary figure
    ↓
plots/long_horizon_fit_summary.png|pdf

依赖模块：
- matplotlib
- numpy
- yaml
- uwnav_dynamics.viz.style.sci_style
- uwnav_dynamics.viz.eval.plot_horizon_metrics

备注：
- 本图只消费评估阶段已写出的摘要字段，不重新回放模型或重算数值评估。
- 语义上属于“长期拟合离线审计图”，不替代闭环控制验证。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uwnav_dynamics.viz.eval.plot_horizon_metrics import _load_csv_hd, _metric_csv_name
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
class LongHorizonCompareCfg:
    """长期拟合 compare 图配置。"""
    fmt: str = "png"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"metrics yaml must be a mapping: {path}")
    return data


def _nested_get(mapping: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_long_horizon_summary_figure(metrics: dict[str, Any]) -> tuple[plt.Figure, np.ndarray]:
    """构建长期拟合摘要图。"""
    setup_mpl()
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("dashboard_2x2_compact"))
    ax_auc, ax_late, ax_slope, ax_final = axes.ravel()

    dense = _nested_get(metrics, "long_horizon_fit", "physical", "dense", default={})
    masked = _nested_get(metrics, "long_horizon_fit", "physical", "masked", default={})
    late_start = int(_nested_get(dense, "late_horizon_start_step", default=1))

    dense_color = "#2A9D8F"
    masked_color = "#C8553D"
    width = 0.34

    auc_labels = ["RMSE AUC", "MAE AUC"]
    dense_auc = [
        float(_nested_get(dense, "rmse_auc_global")),
        float(_nested_get(dense, "mae_auc_global")),
    ]
    masked_auc = [
        float(_nested_get(masked, "rmse_auc_global")),
        float(_nested_get(masked, "mae_auc_global")),
    ]
    x_auc = np.arange(len(auc_labels))
    ax_auc.bar(x_auc - width / 2, dense_auc, width=width, color=dense_color, label="Dense")
    ax_auc.bar(x_auc + width / 2, masked_auc, width=width, color=masked_color, label="Masked")
    ax_auc.set_xticks(x_auc, auc_labels)
    ax_auc.set_ylabel("Mean error")
    ax_auc.set_title("Horizon AUC")
    apply_axes_style(ax_auc, grid=False)
    ax_auc.legend(frameon=False, loc="upper right")

    late_labels = ["Late RMSE", "Late MAE"]
    dense_late = [
        float(_nested_get(dense, "late_horizon_rmse_global_mean")),
        float(_nested_get(dense, "late_horizon_mae_global_mean")),
    ]
    masked_late = [
        float(_nested_get(masked, "late_horizon_rmse_global_mean")),
        float(_nested_get(masked, "late_horizon_mae_global_mean")),
    ]
    x_late = np.arange(len(late_labels))
    ax_late.bar(x_late - width / 2, dense_late, width=width, color=dense_color)
    ax_late.bar(x_late + width / 2, masked_late, width=width, color=masked_color)
    ax_late.set_xticks(x_late, late_labels)
    ax_late.set_ylabel("Late-window error")
    ax_late.set_title(f"Late Horizon Mean (step >= {late_start})")
    apply_axes_style(ax_late, grid=False)

    slope_labels = ["RMSE slope", "MAE slope"]
    dense_slope = [
        float(_nested_get(dense, "rmse_step_slope")),
        float(_nested_get(dense, "mae_step_slope")),
    ]
    masked_slope = [
        float(_nested_get(masked, "rmse_step_slope")),
        float(_nested_get(masked, "mae_step_slope")),
    ]
    x_slope = np.arange(len(slope_labels))
    ax_slope.axhline(0.0, color="#B8C4CF", linewidth=1.0, linestyle="--", zorder=1)
    ax_slope.bar(x_slope - width / 2, dense_slope, width=width, color=dense_color)
    ax_slope.bar(x_slope + width / 2, masked_slope, width=width, color=masked_color)
    ax_slope.set_xticks(x_slope, slope_labels)
    ax_slope.set_ylabel("Error / step")
    ax_slope.set_title("Horizon Error Slope")
    apply_axes_style(ax_slope, grid=False)

    final_labels = ["Final RMSE", "Final MAE"]
    dense_final = [
        float(_nested_get(dense, "final_step_rmse_global")),
        float(_nested_get(dense, "final_step_mae_global")),
    ]
    masked_final = [
        float(_nested_get(masked, "final_step_rmse_global")),
        float(_nested_get(masked, "final_step_mae_global")),
    ]
    x_final = np.arange(len(final_labels))
    ax_final.bar(x_final - width / 2, dense_final, width=width, color=dense_color)
    ax_final.bar(x_final + width / 2, masked_final, width=width, color=masked_color)
    ax_final.set_xticks(x_final, final_labels)
    ax_final.set_ylabel("Final-step error")
    ax_final.set_title("Final-step Metrics")
    apply_axes_style(ax_final, grid=False)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.13, wspace=0.28, hspace=0.34)
    return fig, axes


def plot_long_horizon_summary(eval_dir: Path, out_dir: Path, *, fmt: str) -> None:
    """读取评估目录并导出长期拟合摘要图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_yaml(eval_dir / "metrics.yaml")
    fig, _axes = build_long_horizon_summary_figure(metrics)
    save_figure(fig, out_dir / "long_horizon_fit_summary", fmt=fmt)
    plt.close(fig)


def main() -> int:
    """长期拟合摘要图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.eval.plot_long_horizon_summary")
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    plot_long_horizon_summary(Path(args.eval_dir), Path(args.out_dir), fmt=str(args.fmt))
    print(f"[VIZ] wrote long-horizon summary to: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
