"""
模块名称：评估指标总览面板

模块职责：
从单个评估目录读取 `metrics.yaml`，
把 global / final-step / tail / growth / bias 等多类指标
汇总成一张多子图 dashboard。

主要功能：
1. 读取 `metrics.yaml` 中的 dense/masked 全局误差。
2. 绘制 final-step 组级 RMSE 柱状图。
3. 绘制 tail error 与 rollout growth 指标对比图。
4. 绘制 dense/masked 最坏 bias 摘要图。

数据流：
eval_dir/metrics.yaml
    ↓
dashboard figure
    ↓
plots/metrics_dashboard.png|pdf

依赖模块：
- matplotlib
- yaml
- uwnav_dynamics.viz.style.sci_style

备注：
- 本图是“多指标总览”，不替代现有 horizon / rollout / residual 细节图。
- 如果缺少 masked 指标，本脚本会自动退化为 dense-only 可视化。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uwnav_dynamics.viz.style.sci_style import apply_axes_style, get_figure_size, save_figure, setup_mpl


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


def build_metrics_dashboard_figure(metrics: dict[str, Any]) -> tuple[plt.Figure, np.ndarray]:
    """构建 2x2 评估指标总览图。"""
    setup_mpl()
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("wide"))
    ax_global, ax_groups, ax_tail, ax_bias = axes.ravel()

    dense_color = "#2A9D8F"
    masked_color = "#C8553D"
    accent = "#264653"

    global_labels = ["RMSE", "MAE"]
    dense_global = [
        float(metrics.get("rmse_global", np.nan)),
        float(metrics.get("mae_global", np.nan)),
    ]
    masked_global = [
        float(metrics.get("rmse_global_masked", np.nan)),
        float(metrics.get("mae_global_masked", np.nan)),
    ]
    x = np.arange(len(global_labels))
    width = 0.34
    ax_global.bar(x - width / 2, dense_global, width=width, color=dense_color, label="Dense")
    ax_global.bar(x + width / 2, masked_global, width=width, color=masked_color, label="Masked")
    ax_global.set_xticks(x, global_labels)
    ax_global.set_ylabel("Error")
    ax_global.set_title("Global Metrics")
    apply_axes_style(ax_global, grid=False)
    ax_global.legend(frameon=False, loc="upper right")

    group_labels = ["Acc", "Gyro", "Vel"]
    dense_group = [
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "final_step", "group_rmse", "acc")),
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "final_step", "group_rmse", "gyro")),
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "final_step", "group_rmse", "vel")),
    ]
    masked_group = [
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "final_step", "group_rmse", "acc")),
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "final_step", "group_rmse", "gyro")),
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "final_step", "group_rmse", "vel")),
    ]
    gx = np.arange(len(group_labels))
    ax_groups.bar(gx - width / 2, dense_group, width=width, color=dense_color)
    ax_groups.bar(gx + width / 2, masked_group, width=width, color=masked_color)
    ax_groups.set_xticks(gx, group_labels)
    ax_groups.set_ylabel("Final-step RMSE")
    ax_groups.set_title("Group Final-Step RMSE")
    apply_axes_style(ax_groups, grid=False)

    tail_labels = ["P95", "P99", "Growth"]
    dense_tail = [
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "tail_error", "abs_p95_global")),
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "tail_error", "abs_p99_global")),
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "rollout_growth", "rmse_last_over_first")),
    ]
    masked_tail = [
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "tail_error", "abs_p95_global")),
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "tail_error", "abs_p99_global")),
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "rollout_growth", "rmse_last_over_first")),
    ]
    tx = np.arange(len(tail_labels))
    ax_tail.plot(tx, dense_tail, marker="o", color=dense_color, linewidth=1.8, label="Dense")
    ax_tail.plot(tx, masked_tail, marker="o", color=masked_color, linewidth=1.8, label="Masked")
    ax_tail.set_xticks(tx, tail_labels)
    ax_tail.set_ylabel("Value")
    ax_tail.set_title("Tail And Growth")
    apply_axes_style(ax_tail, grid=False)
    ax_tail.legend(frameon=False, loc="upper left")

    bias_labels = [
        str(_nested_get(metrics, "control_readiness", "physical", "dense", "bias", "worst_component", default="dense")),
        str(_nested_get(metrics, "control_readiness", "physical", "masked", "bias", "worst_component", default="masked")),
    ]
    bias_vals = [
        float(_nested_get(metrics, "control_readiness", "physical", "dense", "bias", "worst_abs_bias")),
        float(_nested_get(metrics, "control_readiness", "physical", "masked", "bias", "worst_abs_bias")),
    ]
    bx = np.arange(len(bias_labels))
    ax_bias.bar(bx, bias_vals, color=[accent, masked_color], width=0.55)
    ax_bias.set_xticks(bx, bias_labels, rotation=15)
    ax_bias.set_ylabel("Abs Bias")
    ax_bias.set_title("Worst Bias Component")
    apply_axes_style(ax_bias, grid=False)

    fig.tight_layout()
    return fig, axes


def plot_metrics_dashboard(eval_dir: Path, out_dir: Path, *, fmt: str) -> None:
    """读取评估目录并导出 dashboard 图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_yaml(eval_dir / "metrics.yaml")
    fig, _axes = build_metrics_dashboard_figure(metrics)
    save_figure(fig, out_dir / "metrics_dashboard", fmt=fmt)
    plt.close(fig)


def main() -> int:
    """评估指标 dashboard 命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.eval.plot_metrics_dashboard")
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    plot_metrics_dashboard(Path(args.eval_dir), Path(args.out_dir), fmt=str(args.fmt))
    print(f"[VIZ] wrote metrics dashboard to: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
