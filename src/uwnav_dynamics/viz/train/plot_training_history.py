"""
模块名称：训练历史可视化

模块职责：
从训练阶段写出的 `train_history.csv` 与 `train_summary.yaml` 读取 epoch 级历史，
生成可直接用于论文与技术汇报的训练过程图。

主要功能：
1. 绘制 train/val loss 曲线，标记 best epoch。
2. 绘制 monitor 曲线，区分 `val_loss` 与 `val_transition_score` 等验证监控量。
3. 绘制 z-space `RMSE / MAE` 与学习率变化曲线。
4. 导出单图与 2×2 dashboard，供训练主流程自动生成 artifact。

数据流：
train_history.csv + optional train_summary.yaml
    ↓
epoch history parsing
    ↓
matplotlib figure
    ↓
train_plots/training_dashboard.png|pdf

依赖模块：
- csv
- yaml
- matplotlib
- numpy
- uwnav_dynamics.viz.style.sci_style

备注：
- 本模块只可视化已落盘的训练历史，不回推或重算训练损失。
- 若 `train_history.csv` 中部分 epoch 未执行验证，会在图中自动跳过对应点。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uwnav_dynamics.viz.style.sci_style import (
    add_axes_legend,
    apply_axes_style,
    get_figure_size,
    save_figure,
    setup_mpl,
)


@dataclass(frozen=True)
class TrainingPlotConfig:
    """训练历史图的导出格式配置。"""
    fmt: str = "png"


def _load_history_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"train summary yaml must be a mapping: {path}")
    return data


def _as_float_or_nan(value: str | Any) -> float:
    if value in ("", None):
        return float("nan")
    if isinstance(value, str) and value.strip().lower() == "skip":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: str | Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _history_series(rows: Sequence[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_as_float_or_nan(row.get(key, "")) for row in rows], dtype=np.float64)


def _history_epochs(rows: Sequence[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)


def _best_epoch(rows: Sequence[dict[str, str]], summary: dict[str, Any]) -> int | None:
    best_from_summary = summary.get("best_epoch")
    if isinstance(best_from_summary, int) and best_from_summary > 0:
        return int(best_from_summary)
    for row in rows:
        if _as_bool(row.get("is_best", "")):
            return int(row["epoch"])
    return None


def _marker_epoch(ax: plt.Axes, best_epoch: int | None, series: np.ndarray, epochs: np.ndarray) -> None:
    if best_epoch is None:
        return
    ax.axvline(float(best_epoch), color="#B8C4CF", linestyle="--", linewidth=1.0, zorder=1)
    idx = np.where(epochs == int(best_epoch))[0]
    if idx.size == 0:
        return
    val = float(series[idx[0]])
    if np.isfinite(val):
        ax.scatter([best_epoch], [val], s=24.0, color="#C8553D", zorder=6)


def build_training_dashboard_figure(
    history_rows: Sequence[dict[str, str]],
    *,
    summary: dict[str, Any] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """构建训练过程的 2×2 dashboard。"""
    setup_mpl()
    summary = summary or {}
    epochs = _history_epochs(history_rows)
    train_loss = _history_series(history_rows, "train_loss")
    val_loss = _history_series(history_rows, "val_loss")
    monitor = _history_series(history_rows, "monitor_value")
    val_rmse = _history_series(history_rows, "val_rmse_global_zspace")
    val_mae = _history_series(history_rows, "val_mae_global_zspace")
    lr = _history_series(history_rows, "lr")
    monitor_name = str(history_rows[-1].get("monitor_name", "monitor")) if history_rows else "monitor"
    best_epoch = _best_epoch(history_rows, summary)

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("dashboard_2x2_compact"))
    ax_loss, ax_monitor, ax_metric, ax_lr = axes.ravel()

    ax_loss.plot(epochs, train_loss, color="#2C7FB8", linewidth=1.8, label="Train")
    if np.any(np.isfinite(val_loss)):
        ax_loss.plot(epochs, val_loss, color="#E76F51", linewidth=1.6, linestyle="--", label="Val")
        _marker_epoch(ax_loss, best_epoch, val_loss, epochs)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / Val Loss")
    apply_axes_style(ax_loss, grid=False)
    add_axes_legend(ax_loss, loc="upper right")

    ax_monitor.plot(epochs, monitor, color="#1B9E77", linewidth=1.8, label=monitor_name)
    _marker_epoch(ax_monitor, best_epoch, monitor, epochs)
    ax_monitor.set_xlabel("Epoch")
    ax_monitor.set_ylabel("Monitor")
    ax_monitor.set_title("Validation Monitor")
    apply_axes_style(ax_monitor, grid=False)
    add_axes_legend(ax_monitor, loc="upper right")

    if np.any(np.isfinite(val_rmse)):
        ax_metric.plot(epochs, val_rmse, color="#2C7FB8", linewidth=1.7, label="Val RMSE (z)")
        _marker_epoch(ax_metric, best_epoch, val_rmse, epochs)
    if np.any(np.isfinite(val_mae)):
        ax_metric.plot(epochs, val_mae, color="#E76F51", linewidth=1.5, linestyle="--", label="Val MAE (z)")
        _marker_epoch(ax_metric, best_epoch, val_mae, epochs)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("z-space error")
    ax_metric.set_title("Validation Error")
    apply_axes_style(ax_metric, grid=False)
    add_axes_legend(ax_metric, loc="upper right")

    ax_lr.plot(epochs, lr, color="#5C6BC0", linewidth=1.7)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_title("Learning Rate")
    if np.all(lr[np.isfinite(lr)] > 0.0):
        ax_lr.set_yscale("log")
    apply_axes_style(ax_lr, grid=False)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.12, wspace=0.28, hspace=0.32)
    return fig, axes


def _build_single_curve(
    *,
    epochs: np.ndarray,
    series_list: Sequence[tuple[str, np.ndarray, str, str]],
    title: str,
    ylabel: str,
    best_epoch: int | None,
    marker_series_index: int = 0,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size("single"))
    for label, series, color, linestyle in series_list:
        if not np.any(np.isfinite(series)):
            continue
        ax.plot(epochs, series, label=label, color=color, linewidth=1.7, linestyle=linestyle)
    if len(series_list) > 0:
        marker_idx = max(0, min(int(marker_series_index), len(series_list) - 1))
        _marker_epoch(ax, best_epoch, series_list[marker_idx][1], epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_axes_style(ax, grid=False)
    add_axes_legend(ax, loc="upper right")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.15)
    return fig


def plot_training_artifacts(
    *,
    history_csv: Path,
    out_dir: Path,
    summary_yaml: Path | None = None,
    cfg: TrainingPlotConfig | None = None,
) -> None:
    """读取训练历史并导出训练图表 artifact。"""
    cfg = cfg or TrainingPlotConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_history_rows(history_csv)
    if len(rows) == 0:
        raise ValueError(f"train history is empty: {history_csv}")
    summary = _load_summary(summary_yaml)
    epochs = _history_epochs(rows)
    train_loss = _history_series(rows, "train_loss")
    val_loss = _history_series(rows, "val_loss")
    monitor = _history_series(rows, "monitor_value")
    val_rmse = _history_series(rows, "val_rmse_global_zspace")
    val_mae = _history_series(rows, "val_mae_global_zspace")
    lr = _history_series(rows, "lr")
    monitor_name = str(rows[-1].get("monitor_name", "monitor")) if rows else "monitor"
    best_epoch = _best_epoch(rows, summary)

    fig_dashboard, _ = build_training_dashboard_figure(rows, summary=summary)
    save_figure(fig_dashboard, out_dir / "training_dashboard", fmt=cfg.fmt)
    plt.close(fig_dashboard)

    fig_loss = _build_single_curve(
        epochs=epochs,
        series_list=(
            ("Train", train_loss, "#2C7FB8", "-"),
            ("Val", val_loss, "#E76F51", "--"),
        ),
        title="Training Loss Curve",
        ylabel="Loss",
        best_epoch=best_epoch,
        marker_series_index=1,
    )
    save_figure(fig_loss, out_dir / "training_loss_curve", fmt=cfg.fmt)
    plt.close(fig_loss)

    fig_monitor = _build_single_curve(
        epochs=epochs,
        series_list=((monitor_name, monitor, "#1B9E77", "-"),),
        title="Validation Monitor Curve",
        ylabel="Monitor",
        best_epoch=best_epoch,
    )
    save_figure(fig_monitor, out_dir / "validation_monitor_curve", fmt=cfg.fmt)
    plt.close(fig_monitor)

    fig_metric = _build_single_curve(
        epochs=epochs,
        series_list=(
            ("Val RMSE (z)", val_rmse, "#2C7FB8", "-"),
            ("Val MAE (z)", val_mae, "#E76F51", "--"),
        ),
        title="Validation Error Curve",
        ylabel="z-space error",
        best_epoch=best_epoch,
        marker_series_index=0,
    )
    save_figure(fig_metric, out_dir / "validation_error_curve", fmt=cfg.fmt)
    plt.close(fig_metric)

    fig_lr, ax_lr = plt.subplots(1, 1, figsize=get_figure_size("single"))
    ax_lr.plot(epochs, lr, color="#5C6BC0", linewidth=1.7)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_title("Learning Rate Curve")
    finite_lr = lr[np.isfinite(lr)]
    if finite_lr.size > 0 and np.all(finite_lr > 0.0):
        ax_lr.set_yscale("log")
    apply_axes_style(ax_lr, grid=False)
    fig_lr.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.15)
    save_figure(fig_lr, out_dir / "learning_rate_curve", fmt=cfg.fmt)
    plt.close(fig_lr)


def main() -> int:
    """训练历史绘图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.train.plot_training_history")
    ap.add_argument("--run_dir", type=str, required=True, help="run dir containing train_history.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="default: <run_dir>/train_plots")
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir / "train_plots")
    plot_training_artifacts(
        history_csv=run_dir / "train_history.csv",
        out_dir=out_dir,
        summary_yaml=run_dir / "train_summary.yaml",
        cfg=TrainingPlotConfig(fmt=str(args.fmt)),
    )
    print(f"[VIZ][TRAIN] wrote training plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
