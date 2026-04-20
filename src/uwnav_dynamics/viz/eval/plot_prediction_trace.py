"""
模块名称：长时序预测诊断绘图

模块职责：
从评估阶段落盘的 `pred_trace.npz` 中读取连续或近连续的预测时序，
生成按物理量分组的 3 子窗共享 x 轴长窗口预测-目标对比图。

主要功能：
1. 读取 `pred_trace.npz` 中的 `y_hat / y_true / target_mask / t_s`。
2. 默认分别输出 Acc / Gyro / Vel 三张图，每张图包含 X/Y/Z 三个子窗。
3. 可选输出 Acc / Gyro / Vel 三组范数图。
4. 默认绘制约 50s 长时序窗口，避免单个 0.1s horizon 图无法反映实际运行状态。
5. 在目标无效位置用轻量 marker 标注 masked-out 样本。
6. 兼容旧评估目录：缺少 `pred_trace.npz` 时，从 `pred_samples.npz + pred_context.npz`
   按 sample index 重建一个尽可能接近 50s 的诊断窗口。

数据流：
eval_dir/pred_trace.npz + metrics.yaml(layout.semantic)
    ↓
group norm trace extraction
    ↓
3×1 shared-x matplotlib figures
    ↓
plots/prediction_trace_acc_axes.png|pdf
plots/prediction_trace_gyro_axes.png|pdf
plots/prediction_trace_vel_axes.png|pdf
以及可选 plots/prediction_trace_group_norm.png|pdf

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.models.utils.semantic_output_layout
- uwnav_dynamics.viz.style.sci_style

备注：
- 本图面向长时序诊断，不替代 horizon 指标图。
- 新 artifact 的 `pred_trace.npz` 是优先数据源；旧 artifact fallback 仅用于重画历史结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from uwnav_dynamics.models.utils.semantic_output_layout import (
    SemanticOutputLayout,
    load_semantic_layout_from_metrics_path,
)
from uwnav_dynamics.viz.style.sci_style import (
    add_figure_legend,
    align_ylabels,
    apply_axes_style,
    apply_shared_xlabels,
    get_figure_size,
    get_group_styles,
    get_observed_pred_styles,
    get_xyz_styles,
    plot_visible_series,
    save_figure,
    setup_mpl,
)


_GROUP_DISPLAY_NAMES = {
    "acc": "Acc",
    "gyro": "Gyro",
    "vel": "Vel",
}
_GROUP_TITLES = {
    "acc": "Acceleration Prediction Trace",
    "gyro": "Angular Velocity Prediction Trace",
    "vel": "Velocity Prediction Trace",
}
_GROUP_YLABELS = {
    "acc": r"||Acc|| (m/s^2)",
    "gyro": r"||Gyro|| (rad/s)",
    "vel": r"||Vel|| (m/s)",
}
_GROUP_FILE_STEMS = {
    "acc": "prediction_trace_acc_axes",
    "gyro": "prediction_trace_gyro_axes",
    "vel": "prediction_trace_vel_axes",
}
_AXIS_KEYS = ("x", "y", "z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1))


def _select_dense_window(sample_index: np.ndarray, *, seconds: float, dt_s: float) -> np.ndarray:
    sample_index = np.asarray(sample_index, dtype=np.int64)
    if sample_index.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(sample_index, kind="stable")
    sorted_index = sample_index[order]
    window_steps = max(1, int(round(max(float(seconds), float(dt_s)) / max(float(dt_s), np.finfo(float).eps))))
    best_start = 0
    best_stop = 1
    stop = 0
    for start in range(sorted_index.shape[0]):
        if stop < start:
            stop = start
        max_index = int(sorted_index[start]) + window_steps - 1
        while stop < sorted_index.shape[0] and int(sorted_index[stop]) <= max_index:
            stop += 1
        if (stop - start) > (best_stop - best_start):
            best_start = start
            best_stop = stop
    return np.asarray(order[best_start:best_stop], dtype=np.int64)


def _load_trace_npz(
    eval_dir: Path,
    *,
    fallback_seconds: float,
    fallback_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace_npz = eval_dir / "pred_trace.npz"
    if not trace_npz.exists():
        return _load_legacy_sample_trace(
            eval_dir,
            fallback_seconds=float(fallback_seconds),
            fallback_dt_s=float(fallback_dt_s),
        )
    with np.load(trace_npz, allow_pickle=False) as z:
        required = ("y_hat", "y_true", "target_mask", "t_s")
        missing = [key for key in required if key not in z]
        if missing:
            raise ValueError(f"{trace_npz} missing required keys: {missing}")
        y_hat = np.asarray(z["y_hat"], dtype=np.float32)
        y_true = np.asarray(z["y_true"], dtype=np.float32)
        target_mask = np.asarray(z["target_mask"], dtype=bool)
        t_s = np.asarray(z["t_s"], dtype=np.float64)
    if y_hat.ndim != 2 or y_true.ndim != 2 or target_mask.ndim != 2:
        raise ValueError(f"Expect trace arrays with shape (T,D), got {y_hat.shape}, {y_true.shape}, {target_mask.shape}")
    if y_hat.shape != y_true.shape or y_hat.shape != target_mask.shape:
        raise ValueError(f"trace shape mismatch: {y_hat.shape}, {y_true.shape}, {target_mask.shape}")
    if t_s.shape != (y_hat.shape[0],):
        raise ValueError(f"t_s shape mismatch: {t_s.shape} vs T={y_hat.shape[0]}")
    return y_hat, y_true, target_mask, t_s


def _load_legacy_sample_trace(
    eval_dir: Path,
    *,
    fallback_seconds: float,
    fallback_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_npz = eval_dir / "pred_samples.npz"
    context_npz = eval_dir / "pred_context.npz"
    if not pred_npz.exists() or not context_npz.exists():
        raise FileNotFoundError(
            f"Missing {eval_dir / 'pred_trace.npz'} and cannot fallback because "
            f"{pred_npz.name} or {context_npz.name} is missing."
        )
    with np.load(pred_npz, allow_pickle=False) as z:
        if "y_hat" not in z or "y_true" not in z:
            raise ValueError(f"{pred_npz} must contain y_hat / y_true")
        y_hat_all = np.asarray(z["y_hat"], dtype=np.float32)
        y_true_all = np.asarray(z["y_true"], dtype=np.float32)
    with np.load(context_npz, allow_pickle=False) as z:
        if "target_mask" not in z or "sample_index" not in z:
            raise ValueError(f"{context_npz} must contain target_mask / sample_index for legacy trace fallback")
        target_mask_all = np.asarray(z["target_mask"], dtype=bool)
        sample_index = np.asarray(z["sample_index"], dtype=np.int64)
    if y_hat_all.ndim != 3 or y_hat_all.shape != y_true_all.shape or y_hat_all.shape != target_mask_all.shape:
        raise ValueError(
            f"legacy sample trace expects (N,H,D), got {y_hat_all.shape}, {y_true_all.shape}, {target_mask_all.shape}"
        )
    slots = _select_dense_window(sample_index, seconds=float(fallback_seconds), dt_s=float(fallback_dt_s))
    horizon_index = int(y_hat_all.shape[1] - 1)
    selected_index = sample_index[slots]
    if selected_index.shape[0] > 0:
        t_s = (selected_index.astype(np.float64) - float(selected_index[0])) * float(fallback_dt_s)
    else:
        t_s = np.zeros((0,), dtype=np.float64)
    return (
        y_hat_all[slots, horizon_index, :],
        y_true_all[slots, horizon_index, :],
        target_mask_all[slots, horizon_index, :],
        t_s,
    )


def _group_specs(semantic_layout: SemanticOutputLayout) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (group_key, semantic_layout.group_indices[group_key], _GROUP_YLABELS[group_key])
        for group_key in ("acc", "gyro", "vel")
    )


def build_prediction_trace_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    target_mask: np.ndarray,
    t_s: np.ndarray,
    semantic_layout: SemanticOutputLayout,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """构建 Acc/Gyro/Vel 三组长时序预测-目标对比图。"""
    setup_mpl()
    observed_style = get_observed_pred_styles()["observed"]
    pred_style = get_observed_pred_styles()["pred"]
    group_styles = get_group_styles()
    group_specs = _group_specs(semantic_layout)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("rollout_3row_compact"))
    masked_legend_needed = False

    for row_idx, (ax, (group_key, indices, ylabel)) in enumerate(zip(axes, group_specs)):
        idx = list(indices)
        obs = _norm(y_true[:, idx])
        pred = _norm(y_hat[:, idx])
        group_valid = np.all(target_mask[:, idx], axis=1)
        group_color = group_styles[_GROUP_DISPLAY_NAMES[group_key]].color

        plot_visible_series(
            ax,
            t_s,
            obs,
            label="Target" if row_idx == 0 else None,
            color=observed_style.color,
            linestyle=observed_style.linestyle,
            linewidth=observed_style.linewidth,
            alpha=observed_style.alpha,
            zorder=observed_style.zorder,
        )
        plot_visible_series(
            ax,
            t_s,
            pred,
            label="Pred" if row_idx == 0 else None,
            color=group_color,
            linestyle=pred_style.linestyle,
            linewidth=pred_style.linewidth,
            alpha=pred_style.alpha,
            zorder=pred_style.zorder,
        )
        invalid = ~group_valid
        if np.any(invalid):
            masked_legend_needed = True
            ax.scatter(
                t_s[invalid],
                obs[invalid],
                label="Masked",
                s=9.0,
                facecolors="white",
                edgecolors="#8C9199",
                linewidths=0.55,
                zorder=5,
            )
        ax.set_ylabel(ylabel)
        apply_axes_style(ax, grid=False)

    if masked_legend_needed and "Masked" not in axes[0].get_legend_handles_labels()[1]:
        axes[0].scatter(
            [],
            [],
            label="Masked",
            s=9.0,
            facecolors="white",
            edgecolors="#8C9199",
            linewidths=0.55,
            zorder=5,
        )

    apply_shared_xlabels(list(axes), "Trace time (s)")
    align_ylabels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Group Norm Prediction Trace", x=0.14, ha="left", y=0.985)
    add_figure_legend(fig, handles, labels, ncol=3 if masked_legend_needed else 2, y=0.895)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.12, hspace=0.08)
    return fig, (axes[0], axes[1], axes[2])


def build_prediction_trace_group_axes_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    target_mask: np.ndarray,
    t_s: np.ndarray,
    semantic_layout: SemanticOutputLayout,
    group_key: str,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """构建单个物理量组的 X/Y/Z 三轴长时序预测-目标对比图。"""
    setup_mpl()
    if group_key not in semantic_layout.group_indices:
        raise KeyError(f"Unknown group key: {group_key}")

    observed_style = get_observed_pred_styles()["observed"]
    pred_style = get_observed_pred_styles()["pred"]
    xyz_styles = get_xyz_styles()
    indices = tuple(semantic_layout.group_indices[group_key])
    if len(indices) != 3:
        raise ValueError(f"{group_key} group must contain 3 axes, got {indices}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("rollout_3row_compact"))
    masked_legend_needed = False

    for row_idx, (ax, axis_key, component_idx) in enumerate(zip(axes, _AXIS_KEYS, indices)):
        component_label = str(semantic_layout.component_labels[component_idx])
        axis_style = xyz_styles.get(axis_key, xyz_styles["x"])
        valid = target_mask[:, component_idx]

        plot_visible_series(
            ax,
            t_s,
            y_true[:, component_idx],
            label="Target" if row_idx == 0 else None,
            color=observed_style.color,
            linestyle=observed_style.linestyle,
            linewidth=observed_style.linewidth,
            alpha=observed_style.alpha,
            zorder=observed_style.zorder,
        )
        plot_visible_series(
            ax,
            t_s,
            y_hat[:, component_idx],
            label="Pred" if row_idx == 0 else None,
            color=axis_style.color,
            linestyle=pred_style.linestyle,
            linewidth=pred_style.linewidth,
            alpha=pred_style.alpha,
            zorder=pred_style.zorder,
        )
        invalid = ~valid
        if np.any(invalid):
            masked_legend_needed = True
            ax.scatter(
                t_s[invalid],
                y_true[invalid, component_idx],
                label="Masked",
                s=9.0,
                facecolors="white",
                edgecolors="#8C9199",
                linewidths=0.55,
                zorder=5,
            )
        ax.set_ylabel(component_label)
        apply_axes_style(ax, grid=False)

    if masked_legend_needed and "Masked" not in axes[0].get_legend_handles_labels()[1]:
        axes[0].scatter(
            [],
            [],
            label="Masked",
            s=9.0,
            facecolors="white",
            edgecolors="#8C9199",
            linewidths=0.55,
            zorder=5,
        )

    apply_shared_xlabels(list(axes), "Trace time (s)")
    align_ylabels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(_GROUP_TITLES.get(group_key, f"{group_key} Prediction Trace"), x=0.14, ha="left", y=0.985)
    add_figure_legend(fig, handles, labels, ncol=3 if masked_legend_needed else 2, y=0.895)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.12, hspace=0.08)
    return fig, (axes[0], axes[1], axes[2])


def plot_prediction_trace(
    eval_dir: Path,
    out_dir: Path,
    *,
    fmt: str = "png",
    fallback_seconds: float = 50.0,
    fallback_dt_s: float = 0.01,
    mode: str = "group_axes",
) -> None:
    """读取 `pred_trace.npz` 并导出长时序预测诊断图。"""
    _ensure_dir(out_dir)
    eval_dir = Path(eval_dir)
    y_hat, y_true, target_mask, t_s = _load_trace_npz(
        eval_dir,
        fallback_seconds=float(fallback_seconds),
        fallback_dt_s=float(fallback_dt_s),
    )
    semantic_layout = load_semantic_layout_from_metrics_path(eval_dir / "metrics.yaml", dout=y_hat.shape[-1])
    if mode in ("group_axes", "both"):
        for group_key, out_name in _GROUP_FILE_STEMS.items():
            fig, _ = build_prediction_trace_group_axes_figure(
                y_hat=y_hat,
                y_true=y_true,
                target_mask=target_mask,
                t_s=t_s,
                semantic_layout=semantic_layout,
                group_key=group_key,
            )
            save_figure(fig, out_dir / out_name, fmt=fmt)
            plt.close(fig)
    if mode in ("group_norm", "both"):
        fig, _ = build_prediction_trace_figure(
            y_hat=y_hat,
            y_true=y_true,
            target_mask=target_mask,
            t_s=t_s,
            semantic_layout=semantic_layout,
        )
        save_figure(fig, out_dir / "prediction_trace_group_norm", fmt=fmt)
        plt.close(fig)
    if mode not in ("group_axes", "group_norm", "both"):
        raise ValueError(f"Unknown prediction trace plot mode: {mode}")


def main() -> int:
    """长时序预测诊断图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.eval.plot_prediction_trace")
    ap.add_argument("--eval_dir", type=str, required=True, help="evaluation dir containing pred_trace.npz")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seconds", type=float, default=50.0, help="legacy pred_samples fallback window length")
    ap.add_argument("--dt", type=float, default=0.01, help="legacy pred_samples fallback sample period")
    ap.add_argument("--mode", type=str, default="group_axes", choices=["group_axes", "group_norm", "both"])
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    plot_prediction_trace(
        Path(args.eval_dir),
        Path(args.out_dir),
        fmt=str(args.fmt),
        fallback_seconds=float(args.seconds),
        fallback_dt_s=float(args.dt),
        mode=str(args.mode),
    )
    print(f"[VIZ] wrote prediction trace plot to: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
