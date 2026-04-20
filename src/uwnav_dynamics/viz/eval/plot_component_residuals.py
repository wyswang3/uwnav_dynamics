"""
模块名称：分量残差绘图

模块职责：
从评估阶段落盘的 `pred_samples.npz` 与可选 `pred_context.npz` 中读取分量级预测、
监督目标与监督有效性，生成适合系统辨识审查的残差图。

主要功能：
1. 对 9 个输出分量分别绘制 `prediction - target` 残差时序。
2. 在存在 `pred_context.npz["target_mask"]` 时，用轻量标记显示 masked-out 位置。
3. 复用仓库统一科研绘图 token，保持与 horizon / rollout 图一致的视觉语言，并使用紧凑九宫格布局。
4. 对单步 horizon 自动补 marker，避免 STEP 类残差图只有坐标轴而无可见点。
5. 支持 `group_norm` 三子窗模式，用于默认汇报图；component 九宫格保留为排障模式。

数据流：
pred_samples.npz + optional pred_context.npz
    ↓
component residual extraction
    ↓
3×3 figure
    ↓
plots/residual_component_000.png

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 当前残差定义为 `y_hat - y_true`。
- 若 `pred_context.npz` 不存在，则仍可出图，只是不会显示 masked-out 标记。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from uwnav_dynamics.models.utils.semantic_output_layout import (
    SemanticOutputLayout,
    canonical_semantic_output_layout,
    load_semantic_layout_from_metrics_path,
)
from uwnav_dynamics.viz.style.sci_style import (
    add_axes_legend,
    align_ylabels,
    apply_axes_style,
    apply_shared_xlabels,
    get_figure_size,
    get_group_styles,
    get_xyz_styles,
    plot_visible_series,
    save_figure,
    setup_mpl,
)


_COMPONENT_AXIS_META = {
    "acc_x": ("Acc X err (m/s^2)", "x"),
    "acc_y": ("Acc Y err (m/s^2)", "y"),
    "acc_z": ("Acc Z err (m/s^2)", "z"),
    "gyro_x": ("Gyro X err (rad/s)", "x"),
    "gyro_y": ("Gyro Y err (rad/s)", "y"),
    "gyro_z": ("Gyro Z err (rad/s)", "z"),
    "vel_x": ("Vel X err (m/s)", "x"),
    "vel_y": ("Vel Y err (m/s)", "y"),
    "vel_z": ("Vel Z err (m/s)", "z"),
}
_GROUP_DISPLAY_NAMES = {
    "acc": "Acc",
    "gyro": "Gyro",
    "vel": "Vel",
}
_GROUP_YLABELS = {
    "acc": r"||Acc err|| (m/s^2)",
    "gyro": r"||Gyro err|| (rad/s)",
    "vel": r"||Vel err|| (m/s)",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_pred_npz(pred_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(pred_npz, allow_pickle=False) as z:
        if "y_hat" not in z or "y_true" not in z:
            raise ValueError(f"{pred_npz} must contain y_hat / y_true")
        y_hat = np.asarray(z["y_hat"], dtype=np.float32)
        y_true = np.asarray(z["y_true"], dtype=np.float32)
    if y_hat.ndim != 3 or y_true.ndim != 3 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect y_hat/y_true with identical shape (N,H,D), got {y_hat.shape} and {y_true.shape}")
    return y_hat, y_true


def _load_target_mask(pred_npz: Path, *, shape: tuple[int, int, int]) -> np.ndarray | None:
    context_npz = pred_npz.parent / "pred_context.npz"
    if not context_npz.exists():
        return None
    with np.load(context_npz, allow_pickle=True) as z:
        if "target_mask" not in z:
            return None
        target_mask = np.asarray(z["target_mask"], dtype=bool)
    if target_mask.shape != shape:
        raise ValueError(f"pred_context target_mask shape mismatch: {target_mask.shape} vs {shape}")
    return target_mask


def _norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1))


def _group_specs(semantic_layout: SemanticOutputLayout) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (group_key, semantic_layout.group_indices[group_key], _GROUP_YLABELS[group_key])
        for group_key in ("acc", "gyro", "vel")
    )


def build_group_residual_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    dt_s: float,
    semantic_layout: SemanticOutputLayout | None = None,
    target_mask: np.ndarray | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """构建单个样本窗口的 Acc/Gyro/Vel 三组残差范数图。"""
    setup_mpl()

    if y_hat.ndim != 2 or y_true.ndim != 2 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect sample arrays with shape (H, D), got {y_hat.shape} and {y_true.shape}")
    if semantic_layout is None:
        semantic_layout = canonical_semantic_output_layout(y_hat.shape[-1])
    if target_mask is not None and target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: {target_mask.shape} vs {y_hat.shape}")

    H = y_hat.shape[0]
    t = np.arange(1, H + 1, dtype=float) * float(dt_s)
    residual = y_hat - y_true
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("rollout_3row_compact"))
    group_styles = get_group_styles()
    masked_legend_needed = False

    for row_idx, (ax, (group_key, indices, ylabel)) in enumerate(zip(axes, _group_specs(semantic_layout))):
        idx = list(indices)
        curve = _norm(residual[:, idx])
        group_color = group_styles[_GROUP_DISPLAY_NAMES[group_key]].color
        plot_visible_series(
            ax,
            t,
            curve,
            label="Residual" if row_idx == 0 else None,
            color=group_color,
            linestyle="-",
            linewidth=1.35,
            alpha=0.98,
            zorder=4,
        )
        if target_mask is not None:
            invalid = ~np.all(target_mask[:, idx], axis=1)
            if np.any(invalid):
                masked_legend_needed = True
                ax.scatter(
                    t[invalid],
                    curve[invalid],
                    label="Masked-out target",
                    s=14.0,
                    facecolors="white",
                    edgecolors="#8C9199",
                    linewidths=0.7,
                    zorder=5,
                )
        ax.set_ylabel(ylabel)
        apply_axes_style(ax, grid=False)

    if masked_legend_needed and "Masked-out target" not in axes[0].get_legend_handles_labels()[1]:
        axes[0].scatter(
            [],
            [],
            label="Masked-out target",
            s=14.0,
            facecolors="white",
            edgecolors="#8C9199",
            linewidths=0.7,
            zorder=5,
        )
    apply_shared_xlabels(list(axes), "Prediction horizon (s)")
    align_ylabels(axes)
    add_axes_legend(axes[0], loc="upper right", ncol=2 if masked_legend_needed else 1)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.12, hspace=0.08)
    return fig, (axes[0], axes[1], axes[2])


def build_component_residual_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    dt_s: float,
    semantic_layout: SemanticOutputLayout | None = None,
    target_mask: np.ndarray | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """构建单个样本窗口的 9 分量残差图。"""
    setup_mpl()

    if y_hat.ndim != 2 or y_true.ndim != 2 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect sample arrays with shape (H, D), got {y_hat.shape} and {y_true.shape}")
    if semantic_layout is None:
        semantic_layout = canonical_semantic_output_layout(y_hat.shape[-1])
    if target_mask is not None and target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: {target_mask.shape} vs {y_hat.shape}")

    H = y_hat.shape[0]
    t = np.arange(1, H + 1, dtype=float) * float(dt_s)
    residual = y_hat - y_true
    fig, axes = plt.subplots(3, 3, sharex=True, figsize=get_figure_size("component_3x3_compact"))
    flat_axes = tuple(axes.reshape(-1))
    xyz_styles = get_xyz_styles()

    for idx, (ax, component_label) in enumerate(zip(flat_axes, semantic_layout.component_labels)):
        component_key = str(component_label)
        ylabel, axis_key = _COMPONENT_AXIS_META.get(component_key, (component_key, "x"))
        style = xyz_styles.get(axis_key, xyz_styles["x"])
        ax.axhline(0.0, color="#30343A", linewidth=0.9, linestyle=":", alpha=0.7, zorder=2)
        line_label = "Residual" if idx == 0 else None
        plot_visible_series(
            ax,
            t,
            residual[:, idx],
            label=line_label,
            color=style.color,
            linestyle="-",
            linewidth=1.35,
            alpha=0.98,
            zorder=4,
        )
        if target_mask is not None:
            invalid = ~target_mask[:, idx]
            if np.any(invalid):
                mask_label = "Masked-out target" if idx == 0 else None
                ax.scatter(
                    t[invalid],
                    residual[invalid, idx],
                    label=mask_label,
                    s=14.0,
                    facecolors="white",
                    edgecolors="#8C9199",
                    linewidths=0.7,
                    zorder=5,
                )
        ax.set_ylabel(ylabel)
        apply_axes_style(ax, grid=False)

    for row in axes[:-1]:
        for ax in row:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
    for ax in axes[-1]:
        ax.set_xlabel("Prediction horizon (s)")
    add_axes_legend(flat_axes[0], loc="upper right", ncol=2 if target_mask is not None else 1)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.09, wspace=0.28, hspace=0.12)
    return fig, flat_axes


def plot_component_residuals_from_npz(
    pred_npz: Path,
    out_dir: Path,
    *,
    n: int = 4,
    dt_s: float = 0.01,
    fmt: str = "png",
    mode: str = "component",
) -> None:
    """从评估样本 artifact 读盘并导出分量残差图。"""
    _ensure_dir(out_dir)
    y_hat, y_true = _load_pred_npz(pred_npz)
    semantic_layout = load_semantic_layout_from_metrics_path(pred_npz.parent / "metrics.yaml", dout=y_hat.shape[-1])
    target_mask_all = _load_target_mask(pred_npz, shape=y_hat.shape)

    nplot = min(int(n), int(y_hat.shape[0]))
    for idx in range(nplot):
        if mode == "component":
            fig, _ = build_component_residual_figure(
                y_hat=y_hat[idx],
                y_true=y_true[idx],
                dt_s=dt_s,
                semantic_layout=semantic_layout,
                target_mask=None if target_mask_all is None else target_mask_all[idx],
            )
            out_name = f"residual_component_{idx:03d}"
        elif mode == "group_norm":
            fig, _ = build_group_residual_figure(
                y_hat=y_hat[idx],
                y_true=y_true[idx],
                dt_s=dt_s,
                semantic_layout=semantic_layout,
                target_mask=None if target_mask_all is None else target_mask_all[idx],
            )
            out_name = f"residual_group_norm_{idx:03d}"
        else:
            raise ValueError(f"Unknown residual plot mode: {mode}")
        save_figure(fig, out_dir / out_name, fmt=fmt)
        plt.close(fig)


def main() -> int:
    """分量残差图命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_component_residuals")
    ap.add_argument("--eval_dir", type=str, required=True, help="evaluation dir containing pred_samples.npz")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--n", type=int, default=4, help="number of sample windows to plot")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--mode", type=str, default="group_norm", choices=["group_norm", "component"])
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "plots")
    plot_component_residuals_from_npz(
        eval_dir / "pred_samples.npz",
        out_dir,
        n=int(args.n),
        dt_s=float(args.dt),
        fmt=str(args.fmt),
        mode=str(args.mode),
    )
    print(f"[VIZ] wrote component residual plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
