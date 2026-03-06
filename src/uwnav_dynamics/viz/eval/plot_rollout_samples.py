"""
模块名称：rollout 样例绘图

模块职责：
从评估阶段落盘的 `pred_samples.npz` 中读取模型预测与监督目标，
生成论文友好的 rollout 样例图，用于快速检查时域拟合质量。

主要功能：
1. 读取 `pred_samples.npz` 中的 `y_hat / y_true / logvar`。
2. 优先根据 `metrics.yaml.layout.semantic` 分组，并在每个组内取范数。
3. 以 3×1 共享 x 轴布局输出 `rollout_sample_*.png|pdf`。

数据流：
pred_samples.npz
    ↓
按 [Acc3, Gyro3, Vel3] 分组
    ↓
3 行共享 x 轴 figure
    ↓
plots/rollout_sample_000.png|pdf

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 当前图中的 “observed/true” 来自评估监督目标 `y_true`，不等同于未经处理的原始传感器输出。
- `logvar` 当前只保留供未来不确定度带扩展，不改变本次最小 patch 的默认显示。
- 若旧 artifact 缺少 layout metadata，则统一 warning 并回退到 canonical `acc/gyro/vel` 分组。
- PR5 第一阶段不接入 sample-level masked visualization，后续若需要单样本监督有效性显示，应通过独立 artifact 扩展。
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
    apply_axes_style,
    apply_minimal_legend,
    apply_shared_xlabels,
    get_figure_size,
    get_group_styles,
    get_observed_pred_styles,
    save_figure,
    setup_mpl,
)


_GROUP_DISPLAY_NAMES = {
    "acc": "Acc",
    "gyro": "Gyro",
    "vel": "Vel",
}
_GROUP_YLABELS = {
    "acc": r"||Acc||",
    "gyro": r"||Gyro||",
    "vel": r"||Vel||",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm3(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=-1))


def _load_pred_npz(pred_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(pred_npz) as z:
        if "y_hat" not in z or "y_true" not in z:
            raise ValueError(f"{pred_npz} must contain keys: y_hat, y_true (and optionally logvar)")
        y_hat = z["y_hat"]
        y_true = z["y_true"]
        logvar = z["logvar"] if "logvar" in z else np.full_like(y_hat, np.nan)

    if y_hat.ndim != 3 or y_true.ndim != 3:
        raise ValueError(f"Expect y_hat/y_true to be 3D, got {y_hat.shape}, {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat={y_hat.shape}, y_true={y_true.shape}")
    return y_hat, y_true, logvar


def _group_specs(semantic_layout: SemanticOutputLayout) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (group_key, semantic_layout.group_indices[group_key], _GROUP_YLABELS[group_key])
        for group_key in ("acc", "gyro", "vel")
    )


def build_rollout_sample_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    dt_s: float,
    semantic_layout: SemanticOutputLayout | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    setup_mpl()

    if y_hat.ndim != 2 or y_true.ndim != 2 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect sample arrays with shape (H, 9), got {y_hat.shape} and {y_true.shape}")
    if semantic_layout is None:
        semantic_layout = canonical_semantic_output_layout(y_hat.shape[-1])
    if y_hat.shape[-1] != len(semantic_layout.component_labels):
        raise ValueError(
            "Sample feature dim does not match semantic layout: "
            f"{y_hat.shape[-1]} vs {len(semantic_layout.component_labels)}"
        )

    H = y_hat.shape[0]
    t = np.arange(1, H + 1, dtype=float) * float(dt_s)
    observed_style = get_observed_pred_styles()["observed"]
    pred_style = get_observed_pred_styles()["pred"]
    group_styles = get_group_styles()

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("rollout_3row"))
    group_specs = _group_specs(semantic_layout)

    for ax, (group_key, indices, ylabel) in zip(axes, group_specs):
        obs = _norm3(y_true[:, list(indices)])
        pred = _norm3(y_hat[:, list(indices)])
        group_color = group_styles[_GROUP_DISPLAY_NAMES[group_key]].color

        ax.plot(
            t,
            obs,
            label="Observed target",
            color=observed_style.color,
            linestyle=observed_style.linestyle,
            linewidth=observed_style.linewidth,
            alpha=observed_style.alpha,
            zorder=observed_style.zorder,
        )
        ax.plot(
            t,
            pred,
            label="Prediction",
            color=group_color,
            linestyle=pred_style.linestyle,
            linewidth=pred_style.linewidth,
            alpha=pred_style.alpha,
            zorder=pred_style.zorder,
        )
        ax.set_ylabel(ylabel)
        apply_axes_style(ax, grid=False)

    apply_shared_xlabels(list(axes), "Prediction horizon (s)")
    leg = axes[0].legend(loc="upper right")
    apply_minimal_legend(leg)
    return fig, (axes[0], axes[1], axes[2])


def plot_rollout_samples_from_npz(
    pred_npz: Path,
    out_dir: Path,
    *,
    dt_s: float = 0.01,
    n: int = 8,
    fmt: str = "png",
) -> None:
    pred_npz = Path(pred_npz)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    y_hat, y_true, _ = _load_pred_npz(pred_npz)
    semantic_layout = load_semantic_layout_from_metrics_path(pred_npz.parent / "metrics.yaml", dout=y_hat.shape[-1])
    N = y_hat.shape[0]
    nplot = min(int(n), int(N))

    for i in range(nplot):
        fig, _ = build_rollout_sample_figure(
            y_hat=y_hat[i],
            y_true=y_true[i],
            dt_s=dt_s,
            semantic_layout=semantic_layout,
        )
        save_figure(fig, out_dir / f"rollout_sample_{i:03d}", fmt=fmt)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_rollout_samples")
    ap.add_argument("--eval_dir", type=str, required=True, help="evaluation dir containing pred_samples.npz")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--n", type=int, default=8, help="number of sample windows to plot")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "plots")

    plot_rollout_samples_from_npz(
        pred_npz=eval_dir / "pred_samples.npz",
        out_dir=out_dir,
        dt_s=float(args.dt),
        n=int(args.n),
        fmt=str(args.fmt),
    )
    print(f"[VIZ] wrote rollout sample plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
