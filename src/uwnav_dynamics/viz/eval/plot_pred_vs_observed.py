"""
模块名称：预测值与监督目标对比图

模块职责：
从评估阶段落盘的 `pred_samples.npz` 中读取 `y_hat / y_true / logvar`，
生成网络预测值与监督目标之间的对比图，为论文与汇报提供稳定图型。

主要功能：
1. 默认以 `group_norm` mode 绘制 semantic layout 驱动的三组范数对比图。
2. 明确将图中的 `observed` 解释为评估阶段监督目标 `y_true`。
3. 为未来 `component` mode 与 uncertainty band 扩展保留接口。

数据流：
pred_samples.npz
    ↓
mode dispatch (`group_norm` / future `component`)
    ↓
3×1 共享 x 轴 figure
    ↓
plots/pred_vs_observed_group_norm_000.png|pdf

依赖模块：
- numpy
- matplotlib
- uwnav_dynamics.viz.style.sci_style

备注：
- 本模块中的 `observed` 指当前评估阶段使用的监督目标 `y_true`。
- 它不等同于未经处理的原始 IMU / DVL / Power 传感器输出。
- 若旧 artifact 缺少 layout metadata，则统一 warning 并回退到 canonical `acc/gyro/vel` 分组。
- PR5 第一阶段不接入 sample-level masked visualization；若未来需要样本级有效性标记，应通过独立 artifact 扩展。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PredObservedPlotCfg:
    dt_s: float = 0.01
    mode: str = "group_norm"
    fmt: str = "png"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm3(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=-1))


def _load_pred_npz(pred_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(pred_npz) as z:
        if "y_hat" not in z or "y_true" not in z:
            raise ValueError(f"{pred_npz} must contain y_hat / y_true")
        y_hat = z["y_hat"]
        y_true = z["y_true"]
        logvar = z["logvar"] if "logvar" in z else np.full_like(y_hat, np.nan)

    if y_hat.ndim != 3 or y_true.ndim != 3 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect y_hat/y_true with identical shape (N,H,D), got {y_hat.shape} and {y_true.shape}")
    return y_hat, y_true, logvar


def _group_specs(semantic_layout: SemanticOutputLayout) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (group_key, semantic_layout.group_indices[group_key], _GROUP_YLABELS[group_key])
        for group_key in ("acc", "gyro", "vel")
    )


def _build_group_norm_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    dt_s: float,
    semantic_layout: SemanticOutputLayout | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    setup_mpl()

    if y_hat.ndim != 2 or y_true.ndim != 2 or y_hat.shape != y_true.shape:
        raise ValueError(f"Expect sample arrays with shape (H, D), got {y_hat.shape} and {y_true.shape}")
    if semantic_layout is None:
        semantic_layout = canonical_semantic_output_layout(y_hat.shape[-1])
    if y_hat.shape[-1] != len(semantic_layout.component_labels):
        raise ValueError(
            "Sample feature dim does not match semantic layout: "
            f"{y_hat.shape[-1]} vs {len(semantic_layout.component_labels)}"
        )

    H = y_hat.shape[0]
    t = np.arange(1, H + 1, dtype=float) * float(dt_s)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=get_figure_size("rollout_3row"))

    observed_style = get_observed_pred_styles()["observed"]
    pred_style = get_observed_pred_styles()["pred"]
    group_styles = get_group_styles()
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

        # TODO(PR-uncertainty): 若未来启用 uncertainty band，可在此处消费 logvar，
        # 绘制 prediction mean ± sigma 的低饱和填充区域。

    apply_shared_xlabels(list(axes), "Prediction horizon (s)")
    apply_minimal_legend(axes[0].legend(loc="upper right"))
    return fig, (axes[0], axes[1], axes[2])


def build_pred_vs_observed_figure(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    cfg: PredObservedPlotCfg,
    semantic_layout: SemanticOutputLayout | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    if cfg.mode == "group_norm":
        return _build_group_norm_figure(
            y_hat=y_hat,
            y_true=y_true,
            dt_s=cfg.dt_s,
            semantic_layout=semantic_layout,
        )
    if cfg.mode == "component":
        raise NotImplementedError("component mode is reserved for a future patch")
    raise ValueError(f"Unknown mode: {cfg.mode}")


def plot_pred_vs_observed_from_npz(
    pred_npz: Path,
    out_dir: Path,
    *,
    n: int = 4,
    cfg: PredObservedPlotCfg = PredObservedPlotCfg(),
) -> None:
    _ensure_dir(out_dir)
    y_hat, y_true, _ = _load_pred_npz(pred_npz)
    semantic_layout = load_semantic_layout_from_metrics_path(pred_npz.parent / "metrics.yaml", dout=y_hat.shape[-1])

    nplot = min(int(n), int(y_hat.shape[0]))
    for idx in range(nplot):
        fig, _ = build_pred_vs_observed_figure(
            y_hat=y_hat[idx],
            y_true=y_true[idx],
            cfg=cfg,
            semantic_layout=semantic_layout,
        )
        save_figure(fig, out_dir / f"pred_vs_observed_{cfg.mode}_{idx:03d}", fmt=cfg.fmt)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.viz.plot_pred_vs_observed")
    ap.add_argument("--eval_dir", type=str, required=True, help="evaluation dir containing pred_samples.npz")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--n", type=int, default=4, help="number of sample windows to plot")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--mode", type=str, default="group_norm", choices=["group_norm", "component"])
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "both"])
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "plots")
    cfg = PredObservedPlotCfg(dt_s=float(args.dt), mode=str(args.mode), fmt=str(args.fmt))
    plot_pred_vs_observed_from_npz(eval_dir / "pred_samples.npz", out_dir, n=int(args.n), cfg=cfg)
    print(f"[VIZ] wrote pred-vs-observed plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
