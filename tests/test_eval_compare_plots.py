"""
模块名称：评估扩展绘图测试

模块职责：
验证 pred-vs-observed 与 model-compare 两类新增图型
已经基于稳定 eval artifact 工作，并满足核心的版式契约。

主要功能：
1. 验证 pred-vs-observed 默认使用 group_norm mode，且输出命名稳定。
2. 验证 model compare 的 horizon 主路径输出稳定，并固化 primary / baseline 视觉层级。
3. 验证 horizon 单模型图仍保持无标题和最小 legend。

数据流：
synthetic eval_dir artifacts
    ↓
viz.eval.plot_* builders / exporters
    ↓
figure contract + output naming assertions

依赖模块：
- numpy
- yaml
- uwnav_dynamics.viz.eval.*
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from uwnav_dynamics.viz.eval.plot_horizon_metrics import HorizonPlotCfg, build_groups_vs_horizon_figure, plot_groups_vs_horizon
from uwnav_dynamics.viz.eval.plot_model_compare import ModelCompareCfg, build_horizon_compare_figure, plot_horizon_compare
from uwnav_dynamics.viz.eval.plot_pred_vs_observed import PredObservedPlotCfg, build_pred_vs_observed_figure, plot_pred_vs_observed_from_npz


def _write_metric_csv(path: Path, data: np.ndarray) -> None:
    header = ",".join(["h"] + [f"d{i}" for i in range(data.shape[1])])
    rows = [header]
    for idx in range(data.shape[0]):
        rows.append(",".join([str(idx + 1)] + [f"{v:.6f}" for v in data[idx]]))
    path.write_text("\n".join(rows), encoding="utf-8")


def _make_eval_dir(root: Path, name: str, scale: float) -> Path:
    eval_dir = root / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "metrics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"name": name}, f, sort_keys=False)

    base = np.linspace(0.1, 0.5, 5, dtype=float).reshape(5, 1)
    dims = np.arange(1, 10, dtype=float).reshape(1, 9)
    rmse = base * dims * scale
    mae = 0.8 * rmse
    _write_metric_csv(eval_dir / "rmse_by_horizon.csv", rmse)
    _write_metric_csv(eval_dir / "mae_by_horizon.csv", mae)

    y_true = np.tile(np.linspace(0.0, 1.0, 6, dtype=float).reshape(1, 6, 1), (2, 1, 9))
    y_hat = y_true + scale * 0.05
    logvar = np.full_like(y_hat, -2.0)
    np.savez_compressed(eval_dir / "pred_samples.npz", y_hat=y_hat, y_true=y_true, logvar=logvar)
    return eval_dir


def test_pred_vs_observed_group_norm_has_bottom_xlabel_and_single_legend(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_test", scale=1.0)
    z = np.load(eval_dir / "pred_samples.npz")
    cfg = PredObservedPlotCfg(dt_s=0.02, mode="group_norm", fmt="png")

    fig, axes = build_pred_vs_observed_figure(y_hat=z["y_hat"][0], y_true=z["y_true"][0], cfg=cfg)

    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Prediction horizon (s)"]
    assert sum(ax.get_legend() is not None for ax in axes) == 1
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == ["Observed target", "Prediction"]

    plot_pred_vs_observed_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, cfg=cfg)
    assert (eval_dir / "plots" / "pred_vs_observed_group_norm_000.png").exists()


def test_model_compare_horizon_highlights_primary_over_baseline(tmp_path):
    ours = _make_eval_dir(tmp_path, "ours_eval", scale=0.9)
    baseline = _make_eval_dir(tmp_path, "baseline_eval", scale=1.2)
    cfg = ModelCompareCfg(dt_s=0.01, use_seconds=True, metric="rmse", fmt="png")

    fig, axes = build_horizon_compare_figure(
        eval_dirs=[ours, baseline],
        labels=["ours", "baseline_lstm"],
        cfg=cfg,
    )

    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Prediction horizon (s)"]
    assert sum(ax.get_legend() is not None for ax in axes) == 1

    acc_lines = {line.get_label(): line for line in axes[0].lines}
    assert acc_lines["ours"].get_linewidth() > acc_lines["baseline_lstm"].get_linewidth()
    assert acc_lines["ours"].get_linestyle() == "-"
    assert acc_lines["baseline_lstm"].get_linestyle() == "--"

    plot_horizon_compare(
        eval_dirs=[ours, baseline],
        labels=["ours", "baseline_lstm"],
        out_dir=tmp_path / "compare_plots",
        cfg=cfg,
    )
    assert (tmp_path / "compare_plots" / "rmse_model_compare_horizon.png").exists()


def test_horizon_groups_plot_keeps_no_title_contract(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "single_eval", scale=1.0)
    cfg = HorizonPlotCfg(dt_s=0.02, use_seconds=False, metric="mae", out_name="mae_horizon_groups", fmt="png")

    fig, ax = build_groups_vs_horizon_figure(eval_dirs=[eval_dir], labels=["single"], cfg=cfg)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "Prediction step (k)"
    assert ax.get_legend() is not None

    plot_groups_vs_horizon(eval_dirs=[eval_dir], labels=["single"], out_dir=tmp_path / "plots", cfg=cfg)
    assert (tmp_path / "plots" / "mae_horizon_groups.png").exists()
