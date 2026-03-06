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
import pytest
import yaml

from uwnav_dynamics.models.utils.semantic_output_layout import (
    SEMANTIC_LAYOUT_SCHEMA_VERSION,
    build_semantic_layout_metadata,
    canonical_semantic_output_layout,
)
from uwnav_dynamics.viz.eval.plot_horizon_metrics import HorizonPlotCfg, build_groups_vs_horizon_figure, plot_groups_vs_horizon
from uwnav_dynamics.viz.eval.plot_model_compare import ModelCompareCfg, build_horizon_compare_figure, plot_horizon_compare
from uwnav_dynamics.viz.eval.plot_pred_vs_observed import PredObservedPlotCfg, build_pred_vs_observed_figure, plot_pred_vs_observed_from_npz
from uwnav_dynamics.viz.eval.plot_rollout_samples import plot_rollout_samples_from_npz


def _write_metric_csv(path: Path, data: np.ndarray) -> None:
    header = ",".join(["h"] + [f"d{i}" for i in range(data.shape[1])])
    rows = [header]
    for idx in range(data.shape[0]):
        rows.append(",".join([str(idx + 1)] + [f"{v:.6f}" for v in data[idx]]))
    path.write_text("\n".join(rows), encoding="utf-8")


def _layout_meta(group_indices: dict[str, list[int]] | None = None) -> dict:
    semantic = canonical_semantic_output_layout(9)
    if group_indices is not None:
        semantic = type(semantic)(
            source=semantic.source,
            component_labels=semantic.component_labels,
            group_indices={key: tuple(indices) for key, indices in group_indices.items()},
            validated_against_target_cols=semantic.validated_against_target_cols,
        )
    return {
        "schema_version": SEMANTIC_LAYOUT_SCHEMA_VERSION,
        "execution": {"source": "cfg_model.y_in_idx", "y_in_idx": list(range(8, 17))},
        "semantic": build_semantic_layout_metadata(semantic),
    }


def _make_eval_dir(root: Path, name: str, scale: float, *, include_layout: bool = True, layout_meta: dict | None = None) -> Path:
    eval_dir = root / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    meta = {"name": name}
    if include_layout:
        meta["layout"] = _layout_meta() if layout_meta is None else layout_meta
    with (eval_dir / "metrics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

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


def test_horizon_groups_plot_prefers_layout_metadata_over_canonical_slices(tmp_path):
    eval_dir = _make_eval_dir(
        tmp_path,
        "layout_eval",
        scale=1.0,
        layout_meta=_layout_meta(
            {
                "acc": [6, 7, 8],
                "gyro": [3, 4, 5],
                "vel": [0, 1, 2],
            }
        ),
    )
    cfg = HorizonPlotCfg(dt_s=0.01, use_seconds=True, metric="rmse", out_name="rmse_horizon_groups", fmt="png")

    fig, ax = build_groups_vs_horizon_figure(eval_dirs=[eval_dir], labels=["single"], cfg=cfg)

    rmse = np.loadtxt(eval_dir / "rmse_by_horizon.csv", delimiter=",", skiprows=1)[:, 1:]
    acc_line = next(line for line in ax.lines if line.get_label() == "Acc")
    vel_line = next(line for line in ax.lines if line.get_label() == "Vel")
    assert np.allclose(acc_line.get_ydata(), rmse[:, 6:9].mean(axis=1))
    assert np.allclose(vel_line.get_ydata(), rmse[:, 0:3].mean(axis=1))


def test_pred_vs_observed_falls_back_with_warning_when_layout_metadata_is_missing(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "legacy_eval", scale=1.0, include_layout=False)
    cfg = PredObservedPlotCfg(dt_s=0.02, mode="group_norm", fmt="png")

    with pytest.warns(UserWarning, match="missing layout\\.semantic metadata"):
        plot_pred_vs_observed_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, cfg=cfg)

    assert (eval_dir / "plots" / "pred_vs_observed_group_norm_000.png").exists()


def test_rollout_samples_fall_back_with_warning_when_layout_metadata_is_missing(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "legacy_rollout_eval", scale=1.0, include_layout=False)

    with pytest.warns(UserWarning, match="missing layout\\.semantic metadata"):
        plot_rollout_samples_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, dt_s=0.02, fmt="png")

    assert (eval_dir / "plots" / "rollout_sample_000.png").exists()
