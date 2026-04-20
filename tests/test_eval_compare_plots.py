"""
模块名称：评估扩展绘图测试

模块职责：
验证 pred-vs-observed、model-compare 与 control-readiness
三类扩展图型
已经基于稳定 eval artifact 工作，并满足核心的版式契约。

主要功能：
1. 验证 pred-vs-observed 默认使用 group_norm mode，且输出命名稳定。
2. 验证 model compare 的 horizon 主路径输出稳定，并固化 primary / baseline 视觉层级。
3. 验证 control-readiness summary / compare 图的命名、排序和版式契约。
4. 验证 horizon 单模型图仍保持无标题和最小 legend。
5. 验证 dense / masked horizon artifact 并行存在时，绘图脚本输出命名稳定且降级策略明确。
6. 验证单步 horizon 产物会补 marker，避免 STEP 类模型导出空图。

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
from uwnav_dynamics.viz.eval.plot_control_readiness import (
    ControlReadinessPlotCfg,
    build_control_readiness_figure,
    plot_control_readiness,
)
from uwnav_dynamics.viz.eval.plot_horizon_metrics import HorizonPlotCfg, build_groups_vs_horizon_figure, plot_groups_vs_horizon
from uwnav_dynamics.viz.eval.plot_model_compare import ModelCompareCfg, build_horizon_compare_figure, plot_horizon_compare
from uwnav_dynamics.viz.eval.plot_component_residuals import (
    build_component_residual_figure,
    plot_component_residuals_from_npz,
)
from uwnav_dynamics.viz.eval.plot_pred_vs_observed import PredObservedPlotCfg, build_pred_vs_observed_figure, plot_pred_vs_observed_from_npz
from uwnav_dynamics.viz.eval.plot_prediction_trace import (
    build_prediction_trace_figure,
    build_prediction_trace_group_axes_figure,
    plot_prediction_trace,
)
from uwnav_dynamics.viz.eval.plot_rollout_samples import build_rollout_sample_figure, plot_rollout_samples_from_npz


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


def _make_eval_dir(
    root: Path,
    name: str,
    scale: float,
    *,
    masked_scale: float | None = None,
    include_layout: bool = True,
    layout_meta: dict | None = None,
) -> Path:
    eval_dir = root / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    meta = {"name": name}
    diag_scale = float(scale)
    masked_diag_scale = float(masked_scale if masked_scale is not None else scale)
    meta["control_readiness"] = {
        "schema_version": "control_readiness_v1",
        "intended_use": "offline_screening_for_control",
        "closed_loop_proof": False,
        "physical": {
            "dense": {
                "final_step": {
                    "rmse_global": 0.5 * diag_scale,
                    "mae_global": 0.4 * diag_scale,
                    "group_rmse": {"acc": 0.3 * diag_scale, "gyro": 0.4 * diag_scale, "vel": 0.5 * diag_scale},
                    "group_mae": {"acc": 0.24 * diag_scale, "gyro": 0.32 * diag_scale, "vel": 0.4 * diag_scale},
                },
                "rollout_growth": {
                    "rmse_last_over_first": 1.0 + 0.2 * diag_scale,
                    "mae_last_over_first": 1.0 + 0.15 * diag_scale,
                    "group_rmse_last_over_first": {"acc": 1.0 + 0.1 * diag_scale, "gyro": 1.0 + 0.15 * diag_scale, "vel": 1.0 + 0.2 * diag_scale},
                    "group_mae_last_over_first": {"acc": 1.0 + 0.08 * diag_scale, "gyro": 1.0 + 0.12 * diag_scale, "vel": 1.0 + 0.16 * diag_scale},
                },
                "tail_error": {
                    "abs_p95_global": 0.8 * diag_scale,
                    "abs_p99_global": 1.1 * diag_scale,
                    "final_step_abs_p95_global": 0.7 * diag_scale,
                },
                "bias": {
                    "worst_component": "vel_x",
                    "worst_abs_bias": 0.15 * diag_scale,
                },
            },
            "masked": {
                "final_step": {
                    "rmse_global": 0.5 * masked_diag_scale,
                    "mae_global": 0.4 * masked_diag_scale,
                    "group_rmse": {"acc": 0.3 * masked_diag_scale, "gyro": 0.4 * masked_diag_scale, "vel": 0.5 * masked_diag_scale},
                    "group_mae": {"acc": 0.24 * masked_diag_scale, "gyro": 0.32 * masked_diag_scale, "vel": 0.4 * masked_diag_scale},
                },
                "rollout_growth": {
                    "rmse_last_over_first": 1.0 + 0.2 * masked_diag_scale,
                    "mae_last_over_first": 1.0 + 0.15 * masked_diag_scale,
                    "group_rmse_last_over_first": {"acc": 1.0 + 0.1 * masked_diag_scale, "gyro": 1.0 + 0.15 * masked_diag_scale, "vel": 1.0 + 0.2 * masked_diag_scale},
                    "group_mae_last_over_first": {"acc": 1.0 + 0.08 * masked_diag_scale, "gyro": 1.0 + 0.12 * masked_diag_scale, "vel": 1.0 + 0.16 * masked_diag_scale},
                },
                "tail_error": {
                    "abs_p95_global": 0.8 * masked_diag_scale,
                    "abs_p99_global": 1.1 * masked_diag_scale,
                    "final_step_abs_p95_global": 0.7 * masked_diag_scale,
                },
                "bias": {
                    "worst_component": "vel_x",
                    "worst_abs_bias": 0.15 * masked_diag_scale,
                },
            },
        },
    }
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
    if masked_scale is not None:
        rmse_masked = base * dims * masked_scale
        mae_masked = 0.8 * rmse_masked
        _write_metric_csv(eval_dir / "rmse_by_horizon_masked.csv", rmse_masked)
        _write_metric_csv(eval_dir / "mae_by_horizon_masked.csv", mae_masked)

    y_true = np.tile(np.linspace(0.0, 1.0, 6, dtype=float).reshape(1, 6, 1), (2, 1, 9))
    y_hat = y_true + scale * 0.05
    logvar = np.full_like(y_hat, -2.0)
    np.savez_compressed(eval_dir / "pred_samples.npz", y_hat=y_hat, y_true=y_true, logvar=logvar)
    target_mask = np.ones_like(y_hat, dtype=bool)
    target_mask[0, -1, 6:] = False
    np.savez_compressed(
        eval_dir / "pred_context.npz",
        target_mask=target_mask,
        sample_index=np.asarray([0, 1], dtype=np.int64),
        component_labels=np.asarray(list(canonical_semantic_output_layout(9).component_labels), dtype=str),
        component_display_labels=np.asarray(["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z", "Vel X", "Vel Y", "Vel Z"], dtype=str),
        component_units=np.asarray(["m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s", "m/s", "m/s", "m/s"], dtype=str),
    )
    trace_t = np.arange(20, dtype=float) * 0.01
    trace_true = np.tile(np.linspace(0.0, 1.0, 20, dtype=float).reshape(20, 1), (1, 9))
    trace_hat = trace_true + scale * 0.05
    np.savez_compressed(
        eval_dir / "pred_trace.npz",
        y_hat=trace_hat.astype(np.float32),
        y_true=trace_true.astype(np.float32),
        logvar=np.full_like(trace_hat, -2.0, dtype=np.float32),
        target_mask=np.ones_like(trace_hat, dtype=bool),
        sample_index=np.arange(20, dtype=np.int64),
        t_s=trace_t,
        component_labels=np.asarray(list(canonical_semantic_output_layout(9).component_labels), dtype=str),
        component_display_labels=np.asarray(["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z", "Vel X", "Vel Y", "Vel Z"], dtype=str),
        component_units=np.asarray(["m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s", "m/s", "m/s", "m/s"], dtype=str),
        horizon_index=np.asarray(0, dtype=np.int64),
        dt_s=np.asarray(0.01, dtype=np.float64),
        requested_seconds=np.asarray(50.0, dtype=np.float64),
    )
    return eval_dir


def test_pred_vs_observed_group_norm_has_bottom_xlabel_and_single_legend(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_test", scale=1.0)
    z = np.load(eval_dir / "pred_samples.npz")
    cfg = PredObservedPlotCfg(dt_s=0.02, mode="group_norm", fmt="png")

    fig, axes = build_pred_vs_observed_figure(y_hat=z["y_hat"][0], y_true=z["y_true"][0], cfg=cfg)

    assert [ax.get_title() for ax in axes] == ["", "", ""]
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Prediction horizon (s)"]
    assert len(fig.legends) == 0
    assert axes[0].get_legend() is not None
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == ["Target", "Pred"]

    plot_pred_vs_observed_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, cfg=cfg)
    assert (eval_dir / "plots" / "pred_vs_observed_group_norm_000.png").exists()


def test_prediction_trace_uses_three_shared_x_axes(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_trace", scale=1.0)
    with np.load(eval_dir / "pred_trace.npz", allow_pickle=False) as z:
        semantic_layout = canonical_semantic_output_layout(9)
        fig, axes = build_prediction_trace_figure(
            y_hat=z["y_hat"],
            y_true=z["y_true"],
            target_mask=z["target_mask"],
            t_s=z["t_s"],
            semantic_layout=semantic_layout,
        )

    assert len(axes) == 3
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Trace time (s)"]
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Group Norm Prediction Trace"
    assert len(fig.legends) == 1
    assert axes[0].get_legend() is None

    plot_prediction_trace(eval_dir, eval_dir / "plots", fmt="png", mode="group_norm")
    assert (eval_dir / "plots" / "prediction_trace_group_norm.png").exists()


def test_prediction_trace_group_axes_writes_one_figure_per_physical_group(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_trace_axes", scale=1.0)
    with np.load(eval_dir / "pred_trace.npz", allow_pickle=False) as z:
        semantic_layout = canonical_semantic_output_layout(9)
        fig, axes = build_prediction_trace_group_axes_figure(
            y_hat=z["y_hat"],
            y_true=z["y_true"],
            target_mask=z["target_mask"],
            t_s=z["t_s"],
            semantic_layout=semantic_layout,
            group_key="acc",
        )

    assert len(axes) == 3
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Trace time (s)"]
    assert [ax.get_ylabel() for ax in axes] == ["acc_x", "acc_y", "acc_z"]
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Acceleration Prediction Trace"
    assert len(fig.legends) == 1
    assert axes[0].get_legend() is None

    plot_prediction_trace(eval_dir, eval_dir / "plots", fmt="png")
    assert (eval_dir / "plots" / "prediction_trace_acc_axes.png").exists()
    assert (eval_dir / "plots" / "prediction_trace_gyro_axes.png").exists()
    assert (eval_dir / "plots" / "prediction_trace_vel_axes.png").exists()


def test_pred_vs_observed_component_mode_writes_3x3_component_figure(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_component", scale=1.0)
    z = np.load(eval_dir / "pred_samples.npz")
    cfg = PredObservedPlotCfg(dt_s=0.02, mode="component", fmt="png")

    fig, axes = build_pred_vs_observed_figure(y_hat=z["y_hat"][0], y_true=z["y_true"][0], cfg=cfg)

    assert len(axes) == 9
    assert [ax.get_title() for ax in axes] == [""] * 9
    assert [ax.get_xlabel() for ax in axes[:6]] == [""] * 6
    assert [ax.get_xlabel() for ax in axes[6:]] == ["Prediction horizon (s)"] * 3
    assert len(fig.legends) == 0
    assert axes[0].get_legend() is not None

    plot_pred_vs_observed_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, cfg=cfg)
    assert (eval_dir / "plots" / "pred_vs_observed_component_000.png").exists()


def test_component_residual_figure_writes_png_and_marks_masked_targets(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_residual", scale=1.0)
    z = np.load(eval_dir / "pred_samples.npz")
    ctx = np.load(eval_dir / "pred_context.npz", allow_pickle=False)

    fig, axes = build_component_residual_figure(
        y_hat=z["y_hat"][0],
        y_true=z["y_true"][0],
        dt_s=0.02,
        target_mask=ctx["target_mask"][0],
    )

    assert len(axes) == 9
    assert [ax.get_xlabel() for ax in axes[:6]] == [""] * 6
    assert [ax.get_xlabel() for ax in axes[6:]] == ["Prediction horizon (s)"] * 3
    assert len(fig.legends) == 0
    assert axes[0].get_legend() is not None

    plot_component_residuals_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, dt_s=0.02, fmt="png")
    assert (eval_dir / "plots" / "residual_component_000.png").exists()


def test_rollout_sample_figure_marks_masked_targets_when_pred_context_exists(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "eval_rollout_masked", scale=1.0)
    z = np.load(eval_dir / "pred_samples.npz")
    ctx = np.load(eval_dir / "pred_context.npz", allow_pickle=False)

    fig, axes = build_rollout_sample_figure(
        y_hat=z["y_hat"][0],
        y_true=z["y_true"][0],
        dt_s=0.02,
        target_mask=ctx["target_mask"][0],
    )

    assert len(axes) == 3
    assert [ax.get_xlabel() for ax in axes] == ["", "", "Prediction horizon (s)"]
    assert len(axes[2].collections) == 1
    assert axes[0].get_legend() is not None
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
        "Target",
        "Pred",
        "Masked",
    ]

    plot_rollout_samples_from_npz(eval_dir / "pred_samples.npz", eval_dir / "plots", n=1, dt_s=0.02, fmt="png")
    assert (eval_dir / "plots" / "rollout_sample_000.png").exists()


def test_single_step_eval_plots_use_markers_to_avoid_blank_figures(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "single_step_eval", scale=1.0, masked_scale=0.5)
    one_step_metric = np.arange(1, 10, dtype=float).reshape(1, 9) * 0.01
    _write_metric_csv(eval_dir / "rmse_by_horizon.csv", one_step_metric)
    _write_metric_csv(eval_dir / "rmse_by_horizon_masked.csv", one_step_metric * 0.5)

    y_true = np.arange(9, dtype=float).reshape(1, 1, 9) * 0.01
    y_hat = y_true + 0.001
    np.savez_compressed(eval_dir / "pred_samples.npz", y_hat=y_hat, y_true=y_true)
    np.savez_compressed(
        eval_dir / "pred_context.npz",
        target_mask=np.ones_like(y_hat, dtype=bool),
    )

    horizon_cfg = HorizonPlotCfg(dt_s=0.01, use_seconds=True, metric="rmse", out_name="rmse_horizon_groups", fmt="png")
    fig_horizon, ax_horizon = build_groups_vs_horizon_figure(eval_dirs=[eval_dir], labels=["single"], cfg=horizon_cfg)
    assert all(line.get_marker() == "o" for line in ax_horizon.lines)

    pred_cfg = PredObservedPlotCfg(dt_s=0.01, mode="component", fmt="png")
    fig_pred, pred_axes = build_pred_vs_observed_figure(y_hat=y_hat[0], y_true=y_true[0], cfg=pred_cfg)
    assert all(line.get_marker() == "o" for ax in pred_axes for line in ax.lines)

    fig_rollout, rollout_axes = build_rollout_sample_figure(y_hat=y_hat[0], y_true=y_true[0], dt_s=0.01)
    assert all(line.get_marker() == "o" for ax in rollout_axes for line in ax.lines)

    fig_residual, residual_axes = build_component_residual_figure(y_hat=y_hat[0], y_true=y_true[0], dt_s=0.01)
    residual_lines = [line for ax in residual_axes for line in ax.lines if line.get_label() != "_child0"]
    assert all(line.get_marker() == "o" for line in residual_lines)

    for fig in (fig_horizon, fig_pred, fig_rollout, fig_residual):
        fig.clf()


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
    assert len(fig.legends) == 1

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


def test_model_compare_normalizes_legacy_labels_in_legend(tmp_path):
    dynunc = _make_eval_dir(tmp_path, "dynunc_eval", scale=0.9)
    unc = _make_eval_dir(tmp_path, "unc_eval", scale=1.1)
    cfg = ModelCompareCfg(dt_s=0.01, use_seconds=True, metric="rmse", fmt="png")

    fig, axes = build_horizon_compare_figure(
        eval_dirs=[dynunc, unc],
        labels=["B4+U1 final-confirm seed8", "U1 uncertainty seed8"],
        cfg=cfg,
    )

    acc_lines = {line.get_label(): line for line in axes[0].lines}
    assert set(acc_lines.keys()) == {"DynUnc s8", "Unc s8"}
    fig.clf()


def test_model_compare_emits_masked_compare_when_all_eval_dirs_have_masked_csv(tmp_path):
    ours = _make_eval_dir(tmp_path, "ours_eval", scale=0.9, masked_scale=0.6)
    baseline = _make_eval_dir(tmp_path, "baseline_eval", scale=1.2, masked_scale=0.8)
    cfg = ModelCompareCfg(dt_s=0.01, use_seconds=True, metric="rmse", fmt="png")

    plot_horizon_compare(
        eval_dirs=[ours, baseline],
        labels=["ours", "baseline_lstm"],
        out_dir=tmp_path / "compare_plots",
        cfg=cfg,
    )

    assert (tmp_path / "compare_plots" / "rmse_model_compare_horizon.png").exists()
    assert (tmp_path / "compare_plots" / "rmse_model_compare_horizon_masked.png").exists()


def test_control_readiness_summary_writes_dense_and_masked_artifacts(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "single_eval", scale=1.0, masked_scale=0.6)
    cfg = ControlReadinessPlotCfg(fmt="png")

    fig, axes = build_control_readiness_figure(
        eval_dirs=[eval_dir],
        labels=["single"],
        cfg=cfg,
    )

    assert len(axes) == 4
    assert [ax.get_xlabel() for ax in axes[:2]] == ["", ""]
    assert [ax.get_xlabel() for ax in axes[2:]] == ["Model variant", "Model variant"]
    assert [tick.get_text() for tick in axes[2].get_xticklabels()] == ["single"]

    plot_control_readiness(
        eval_dirs=[eval_dir],
        labels=["single"],
        out_dir=tmp_path / "plots",
        cfg=cfg,
    )
    assert (tmp_path / "plots" / "control_readiness_summary.png").exists()
    assert (tmp_path / "plots" / "control_readiness_summary_masked.png").exists()


def test_control_readiness_compare_orders_primary_before_baseline(tmp_path):
    ours = _make_eval_dir(tmp_path, "ours_eval", scale=0.9, masked_scale=0.6)
    baseline = _make_eval_dir(tmp_path, "baseline_eval", scale=1.2, masked_scale=0.8)
    cfg = ControlReadinessPlotCfg(fmt="png")

    fig, axes = build_control_readiness_figure(
        eval_dirs=[baseline, ours],
        labels=["baseline_lstm", "ours"],
        cfg=cfg,
    )

    final_rmse_bars = axes[0].patches
    assert [tick.get_text() for tick in axes[2].get_xticklabels()] == ["ours", "baseline_lstm"]
    assert final_rmse_bars[0].get_height() == pytest.approx(0.45)
    assert final_rmse_bars[1].get_height() == pytest.approx(0.6)

    plot_control_readiness(
        eval_dirs=[baseline, ours],
        labels=["baseline_lstm", "ours"],
        out_dir=tmp_path / "compare_plots",
        cfg=cfg,
    )
    assert (tmp_path / "compare_plots" / "control_readiness_compare.png").exists()
    assert (tmp_path / "compare_plots" / "control_readiness_compare_masked.png").exists()


def test_control_readiness_normalizes_legacy_tick_labels(tmp_path):
    dyn = _make_eval_dir(tmp_path, "dyn_eval", scale=0.9, masked_scale=0.6)
    base = _make_eval_dir(tmp_path, "base_eval", scale=1.2, masked_scale=0.8)
    cfg = ControlReadinessPlotCfg(fmt="png")

    fig, axes = build_control_readiness_figure(
        eval_dirs=[base, dyn],
        labels=["B0 baseline seed0", "B4 thruster+hydro seed0"],
        cfg=cfg,
    )

    assert [tick.get_text() for tick in axes[2].get_xticklabels()] == ["Base s0", "Dyn s0"]
    fig.clf()


def test_horizon_groups_plot_keeps_no_title_contract(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "single_eval", scale=1.0)
    cfg = HorizonPlotCfg(dt_s=0.02, use_seconds=False, metric="mae", out_name="mae_horizon_groups", fmt="png")

    fig, ax = build_groups_vs_horizon_figure(eval_dirs=[eval_dir], labels=["single"], cfg=cfg)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "Prediction step (k)"
    assert len(fig.legends) == 1

    plot_groups_vs_horizon(eval_dirs=[eval_dir], labels=["single"], out_dir=tmp_path / "plots", cfg=cfg)
    assert (tmp_path / "plots" / "mae_horizon_groups.png").exists()


def test_horizon_groups_plot_emits_masked_artifact_when_masked_csv_exists(tmp_path):
    eval_dir = _make_eval_dir(tmp_path, "masked_eval", scale=1.0, masked_scale=0.5)
    cfg = HorizonPlotCfg(dt_s=0.02, use_seconds=True, metric="rmse", out_name="rmse_horizon_groups", fmt="png")

    fig, ax = build_groups_vs_horizon_figure(
        eval_dirs=[eval_dir],
        labels=["single"],
        cfg=cfg,
        artifact_variant="masked",
    )
    masked_rmse = np.loadtxt(eval_dir / "rmse_by_horizon_masked.csv", delimiter=",", skiprows=1)[:, 1:]
    vel_line = next(line for line in ax.lines if line.get_label() == "Vel")
    assert ax.get_title() == ""
    assert np.allclose(vel_line.get_ydata(), masked_rmse[:, 6:9].mean(axis=1))

    plot_groups_vs_horizon(eval_dirs=[eval_dir], labels=["single"], out_dir=tmp_path / "plots", cfg=cfg)
    assert (tmp_path / "plots" / "rmse_horizon_groups.png").exists()
    assert (tmp_path / "plots" / "rmse_horizon_groups_masked.png").exists()


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


def test_model_compare_warns_when_masked_artifacts_are_incomplete(tmp_path):
    ours = _make_eval_dir(tmp_path, "ours_eval", scale=0.9, masked_scale=0.6)
    baseline = _make_eval_dir(tmp_path, "baseline_eval", scale=1.2)
    cfg = ModelCompareCfg(dt_s=0.01, use_seconds=True, metric="rmse", fmt="png")

    with pytest.warns(UserWarning, match="skip masked compare"):
        plot_horizon_compare(
            eval_dirs=[ours, baseline],
            labels=["ours", "baseline_lstm"],
            out_dir=tmp_path / "compare_plots",
            cfg=cfg,
        )

    assert (tmp_path / "compare_plots" / "rmse_model_compare_horizon.png").exists()
    assert not (tmp_path / "compare_plots" / "rmse_model_compare_horizon_masked.png").exists()
