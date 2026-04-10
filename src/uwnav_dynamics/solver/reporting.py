"""
模块名称：状态求解器 replay 汇总工具

模块职责：
负责把长序列 replay 评估产生的嵌套 `metrics.yaml`
压平成稳定的一行报表字段，
并提供多方案比较时统一使用的标准排行协议。

主要功能：
1. 定义 replay `summary.csv` 的稳定字段集合。
2. 从 `metrics.yaml` 抽取全局误差、末步误差、误差增长、阈值失效、尾部风险与偏差指标。
3. 定义标准排行协议，明确哪些指标参与“哪种方案更好”的排序。

数据流：
replay metrics.yaml
    ↓
flatten_replay_metrics()
    ↓
summary.csv / ranking.csv

依赖模块：
- typing

备注：
- 当前排行协议面向“经验型状态求解器离线 replay 筛选”，不是闭环最终结论。
- 所有主指标统一采用“越小越好”口径。
"""

from __future__ import annotations

from typing import Any, Mapping


REPLAY_SUMMARY_FIELDS = [
    "split",
    "solver_step_semantics",
    "model_pred_len",
    "segment_count",
    "total_steps",
    "nonfinite_trigger_count",
    "finite_pass_rate",
    "segment_len_mean",
    "segment_len_p95",
    "rmse_global",
    "mae_global",
    "final_step_rmse_global_mean",
    "final_step_mae_global_mean",
    "final_step_abs_p95_global",
    "rmse_growth_mean",
    "rmse_growth_p95",
    "mae_growth_mean",
    "mae_growth_p95",
    "rmse_step_slope",
    "log_rmse_step_slope",
    "tail_abs_p95_global",
    "tail_abs_p99_global",
    "rmse_threshold",
    "rmse_threshold_failure_rate",
    "rmse_threshold_breach_step_mean",
    "abs_error_threshold",
    "abs_error_threshold_failure_rate",
    "abs_error_threshold_breach_step_mean",
    "worst_bias_component",
    "worst_abs_bias",
]


REPLAY_RANK_METRICS = [
    ("rmse_global", 3.0),
    ("mae_global", 2.0),
    ("final_step_rmse_global_mean", 3.0),
    ("rmse_growth_p95", 2.0),
    ("tail_abs_p95_global", 2.0),
    ("rmse_threshold_failure_rate", 2.0),
    ("tail_abs_p99_global", 1.0),
    ("abs_error_threshold_failure_rate", 1.0),
    ("worst_abs_bias", 1.0),
]


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def flatten_replay_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """把 replay `metrics.yaml` 压平成 summary.csv 可直接写出的稳定字段。"""
    if not isinstance(metrics, Mapping):
        return {field: "" for field in REPLAY_SUMMARY_FIELDS}

    final_step = _nested_get(metrics, "final_step", default={})
    rollout_growth = _nested_get(metrics, "rollout_growth", default={})
    tail_error = _nested_get(metrics, "tail_error", default={})
    long_horizon = _nested_get(metrics, "long_horizon", default={})
    threshold_cfg = _nested_get(long_horizon, "thresholds", default={})
    threshold_time = _nested_get(long_horizon, "time_to_threshold", default={})
    rmse_threshold = _nested_get(threshold_time, "rmse", default={})
    abs_threshold = _nested_get(threshold_time, "abs_error", default={})
    bias = _nested_get(metrics, "bias", default={})
    robustness = _nested_get(metrics, "robustness", default={})
    segment_stats = _nested_get(metrics, "segment_stats", default={})

    return {
        "split": metrics.get("split", ""),
        "solver_step_semantics": metrics.get("solver_step_semantics", ""),
        "model_pred_len": metrics.get("model_pred_len", ""),
        "segment_count": metrics.get("segment_count", ""),
        "total_steps": metrics.get("total_steps", ""),
        "nonfinite_trigger_count": metrics.get("nonfinite_trigger_count", ""),
        "finite_pass_rate": _nested_get(robustness, "finite_pass_rate"),
        "segment_len_mean": _nested_get(segment_stats, "n_steps_mean"),
        "segment_len_p95": _nested_get(segment_stats, "n_steps_p95"),
        "rmse_global": metrics.get("rmse_global", ""),
        "mae_global": metrics.get("mae_global", ""),
        "final_step_rmse_global_mean": _nested_get(final_step, "rmse_global_mean"),
        "final_step_mae_global_mean": _nested_get(final_step, "mae_global_mean"),
        "final_step_abs_p95_global": _nested_get(tail_error, "final_step_abs_p95_global"),
        "rmse_growth_mean": _nested_get(rollout_growth, "rmse_last_over_first_mean"),
        "rmse_growth_p95": _nested_get(rollout_growth, "rmse_last_over_first_p95"),
        "mae_growth_mean": _nested_get(rollout_growth, "mae_last_over_first_mean"),
        "mae_growth_p95": _nested_get(rollout_growth, "mae_last_over_first_p95"),
        "rmse_step_slope": _nested_get(long_horizon, "rmse_slope"),
        "log_rmse_step_slope": _nested_get(long_horizon, "log_rmse_slope"),
        "tail_abs_p95_global": _nested_get(tail_error, "abs_p95_global"),
        "tail_abs_p99_global": _nested_get(tail_error, "abs_p99_global"),
        "rmse_threshold": _nested_get(threshold_cfg, "rmse"),
        "rmse_threshold_failure_rate": _nested_get(rmse_threshold, "failure_rate"),
        "rmse_threshold_breach_step_mean": _nested_get(rmse_threshold, "breach_step_mean"),
        "abs_error_threshold": _nested_get(threshold_cfg, "abs_error"),
        "abs_error_threshold_failure_rate": _nested_get(abs_threshold, "failure_rate"),
        "abs_error_threshold_breach_step_mean": _nested_get(abs_threshold, "breach_step_mean"),
        "worst_bias_component": _nested_get(bias, "worst_component"),
        "worst_abs_bias": _nested_get(bias, "worst_abs_bias"),
    }


def replay_ranking_protocol() -> dict[str, Any]:
    """返回当前多方案 replay 排行所使用的标准协议。"""
    return {
        "schema_version": "transition_replay_ranking_v1",
        "intended_use": "offline_model_selection_for_transition_solver",
        "closed_loop_proof": False,
        "lower_is_better": True,
        "hard_gates": {
            "segment_count_gt_zero": True,
            "nonfinite_trigger_count_eq_zero": True,
        },
        "rank_metrics": [
            {"name": str(name), "weight": float(weight)}
            for name, weight in REPLAY_RANK_METRICS
        ],
        "tie_break_order": [
            "rmse_global",
            "final_step_rmse_global_mean",
            "rmse_growth_p95",
            "rmse_threshold_failure_rate",
            "name",
        ],
    }


__all__ = [
    "REPLAY_RANK_METRICS",
    "REPLAY_SUMMARY_FIELDS",
    "flatten_replay_metrics",
    "replay_ranking_protocol",
]
