"""
模块名称：实验汇总与报表工具

模块职责：
负责把训练摘要与评估 `metrics.yaml` 中的嵌套指标
压平成稳定的一行实验报表字段，
供 `summary.csv`、baseline runner 与矩阵实验统一复用。

主要功能：
1. 提供稳定的 summary 列名集合。
2. 从 `metrics.yaml` 提取 global / final-step / tail / growth / bias 指标。
3. 从 `train_summary.yaml` 提取训练停止点、monitor、学习率与耗时摘要。

数据流：
train_summary.yaml / metrics.yaml
    ↓
flatten_train_summary() / flatten_eval_metrics()
    ↓
dict[str, Any]
    ↓
summary.csv

依赖模块：
- typing

备注：
- 缺失字段会回退为空字符串或 `nan`，避免单个运行缺某项时破坏整张总表。
- 该模块只做字段抽取，不负责文件读写与路径格式化。
"""

from __future__ import annotations

from typing import Any, Mapping


TRAIN_SUMMARY_FIELDS = [
    "best_val",
    "best_epoch",
    "monitor_name",
    "best_monitor",
    "best_monitor_epoch",
    "selected_val_loss",
    "best_val_loss",
    "best_val_loss_epoch",
    "epochs_ran",
    "stopped_early",
    "final_lr",
    "train_wall_time_sec",
    "split_strategy",
    "dropped_window_count",
]


EVAL_SUMMARY_FIELDS = [
    "n_eval",
    "rmse_global",
    "mae_global",
    "rmse_global_masked",
    "mae_global_masked",
    "rmse_global_zspace",
    "mae_global_zspace",
    "rmse_global_masked_zspace",
    "mae_global_masked_zspace",
    "final_step_rmse_global_dense",
    "final_step_mae_global_dense",
    "final_step_rmse_global_masked",
    "final_step_mae_global_masked",
    "final_step_rmse_acc_dense",
    "final_step_rmse_gyro_dense",
    "final_step_rmse_vel_dense",
    "final_step_rmse_acc_masked",
    "final_step_rmse_gyro_masked",
    "final_step_rmse_vel_masked",
    "final_step_mae_acc_dense",
    "final_step_mae_gyro_dense",
    "final_step_mae_vel_dense",
    "final_step_mae_acc_masked",
    "final_step_mae_gyro_masked",
    "final_step_mae_vel_masked",
    "tail_abs_p95_dense",
    "tail_abs_p99_dense",
    "tail_final_step_abs_p95_dense",
    "tail_abs_p95_masked",
    "tail_abs_p99_masked",
    "tail_final_step_abs_p95_masked",
    "long_horizon_rmse_auc_dense",
    "long_horizon_mae_auc_dense",
    "long_horizon_late_rmse_dense",
    "long_horizon_late_mae_dense",
    "long_horizon_rmse_slope_dense",
    "long_horizon_mae_slope_dense",
    "long_horizon_rmse_auc_masked",
    "long_horizon_mae_auc_masked",
    "long_horizon_late_rmse_masked",
    "long_horizon_late_mae_masked",
    "long_horizon_rmse_slope_masked",
    "long_horizon_mae_slope_masked",
    "rmse_growth_dense",
    "mae_growth_dense",
    "rmse_growth_masked",
    "mae_growth_masked",
    "worst_bias_component_dense",
    "worst_abs_bias_dense",
    "worst_bias_component_masked",
    "worst_abs_bias_masked",
]


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def flatten_train_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """抽取训练摘要中的稳定字段。"""
    if not isinstance(summary, Mapping):
        return {field: "" for field in TRAIN_SUMMARY_FIELDS}
    return {
        "best_val": summary.get("best_val", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "monitor_name": summary.get("monitor_name", ""),
        "best_monitor": summary.get("best_monitor", ""),
        "best_monitor_epoch": summary.get("best_monitor_epoch", ""),
        "selected_val_loss": summary.get("selected_val_loss", ""),
        "best_val_loss": summary.get("best_val_loss", ""),
        "best_val_loss_epoch": summary.get("best_val_loss_epoch", ""),
        "epochs_ran": summary.get("epochs_ran", ""),
        "stopped_early": summary.get("stopped_early", ""),
        "final_lr": summary.get("final_lr", ""),
        "train_wall_time_sec": summary.get("train_wall_time_sec", ""),
        "split_strategy": summary.get("split_strategy", ""),
        "dropped_window_count": summary.get("dropped_window_count", ""),
    }


def flatten_eval_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """把嵌套的评估指标压平为 summary.csv 可直接写出的字段。"""
    if not isinstance(metrics, Mapping):
        return {field: "" for field in EVAL_SUMMARY_FIELDS}

    dense = _nested_get(metrics, "control_readiness", "physical", "dense", default={})
    masked = _nested_get(metrics, "control_readiness", "physical", "masked", default={})
    long_horizon = _nested_get(metrics, "long_horizon_fit", "physical", default={})
    long_dense = _nested_get(long_horizon, "dense", default={})
    long_masked = _nested_get(long_horizon, "masked", default={})

    dense_final = _nested_get(dense, "final_step", default={})
    dense_tail = _nested_get(dense, "tail_error", default={})
    dense_growth = _nested_get(dense, "rollout_growth", default={})
    dense_bias = _nested_get(dense, "bias", default={})

    masked_final = _nested_get(masked, "final_step", default={})
    masked_tail = _nested_get(masked, "tail_error", default={})
    masked_growth = _nested_get(masked, "rollout_growth", default={})
    masked_bias = _nested_get(masked, "bias", default={})

    dense_group_rmse = _nested_get(dense_final, "group_rmse", default={})
    masked_group_rmse = _nested_get(masked_final, "group_rmse", default={})
    dense_group_mae = _nested_get(dense_final, "group_mae", default={})
    masked_group_mae = _nested_get(masked_final, "group_mae", default={})

    return {
        "n_eval": metrics.get("n_eval", ""),
        "rmse_global": metrics.get("rmse_global", ""),
        "mae_global": metrics.get("mae_global", ""),
        "rmse_global_masked": metrics.get("rmse_global_masked", ""),
        "mae_global_masked": metrics.get("mae_global_masked", ""),
        "rmse_global_zspace": metrics.get("rmse_global_zspace", ""),
        "mae_global_zspace": metrics.get("mae_global_zspace", ""),
        "rmse_global_masked_zspace": metrics.get("rmse_global_masked_zspace", ""),
        "mae_global_masked_zspace": metrics.get("mae_global_masked_zspace", ""),
        "final_step_rmse_global_dense": _nested_get(dense_final, "rmse_global"),
        "final_step_mae_global_dense": _nested_get(dense_final, "mae_global"),
        "final_step_rmse_global_masked": _nested_get(masked_final, "rmse_global"),
        "final_step_mae_global_masked": _nested_get(masked_final, "mae_global"),
        "final_step_rmse_acc_dense": _nested_get(dense_group_rmse, "acc"),
        "final_step_rmse_gyro_dense": _nested_get(dense_group_rmse, "gyro"),
        "final_step_rmse_vel_dense": _nested_get(dense_group_rmse, "vel"),
        "final_step_rmse_acc_masked": _nested_get(masked_group_rmse, "acc"),
        "final_step_rmse_gyro_masked": _nested_get(masked_group_rmse, "gyro"),
        "final_step_rmse_vel_masked": _nested_get(masked_group_rmse, "vel"),
        "final_step_mae_acc_dense": _nested_get(dense_group_mae, "acc"),
        "final_step_mae_gyro_dense": _nested_get(dense_group_mae, "gyro"),
        "final_step_mae_vel_dense": _nested_get(dense_group_mae, "vel"),
        "final_step_mae_acc_masked": _nested_get(masked_group_mae, "acc"),
        "final_step_mae_gyro_masked": _nested_get(masked_group_mae, "gyro"),
        "final_step_mae_vel_masked": _nested_get(masked_group_mae, "vel"),
        "tail_abs_p95_dense": _nested_get(dense_tail, "abs_p95_global"),
        "tail_abs_p99_dense": _nested_get(dense_tail, "abs_p99_global"),
        "tail_final_step_abs_p95_dense": _nested_get(dense_tail, "final_step_abs_p95_global"),
        "tail_abs_p95_masked": _nested_get(masked_tail, "abs_p95_global"),
        "tail_abs_p99_masked": _nested_get(masked_tail, "abs_p99_global"),
        "tail_final_step_abs_p95_masked": _nested_get(masked_tail, "final_step_abs_p95_global"),
        "long_horizon_rmse_auc_dense": _nested_get(long_dense, "rmse_auc_global"),
        "long_horizon_mae_auc_dense": _nested_get(long_dense, "mae_auc_global"),
        "long_horizon_late_rmse_dense": _nested_get(long_dense, "late_horizon_rmse_global_mean"),
        "long_horizon_late_mae_dense": _nested_get(long_dense, "late_horizon_mae_global_mean"),
        "long_horizon_rmse_slope_dense": _nested_get(long_dense, "rmse_step_slope"),
        "long_horizon_mae_slope_dense": _nested_get(long_dense, "mae_step_slope"),
        "long_horizon_rmse_auc_masked": _nested_get(long_masked, "rmse_auc_global"),
        "long_horizon_mae_auc_masked": _nested_get(long_masked, "mae_auc_global"),
        "long_horizon_late_rmse_masked": _nested_get(long_masked, "late_horizon_rmse_global_mean"),
        "long_horizon_late_mae_masked": _nested_get(long_masked, "late_horizon_mae_global_mean"),
        "long_horizon_rmse_slope_masked": _nested_get(long_masked, "rmse_step_slope"),
        "long_horizon_mae_slope_masked": _nested_get(long_masked, "mae_step_slope"),
        "rmse_growth_dense": _nested_get(dense_growth, "rmse_last_over_first"),
        "mae_growth_dense": _nested_get(dense_growth, "mae_last_over_first"),
        "rmse_growth_masked": _nested_get(masked_growth, "rmse_last_over_first"),
        "mae_growth_masked": _nested_get(masked_growth, "mae_last_over_first"),
        "worst_bias_component_dense": _nested_get(dense_bias, "worst_component"),
        "worst_abs_bias_dense": _nested_get(dense_bias, "worst_abs_bias"),
        "worst_bias_component_masked": _nested_get(masked_bias, "worst_component"),
        "worst_abs_bias_masked": _nested_get(masked_bias, "worst_abs_bias"),
    }
