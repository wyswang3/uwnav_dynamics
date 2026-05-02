"""
模块名称：离线数值评估主程序

模块职责：
负责加载训练阶段产出的数据划分、归一化器与 checkpoint，
执行纯数值 rollout 评估，并把“物理量纲主输出 + z-space 辅助输出 +
控制前诊断指标”稳定落盘。

主要功能：
1. 复用训练阶段的 split / scaler artifact，执行确定性的离线评估。
2. 计算全局、group 与 horizon 级 RMSE / MAE 指标，并并行写出 dense / masked artifact。
3. 以物理量纲写出代表性 `pred_samples.npz`，同时保留 `pred_samples_zspace.npz` 供调参与排障。
4. 额外导出 `pred_context.npz`、`pred_sample_manifest.csv` 与 `component_metrics*.csv`，
   供 component/residual 可视化与论文图表复用。
5. 额外写出 `pred_trace.npz`，提供约 50s 的连续时序诊断窗口。
6. 在 `metrics.yaml` 中写出控制前诊断摘要，用于筛查长时漂移、尾部误差与系统偏差。
7. 对经过 scaler 后仍残留在输入 `X` 中的非有限值做最小清洗，与训练消费端保持一致。
8. 额外写出 `long_horizon_fit` 摘要，固化长期 rollout 拟合能力证据。

数据流：
train yaml + checkpoint + run_dir artifacts
    ↓
EvalConfig / S1PredictorConfig
    ↓
加载 features.npz / labels.npz / split_indices.npz / scalers
    ↓
model rollout(z-space) + inverse_transform(physical) + target_mask-aware metric aggregation
    ↓
metrics.yaml(metric_space + layout + supervision + control_readiness) +
long_horizon_fit +
rmse/mae_by_horizon*.csv +
component_metrics*.csv +
pred_samples.npz + pred_samples_zspace.npz + pred_context.npz + pred_sample_manifest.csv +
pred_trace.npz
    ↓
cli/eval.py 或 cli/pipeline.py 再调起 viz 层出图

依赖模块：
- uwnav_dynamics.eval.config
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.models.losses.state_transition

备注：
- 本模块只负责数值评估与 artifact 落盘。
- 正式用户入口为 `cli/eval.py` 与 `cli/pipeline.py`；`evaluate.py` 不直接编排绘图。
- 主指标与主样例产物使用物理量纲；z-space 指标只作为并行辅助信息保留。
- 当前样例产物默认保存代表性窗口，而不是简单截取前 N 个样本。
- `pred_trace.npz` 按原 dataset sample index 选择一个最密集的长时序窗口，
  用于 3 子窗共享 x 轴的实际时序诊断图。
- `control_readiness` 只提供“是否值得进入后续控制验证”的离线筛查信息，
  不能替代真正的闭环控制验证。
- runtime mask 的唯一执行真源是评估 batch 中的 `target_mask`。
- 输入 `X` 中由稀疏辅助通道保留的 NaN 不会回写 dataset artifact，
  仅在评估消费端被清洗为 0.0（对应 z-score 后的 train 均值）。
"""

# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, List, Mapping, Sequence

import numpy as np
import torch
import yaml

from uwnav_dynamics.dataset.normalize import inverse_transform, load_scaler, transform
from uwnav_dynamics.dataset.split import load_split_indices
from uwnav_dynamics.eval.config import EvalConfig, build_eval_config
from uwnav_dynamics.experiment.representative import RepresentativeRule, select_representative_rows
from uwnav_dynamics.experiment.paths import infer_repo_root, relative_path_str, to_snapshot_value
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.models.utils.execution_layout import (
    build_execution_layout_metadata,
    extract_y0_from_x_last,
)
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.losses.state_transition import resolve_late_horizon_start
from uwnav_dynamics.models.utils.semantic_output_layout import (
    SEMANTIC_LAYOUT_SCHEMA_VERSION,
    SemanticOutputLayout,
    build_semantic_layout_metadata,
    resolve_semantic_output_layout,
)
from uwnav_dynamics.supervision_mask import build_dense_target_mask, build_target_mask_from_dvl_mask


SUPERVISION_SCHEMA_VERSION = "supervision_v1"
CONTROL_READINESS_SCHEMA_VERSION = "control_readiness_v1"
LONG_HORIZON_FIT_SCHEMA_VERSION = "long_horizon_fit_v1"
PRIMARY_METRIC_SPACE = "physical"
SECONDARY_METRIC_SPACE = "zspace"

_COMPONENT_DISPLAY = {
    "acc_x": ("Acc X", "m/s^2"),
    "acc_y": ("Acc Y", "m/s^2"),
    "acc_z": ("Acc Z", "m/s^2"),
    "gyro_x": ("Gyro X", "rad/s"),
    "gyro_y": ("Gyro Y", "rad/s"),
    "gyro_z": ("Gyro Z", "rad/s"),
    "vel_x": ("Vel X", "m/s"),
    "vel_y": ("Vel Y", "m/s"),
    "vel_z": ("Vel Z", "m/s"),
}


@dataclass(frozen=True)
class LoadedEvalArtifacts:
    """评估阶段共享的数据 artifact 真源。"""
    X: np.ndarray
    Y: np.ndarray
    target_mask: np.ndarray
    split_indices: Dict[str, np.ndarray]
    x_scaler: Dict[str, Any]
    y_scaler: Dict[str, Any]
    semantic_layout: SemanticOutputLayout
    raw_mask_source: str
    n_total: int


@dataclass(frozen=True)
class SplitEvalArtifacts:
    """单个 split 切片后的评估输入。"""
    X_scaled: np.ndarray
    Y_scaled: np.ndarray
    target_mask: np.ndarray
    sample_index: np.ndarray
    x_scaler: Dict[str, Any]
    y_scaler: Dict[str, Any]
    semantic_layout: SemanticOutputLayout
    raw_mask_source: str
    split_name: str
    n_total: int


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _np_load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _sanitize_scaled_inputs_for_eval(x: np.ndarray) -> np.ndarray:
    """
    对评估阶段经过 scaler 后的输入特征做最小非有限值清洗。

    设计原因：
    - `features.npz` 允许保留稀疏辅助输入的 NaN（例如 power 缺测段）。
    - `transform()` 会保留 NaN；若直接送入模型，预测与指标都会变成 NaN。
    - 训练侧已经在消费端将这些值置为 0.0；评估侧必须保持同一语义。
    """
    bad = ~np.isfinite(x)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return x

    x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    print(
        "[EVAL] sanitized non-finite X: "
        f"replaced {bad_count} values with 0.0 after z-score transform"
    )
    return x_safe.astype(np.float32, copy=False)


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


# =============================================================================
# Metrics
# =============================================================================

def _rmse_mae_by_horizon(y_hat: torch.Tensor, y_true: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    err = (y_hat - y_true)  # (N,H,D)
    mse = torch.mean(err * err, dim=0)        # (H,D)
    rmse = torch.sqrt(mse)                    # (H,D)
    mae = torch.mean(torch.abs(err), dim=0)   # (H,D)
    return rmse.detach().cpu().numpy(), mae.detach().cpu().numpy()


def _rmse_mae_by_horizon_masked(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: expect {tuple(y_hat.shape)}, got {tuple(target_mask.shape)}")
    mask = target_mask.to(device=y_hat.device, dtype=y_hat.dtype)
    valid = mask.sum(dim=0)  # (H,D)

    err = y_hat - y_true
    sq = (err * err) * mask
    ab = torch.abs(err) * mask

    rmse = torch.full_like(valid, float("nan"), dtype=y_hat.dtype)
    mae = torch.full_like(valid, float("nan"), dtype=y_hat.dtype)
    valid_pos = valid > 0
    rmse[valid_pos] = torch.sqrt(sq.sum(dim=0)[valid_pos] / valid[valid_pos])
    mae[valid_pos] = ab.sum(dim=0)[valid_pos] / valid[valid_pos]
    return rmse.detach().cpu().numpy(), mae.detach().cpu().numpy()


def _global_rmse_mae(y_hat: torch.Tensor, y_true: torch.Tensor, target_mask: torch.Tensor | None = None) -> Tuple[float, float]:
    err = y_hat - y_true
    if target_mask is None:
        mse = torch.mean(err * err)
        mae = torch.mean(torch.abs(err))
        return float(torch.sqrt(mse).item()), float(mae.item())

    if target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: expect {tuple(y_hat.shape)}, got {tuple(target_mask.shape)}")
    mask = target_mask.to(device=y_hat.device, dtype=y_hat.dtype)
    valid = mask.sum()
    if float(valid.item()) <= 0.0:
        raise ValueError("target_mask contains zero valid supervision elements during evaluation")
    mse = ((err * err) * mask).sum() / valid
    mae = (torch.abs(err) * mask).sum() / valid
    return float(torch.sqrt(mse).item()), float(mae.item())


def _aggregate_group_curve(metric_hd: np.ndarray, groups: Dict[str, Sequence[int]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key, indices in groups.items():
        group_vals = metric_hd[:, list(indices)]
        valid_count = np.sum(~np.isnan(group_vals), axis=1)
        curve = np.full(group_vals.shape[0], np.nan, dtype=float)
        valid = valid_count > 0
        if np.any(valid):
            curve[valid] = np.nansum(group_vals[valid], axis=1) / valid_count[valid]
        out[key] = curve.tolist()
    return out


def _global_curve_from_hd(metric_hd: np.ndarray) -> np.ndarray:
    valid_count = np.sum(~np.isnan(metric_hd), axis=1)
    curve = np.full(metric_hd.shape[0], np.nan, dtype=np.float64)
    valid = valid_count > 0
    if np.any(valid):
        curve[valid] = np.nansum(metric_hd[valid], axis=1) / valid_count[valid]
    return curve


def _curve_auc(curve: Sequence[float]) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _curve_late_mean(curve: Sequence[float], *, start_idx: int) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    start = max(0, min(int(start_idx), max(int(arr.shape[0]) - 1, 0)))
    finite = arr[start:][np.isfinite(arr[start:])]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _curve_slope(curve: Sequence[float]) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size < 2:
        return float("nan")
    x = finite_idx.astype(np.float64)
    y = arr[finite_idx]
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def _write_csv_hd(path: Path, hd: np.ndarray, col_prefix: str = "d") -> None:
    H, D = hd.shape
    header = ",".join(["h"] + [f"{col_prefix}{i}" for i in range(D)])
    lines = [header]
    for h in range(H):
        row = ",".join([str(h + 1)] + [f"{hd[h, d]:.8f}" for d in range(D)])
        lines.append(row)
    path.write_text("\n".join(lines), encoding="utf-8")


def _component_metric_rows(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    component_labels: Sequence[str],
    component_units: Sequence[str],
    target_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if y_hat.shape != y_true.shape:
        raise ValueError(f"y_hat/y_true shape mismatch: {y_hat.shape} vs {y_true.shape}")
    if target_mask is not None and target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: {target_mask.shape} vs {y_hat.shape}")

    err = np.asarray(y_hat - y_true, dtype=np.float64)
    if target_mask is None:
        valid = np.ones_like(err, dtype=bool)
    else:
        valid = np.asarray(target_mask, dtype=bool)

    rows: list[dict[str, Any]] = []
    for dim, (label, unit) in enumerate(zip(component_labels, component_units)):
        err_d = err[..., dim]
        valid_d = valid[..., dim]
        count = int(np.count_nonzero(valid_d))
        if count <= 0:
            rows.append(
                {
                    "component": str(label),
                    "unit": str(unit),
                    "count": 0,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "bias": float("nan"),
                }
            )
            continue
        vals = err_d[valid_d]
        rows.append(
            {
                "component": str(label),
                "unit": str(unit),
                "count": count,
                "rmse": float(np.sqrt(np.mean(vals * vals))),
                "mae": float(np.mean(np.abs(vals))),
                "bias": float(np.mean(vals)),
            }
        )
    return rows


def _write_component_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = "component,unit,count,rmse,mae,bias"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["component"]),
                    str(row["unit"]),
                    str(int(row["count"])),
                    f"{float(row['rmse']):.8f}",
                    f"{float(row['mae']):.8f}",
                    f"{float(row['bias']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sample_manifest_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = (
        "sample_slot,dataset_sample_index,representative_tag,representative_metric,representative_mode,"
        "representative_value,valid_value_count,rmse_global,mae_global,final_step_rmse,final_step_mae,tail_abs_p95"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["sample_slot"])),
                    str(int(row["dataset_sample_index"])),
                    str(row["representative_tag"]),
                    str(row["representative_metric"]),
                    str(row["representative_mode"]),
                    f"{float(row['representative_value']):.8f}",
                    str(int(row["valid_value_count"])),
                    f"{float(row['rmse_global']):.8f}",
                    f"{float(row['mae_global']):.8f}",
                    f"{float(row['final_step_rmse']):.8f}",
                    f"{float(row['final_step_mae']):.8f}",
                    f"{float(row['tail_abs_p95']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _global_rmse_mae_np(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    if y_hat.shape != y_true.shape:
        raise ValueError(f"y_hat/y_true shape mismatch: {y_hat.shape} vs {y_true.shape}")
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    if target_mask is None:
        vals = err.reshape(-1)
    else:
        valid = np.asarray(target_mask, dtype=bool)
        if valid.shape != err.shape:
            raise ValueError(f"target_mask shape mismatch: {valid.shape} vs {err.shape}")
        vals = err[valid]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(vals * vals))), float(np.mean(np.abs(vals)))


def _build_sample_metric_rows(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    target_mask: np.ndarray,
    dataset_sample_index: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_samples = int(y_hat.shape[0])
    for sample_slot in range(n_samples):
        y_hat_i = np.asarray(y_hat[sample_slot:sample_slot + 1], dtype=np.float64)
        y_true_i = np.asarray(y_true[sample_slot:sample_slot + 1], dtype=np.float64)
        mask_i = np.asarray(target_mask[sample_slot:sample_slot + 1], dtype=bool)
        rmse_i, mae_i = _global_rmse_mae_np(y_hat_i, y_true_i, target_mask=mask_i)
        final_rmse_i, final_mae_i = _global_rmse_mae_np(
            y_hat_i[:, -1:, :],
            y_true_i[:, -1:, :],
            target_mask=mask_i[:, -1:, :],
        )
        tail_i = _abs_error_percentiles(y_hat_i, y_true_i, target_mask=mask_i, percentiles=(95.0,))
        rows.append(
            {
                "sample_slot": int(sample_slot),
                "dataset_sample_index": int(dataset_sample_index[sample_slot]),
                "valid_value_count": int(np.count_nonzero(mask_i)),
                "rmse_global": float(rmse_i),
                "mae_global": float(mae_i),
                "final_step_rmse": float(final_rmse_i),
                "final_step_mae": float(final_mae_i),
                "tail_abs_p95": float(tail_i["p95"]),
            }
        )
    return rows


def _select_representative_sample_rows(
    *,
    rows: Sequence[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    return select_representative_rows(
        rows,
        id_key="sample_slot",
        max_items=max_items,
        rules=(
            RepresentativeRule(tag="best_rmse", metric="rmse_global", mode="min"),
            RepresentativeRule(tag="median_rmse", metric="rmse_global", mode="median"),
            RepresentativeRule(tag="worst_rmse", metric="rmse_global", mode="max"),
            RepresentativeRule(tag="worst_final_step", metric="final_step_rmse", mode="max"),
            RepresentativeRule(tag="worst_tail_p95", metric="tail_abs_p95", mode="max"),
        ),
    )


def _select_trace_slots(
    *,
    dataset_sample_index: np.ndarray,
    trace_seconds: float,
    trace_dt_s: float,
) -> np.ndarray:
    """选择一个按原始样本索引计约 50s 的最密集诊断窗口。"""
    sample_index = np.asarray(dataset_sample_index, dtype=np.int64)
    if sample_index.ndim != 1:
        raise ValueError(f"dataset_sample_index must be 1D, got {sample_index.shape}")
    if sample_index.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    sorted_slots = np.argsort(sample_index, kind="stable")
    sorted_index = sample_index[sorted_slots]
    dt = max(float(trace_dt_s), np.finfo(float).eps)
    window_steps = max(1, int(round(max(float(trace_seconds), dt) / dt)))

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

    return np.asarray(sorted_slots[best_start:best_stop], dtype=np.int64)


def _build_prediction_trace_artifact(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    logvar: np.ndarray,
    target_mask: np.ndarray,
    dataset_sample_index: np.ndarray,
    component_labels: Sequence[str],
    component_display_labels: Sequence[str],
    component_units: Sequence[str],
    trace_seconds: float,
    trace_dt_s: float,
) -> dict[str, np.ndarray | float | int]:
    """构建长时序预测诊断 artifact，供 3 子窗共享 x 轴绘图消费。"""
    if y_hat.shape != y_true.shape or y_hat.shape != logvar.shape or y_hat.shape != target_mask.shape:
        raise ValueError(
            "trace inputs must share shape, got "
            f"y_hat={y_hat.shape}, y_true={y_true.shape}, logvar={logvar.shape}, mask={target_mask.shape}"
        )
    if y_hat.ndim != 3:
        raise ValueError(f"trace inputs must be (N,H,D), got {y_hat.shape}")

    slots = _select_trace_slots(
        dataset_sample_index=np.asarray(dataset_sample_index, dtype=np.int64),
        trace_seconds=float(trace_seconds),
        trace_dt_s=float(trace_dt_s),
    )
    horizon_index = int(y_hat.shape[1] - 1)
    selected_index = np.asarray(dataset_sample_index, dtype=np.int64)[slots]
    if selected_index.shape[0] > 0:
        t_s = (selected_index.astype(np.float64) - float(selected_index[0])) * float(trace_dt_s)
    else:
        t_s = np.zeros((0,), dtype=np.float64)

    return {
        "y_hat": np.asarray(y_hat[slots, horizon_index, :], dtype=np.float32),
        "y_true": np.asarray(y_true[slots, horizon_index, :], dtype=np.float32),
        "logvar": np.asarray(logvar[slots, horizon_index, :], dtype=np.float32),
        "target_mask": np.asarray(target_mask[slots, horizon_index, :], dtype=bool),
        "sample_index": np.asarray(selected_index, dtype=np.int64),
        "t_s": np.asarray(t_s, dtype=np.float64),
        "component_labels": np.asarray(component_labels, dtype=str),
        "component_display_labels": np.asarray(component_display_labels, dtype=str),
        "component_units": np.asarray(component_units, dtype=str),
        "horizon_index": int(horizon_index),
        "dt_s": float(trace_dt_s),
        "requested_seconds": float(trace_seconds),
    }


def _abs_error_percentiles(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    target_mask: np.ndarray | None = None,
    percentiles: Sequence[float] = (95.0, 99.0),
) -> dict[str, float]:
    if y_hat.shape != y_true.shape:
        raise ValueError(f"y_hat/y_true shape mismatch: {y_hat.shape} vs {y_true.shape}")
    abs_err = np.abs(np.asarray(y_hat - y_true, dtype=np.float64))
    if target_mask is None:
        vals = abs_err.reshape(-1)
    else:
        valid = np.asarray(target_mask, dtype=bool)
        if valid.shape != abs_err.shape:
            raise ValueError(f"target_mask shape mismatch: {valid.shape} vs {abs_err.shape}")
        vals = abs_err[valid]

    out: dict[str, float] = {}
    for q in percentiles:
        key = f"p{int(q)}"
        out[key] = float(np.percentile(vals, q)) if vals.size > 0 else float("nan")
    return out


def _last_finite_value(curve: Sequence[float]) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return float("nan")
    return float(arr[finite_idx[-1]])


def _edge_ratio(curve: Sequence[float]) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return float("nan")
    first = float(arr[finite_idx[0]])
    last = float(arr[finite_idx[-1]])
    eps = 1e-12
    if abs(first) <= eps:
        return 1.0 if abs(last) <= eps else float("inf")
    return float(last / first)


def _worst_component_abs_bias(rows: Sequence[dict[str, Any]]) -> tuple[str | None, float]:
    best_label: str | None = None
    best_abs_bias = float("nan")
    for row in rows:
        bias = float(row["bias"])
        if not np.isfinite(bias):
            continue
        abs_bias = abs(bias)
        if best_label is None or abs_bias > best_abs_bias:
            best_label = str(row["component"])
            best_abs_bias = float(abs_bias)
    return best_label, best_abs_bias


def _build_control_readiness_summary(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    rmse_groups: Mapping[str, Sequence[float]],
    mae_groups: Mapping[str, Sequence[float]],
    component_metrics: Sequence[dict[str, Any]],
    target_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    target_mask_first = None if target_mask is None else np.asarray(target_mask[:, :1, :], dtype=bool)
    target_mask_last = None if target_mask is None else np.asarray(target_mask[:, -1:, :], dtype=bool)
    first_rmse, first_mae = _global_rmse_mae_np(
        y_hat[:, :1, :],
        y_true[:, :1, :],
        target_mask=target_mask_first,
    )
    final_rmse, final_mae = _global_rmse_mae_np(
        y_hat[:, -1:, :],
        y_true[:, -1:, :],
        target_mask=target_mask_last,
    )
    overall_tail = _abs_error_percentiles(y_hat, y_true, target_mask=target_mask, percentiles=(95.0, 99.0))
    final_tail = _abs_error_percentiles(
        y_hat[:, -1:, :],
        y_true[:, -1:, :],
        target_mask=target_mask_last,
        percentiles=(95.0,),
    )
    worst_component_label, worst_abs_bias = _worst_component_abs_bias(component_metrics)
    rmse_last_over_first = _edge_ratio([first_rmse, final_rmse])
    mae_last_over_first = _edge_ratio([first_mae, final_mae])

    return {
        "final_step": {
            "rmse_global": final_rmse,
            "mae_global": final_mae,
            "group_rmse": {key: _last_finite_value(curve) for key, curve in rmse_groups.items()},
            "group_mae": {key: _last_finite_value(curve) for key, curve in mae_groups.items()},
        },
        "rollout_growth": {
            "rmse_last_over_first": rmse_last_over_first,
            "mae_last_over_first": mae_last_over_first,
            "group_rmse_last_over_first": {key: _edge_ratio(curve) for key, curve in rmse_groups.items()},
            "group_mae_last_over_first": {key: _edge_ratio(curve) for key, curve in mae_groups.items()},
        },
        "tail_error": {
            "abs_p95_global": overall_tail["p95"],
            "abs_p99_global": overall_tail["p99"],
            "final_step_abs_p95_global": final_tail["p95"],
        },
        "bias": {
            "worst_component": worst_component_label,
            "worst_abs_bias": worst_abs_bias,
        },
    }


def _build_long_horizon_fit_summary(
    *,
    rmse_hd: np.ndarray,
    mae_hd: np.ndarray,
    final_step: Mapping[str, Any],
    rollout_growth: Mapping[str, Any],
    late_horizon_fraction: float,
) -> dict[str, Any]:
    rmse_curve = _global_curve_from_hd(rmse_hd)
    mae_curve = _global_curve_from_hd(mae_hd)
    late_start = resolve_late_horizon_start(
        rmse_hd.shape[0],
        late_horizon_fraction=float(late_horizon_fraction),
    )
    return {
        "late_horizon_fraction": float(late_horizon_fraction),
        "late_horizon_start_step": int(late_start + 1),
        "rmse_auc_global": _curve_auc(rmse_curve),
        "mae_auc_global": _curve_auc(mae_curve),
        "late_horizon_rmse_global_mean": _curve_late_mean(rmse_curve, start_idx=late_start),
        "late_horizon_mae_global_mean": _curve_late_mean(mae_curve, start_idx=late_start),
        "rmse_step_slope": _curve_slope(rmse_curve),
        "mae_step_slope": _curve_slope(mae_curve),
        "final_step_rmse_global": float(final_step.get("rmse_global", float("nan"))),
        "final_step_mae_global": float(final_step.get("mae_global", float("nan"))),
        "rmse_last_over_first": float(rollout_growth.get("rmse_last_over_first", float("nan"))),
        "mae_last_over_first": float(rollout_growth.get("mae_last_over_first", float("nan"))),
    }


def _component_metadata(semantic_layout: SemanticOutputLayout) -> tuple[list[str], list[str], list[str]]:
    display_labels: list[str] = []
    unit_labels: list[str] = []
    raw_labels: list[str] = []
    for label in semantic_layout.component_labels:
        raw = str(label)
        raw_labels.append(raw)
        display, unit = _COMPONENT_DISPLAY.get(raw, (raw, "a.u."))
        display_labels.append(display)
        unit_labels.append(unit)
    return raw_labels, display_labels, unit_labels


def _convert_logvar_to_physical(logvar_z: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(logvar_z, dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if arr.shape[-1] != std.shape[0]:
        raise ValueError(f"logvar/scaler std mismatch: {arr.shape[-1]} vs {std.shape[0]}")
    safe_std = np.maximum(std, 1e-12)
    view_shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    return arr + 2.0 * np.log(safe_std.reshape(view_shape))


def _extract_target_cols(label_npz: Dict[str, Any]) -> tuple[str, ...] | None:
    if "target_cols" not in label_npz:
        return None
    target_cols = np.asarray(label_npz["target_cols"])
    if target_cols.ndim == 0:
        return (str(target_cols.item()),)
    return tuple(str(x) for x in target_cols.tolist())


def _resolve_semantic_layout(label_npz: Dict[str, Any], dout: int) -> SemanticOutputLayout:
    target_cols = _extract_target_cols(label_npz)
    return resolve_semantic_output_layout(dout=dout, target_cols=target_cols)


def _build_eval_target_mask(
    label_npz: Dict[str, Any],
    *,
    semantic_layout: SemanticOutputLayout,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    if "dvl_mask" not in label_npz:
        return build_dense_target_mask(target_shape), "implicit_all_true"
    return (
        build_target_mask_from_dvl_mask(
            np.asarray(label_npz["dvl_mask"], dtype=bool),
            semantic_layout,
            target_shape=target_shape,
        ),
        "dvl_mask",
    )


def load_eval_artifacts(*, cfg_eval: EvalConfig, cfg_model: S1PredictorConfig) -> LoadedEvalArtifacts:
    """加载评估所需的 dataset / split / scaler 真源，但不执行模型前向。"""
    feat = _np_load_npz(cfg_eval.data_dir / "features.npz")
    lab = _np_load_npz(cfg_eval.data_dir / "labels.npz")
    X = feat["X"]  # (N,L,Din)
    Y = lab["Y"]   # (N,H,Dout)
    mask_keys = [k for k in ("dvl_mask_hist", "power_mask_hist") if k in feat]
    mask_keys += [k for k in ("dvl_mask", "power_mask") if k in lab]
    if mask_keys:
        print(f"[EVAL] found mask tensors: {mask_keys}")

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expect X/Y to be 3D arrays, got X={X.shape}, Y={Y.shape}")

    n, _L, Din = X.shape
    n2, H, Dout = Y.shape
    if n2 != n:
        raise ValueError(f"X and Y window counts mismatch: {n} vs {n2}")
    if Din != cfg_model.din:
        raise ValueError(f"Din mismatch: data Din={Din}, cfg_model.din={cfg_model.din}")
    if Dout != cfg_model.dout or H != cfg_model.pred_len:
        raise ValueError(f"Y shape mismatch: data (H,D)={(H,Dout)} vs cfg {(cfg_model.pred_len,cfg_model.dout)}")

    semantic_layout = _resolve_semantic_layout(lab, Dout)
    target_mask_all_np, raw_mask_source = _build_eval_target_mask(
        lab,
        semantic_layout=semantic_layout,
        target_shape=(n, H, Dout),
    )

    split_indices = load_split_indices(cfg_eval.split_indices_path)
    x_scaler = load_scaler(cfg_eval.x_scaler_path)
    y_scaler = load_scaler(cfg_eval.y_scaler_path)
    return LoadedEvalArtifacts(
        X=X,
        Y=Y,
        target_mask=np.asarray(target_mask_all_np, dtype=bool),
        split_indices=split_indices,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        semantic_layout=semantic_layout,
        raw_mask_source=raw_mask_source,
        n_total=int(n),
    )


def slice_eval_artifacts(loaded: LoadedEvalArtifacts, *, split_name: str) -> SplitEvalArtifacts:
    """把共享 artifact 切成某个 split 的实际评估输入。"""
    if split_name not in loaded.split_indices:
        raise KeyError(f"split '{split_name}' not found in split artifact")
    idx = np.asarray(loaded.split_indices[split_name], dtype=np.int64)
    if idx.size == 0:
        raise RuntimeError(f"Split {split_name} is empty. Check ratios.")
    if np.any(idx < 0) or np.any(idx >= loaded.n_total):
        raise ValueError(f"Split indices out of range for n={loaded.n_total}")

    X_scaled = transform(np.array(loaded.X[idx], copy=True), loaded.x_scaler).astype(np.float32, copy=False)
    X_scaled = _sanitize_scaled_inputs_for_eval(X_scaled)
    Y_scaled = transform(np.array(loaded.Y[idx], copy=True), loaded.y_scaler).astype(np.float32, copy=False)
    target_mask = np.asarray(loaded.target_mask[idx], dtype=bool)
    return SplitEvalArtifacts(
        X_scaled=X_scaled,
        Y_scaled=Y_scaled,
        target_mask=target_mask,
        sample_index=idx,
        x_scaler=loaded.x_scaler,
        y_scaler=loaded.y_scaler,
        semantic_layout=loaded.semantic_layout,
        raw_mask_source=loaded.raw_mask_source,
        split_name=str(split_name),
        n_total=int(loaded.n_total),
    )


def summarize_eval_predictions(
    *,
    split_artifacts: SplitEvalArtifacts,
    y_hat_z_np: np.ndarray,
    y_true_z_np: np.ndarray,
    logvar_z_np: np.ndarray,
    save_samples: int,
    late_horizon_fraction: float,
    trace_seconds: float,
    trace_dt_s: float,
) -> Dict[str, Any]:
    """把 z-space 预测结果统一汇总成指标、CSV/NPZ 所需 artifact 内容。"""
    if y_hat_z_np.shape != y_true_z_np.shape or y_hat_z_np.shape != logvar_z_np.shape:
        raise ValueError(
            "prediction arrays must share the same shape, got "
            f"y_hat={y_hat_z_np.shape}, y_true={y_true_z_np.shape}, logvar={logvar_z_np.shape}"
        )
    if split_artifacts.target_mask.shape != y_hat_z_np.shape:
        raise ValueError(
            f"target_mask shape mismatch: {split_artifacts.target_mask.shape} vs {y_hat_z_np.shape}"
        )

    target_mask_np_eval = np.asarray(split_artifacts.target_mask, dtype=bool)

    # 物理量纲指标是主输出；z-space 指标只作为训练/调参的并行辅助信息保留。
    y_hat_phys_np = inverse_transform(y_hat_z_np, split_artifacts.y_scaler).astype(np.float32, copy=False)
    y_true_phys_np = inverse_transform(y_true_z_np, split_artifacts.y_scaler).astype(np.float32, copy=False)
    logvar_phys_np = _convert_logvar_to_physical(logvar_z_np, split_artifacts.y_scaler).astype(np.float32, copy=False)

    y_hat_phys = torch.from_numpy(y_hat_phys_np)
    y_true_phys = torch.from_numpy(y_true_phys_np)
    y_hat_z = torch.from_numpy(y_hat_z_np)
    y_true_z = torch.from_numpy(y_true_z_np)
    target_mask_t = torch.from_numpy(target_mask_np_eval)

    rmse_hd, mae_hd = _rmse_mae_by_horizon(y_hat_phys, y_true_phys)
    rmse_hd_masked, mae_hd_masked = _rmse_mae_by_horizon_masked(y_hat_phys, y_true_phys, target_mask_t)
    rmse_groups = _aggregate_group_curve(rmse_hd, split_artifacts.semantic_layout.group_indices)
    mae_groups = _aggregate_group_curve(mae_hd, split_artifacts.semantic_layout.group_indices)
    rmse_groups_masked = _aggregate_group_curve(rmse_hd_masked, split_artifacts.semantic_layout.group_indices)
    mae_groups_masked = _aggregate_group_curve(mae_hd_masked, split_artifacts.semantic_layout.group_indices)
    rmse_global, mae_global = _global_rmse_mae(y_hat_phys, y_true_phys)
    rmse_global_masked, mae_global_masked = _global_rmse_mae(y_hat_phys, y_true_phys, target_mask_t)

    rmse_hd_z, mae_hd_z = _rmse_mae_by_horizon(y_hat_z, y_true_z)
    rmse_hd_masked_z, mae_hd_masked_z = _rmse_mae_by_horizon_masked(y_hat_z, y_true_z, target_mask_t)
    rmse_groups_z = _aggregate_group_curve(rmse_hd_z, split_artifacts.semantic_layout.group_indices)
    mae_groups_z = _aggregate_group_curve(mae_hd_z, split_artifacts.semantic_layout.group_indices)
    rmse_groups_masked_z = _aggregate_group_curve(rmse_hd_masked_z, split_artifacts.semantic_layout.group_indices)
    mae_groups_masked_z = _aggregate_group_curve(mae_hd_masked_z, split_artifacts.semantic_layout.group_indices)
    rmse_global_z, mae_global_z = _global_rmse_mae(y_hat_z, y_true_z)
    rmse_global_masked_z, mae_global_masked_z = _global_rmse_mae(y_hat_z, y_true_z, target_mask_t)

    component_labels, component_display_labels, component_units = _component_metadata(split_artifacts.semantic_layout)
    component_metrics = _component_metric_rows(
        y_hat_phys_np,
        y_true_phys_np,
        component_labels=component_labels,
        component_units=component_units,
        target_mask=None,
    )
    component_metrics_masked = _component_metric_rows(
        y_hat_phys_np,
        y_true_phys_np,
        component_labels=component_labels,
        component_units=component_units,
        target_mask=target_mask_np_eval,
    )
    component_metrics_z = _component_metric_rows(
        y_hat_z_np,
        y_true_z_np,
        component_labels=component_labels,
        component_units=["z-score"] * len(component_labels),
        target_mask=None,
    )
    component_metrics_masked_z = _component_metric_rows(
        y_hat_z_np,
        y_true_z_np,
        component_labels=component_labels,
        component_units=["z-score"] * len(component_labels),
        target_mask=target_mask_np_eval,
    )
    control_readiness_dense = _build_control_readiness_summary(
        y_hat=y_hat_phys_np,
        y_true=y_true_phys_np,
        rmse_groups=rmse_groups,
        mae_groups=mae_groups,
        component_metrics=component_metrics,
        target_mask=None,
    )
    control_readiness_masked = _build_control_readiness_summary(
        y_hat=y_hat_phys_np,
        y_true=y_true_phys_np,
        rmse_groups=rmse_groups_masked,
        mae_groups=mae_groups_masked,
        component_metrics=component_metrics_masked,
        target_mask=target_mask_np_eval,
    )
    long_horizon_dense = _build_long_horizon_fit_summary(
        rmse_hd=rmse_hd,
        mae_hd=mae_hd,
        final_step=control_readiness_dense["final_step"],
        rollout_growth=control_readiness_dense["rollout_growth"],
        late_horizon_fraction=float(late_horizon_fraction),
    )
    long_horizon_masked = _build_long_horizon_fit_summary(
        rmse_hd=rmse_hd_masked,
        mae_hd=mae_hd_masked,
        final_step=control_readiness_masked["final_step"],
        rollout_growth=control_readiness_masked["rollout_growth"],
        late_horizon_fraction=float(late_horizon_fraction),
    )

    sample_metric_rows = _build_sample_metric_rows(
        y_hat=y_hat_phys_np,
        y_true=y_true_phys_np,
        target_mask=target_mask_np_eval,
        dataset_sample_index=np.asarray(split_artifacts.sample_index, dtype=np.int64),
    )
    representative_rows = _select_representative_sample_rows(
        rows=sample_metric_rows,
        max_items=int(min(save_samples, y_hat_z_np.shape[0])),
    )
    selected_slots = np.asarray([int(row["sample_slot"]) for row in representative_rows], dtype=np.int64)
    n_samp = int(selected_slots.shape[0])
    samp = {
        "y_hat": y_hat_phys_np[selected_slots],
        "y_true": y_true_phys_np[selected_slots],
        "logvar": logvar_phys_np[selected_slots],
    }
    samp_z = {
        "y_hat": y_hat_z_np[selected_slots],
        "y_true": y_true_z_np[selected_slots],
        "logvar": logvar_z_np[selected_slots],
    }
    pred_context = {
        "target_mask": target_mask_np_eval[selected_slots].astype(bool, copy=False),
        "sample_index": np.asarray(split_artifacts.sample_index[selected_slots], dtype=np.int64),
        "component_labels": np.asarray(component_labels, dtype=str),
        "component_display_labels": np.asarray(component_display_labels, dtype=str),
        "component_units": np.asarray(component_units, dtype=str),
        "sample_tag": np.asarray([str(row["representative_tag"]) for row in representative_rows], dtype=str),
    }
    pred_trace = _build_prediction_trace_artifact(
        y_hat=y_hat_phys_np,
        y_true=y_true_phys_np,
        logvar=logvar_phys_np,
        target_mask=target_mask_np_eval,
        dataset_sample_index=np.asarray(split_artifacts.sample_index, dtype=np.int64),
        component_labels=component_labels,
        component_display_labels=component_display_labels,
        component_units=component_units,
        trace_seconds=float(trace_seconds),
        trace_dt_s=float(trace_dt_s),
    )

    return {
        "n_total": int(split_artifacts.n_total),
        "n_eval": int(y_hat_z_np.shape[0]),
        "split": split_artifacts.split_name,
        "rmse_global": rmse_global,
        "mae_global": mae_global,
        "rmse_global_masked": rmse_global_masked,
        "mae_global_masked": mae_global_masked,
        "rmse_hd": rmse_hd,
        "mae_hd": mae_hd,
        "rmse_hd_masked": rmse_hd_masked,
        "mae_hd_masked": mae_hd_masked,
        "rmse_groups": rmse_groups,
        "mae_groups": mae_groups,
        "rmse_groups_masked": rmse_groups_masked,
        "mae_groups_masked": mae_groups_masked,
        "rmse_global_zspace": rmse_global_z,
        "mae_global_zspace": mae_global_z,
        "rmse_global_masked_zspace": rmse_global_masked_z,
        "mae_global_masked_zspace": mae_global_masked_z,
        "rmse_hd_zspace": rmse_hd_z,
        "mae_hd_zspace": mae_hd_z,
        "rmse_hd_masked_zspace": rmse_hd_masked_z,
        "mae_hd_masked_zspace": mae_hd_masked_z,
        "rmse_groups_zspace": rmse_groups_z,
        "mae_groups_zspace": mae_groups_z,
        "rmse_groups_masked_zspace": rmse_groups_masked_z,
        "mae_groups_masked_zspace": mae_groups_masked_z,
        "component_metrics": component_metrics,
        "component_metrics_masked": component_metrics_masked,
        "component_metrics_zspace": component_metrics_z,
        "component_metrics_masked_zspace": component_metrics_masked_z,
        "control_readiness_dense": control_readiness_dense,
        "control_readiness_masked": control_readiness_masked,
        "long_horizon_fit_dense": long_horizon_dense,
        "long_horizon_fit_masked": long_horizon_masked,
        "semantic_layout": split_artifacts.semantic_layout,
        "raw_mask_source": split_artifacts.raw_mask_source,
        "samples": samp,
        "samples_zspace": samp_z,
        "pred_context": pred_context,
        "pred_trace": pred_trace,
        "sample_manifest_rows": representative_rows,
    }


def write_eval_outputs(
    *,
    out_dir: Path,
    res: Mapping[str, Any],
    cfg_model: S1PredictorConfig,
    cfg_snapshot: Mapping[str, Any],
    path_root: Path,
) -> None:
    """把汇总后的评估结果统一写成 metrics/csv/npz artifact。"""
    _ensure_dir(out_dir)
    metrics = {
        "split": res["split"],
        "n_total": res["n_total"],
        "n_eval": res["n_eval"],
        "metric_space": {
            "primary": PRIMARY_METRIC_SPACE,
            "auxiliary": SECONDARY_METRIC_SPACE,
        },
        "rmse_global": res["rmse_global"],
        "mae_global": res["mae_global"],
        "rmse_global_masked": res["rmse_global_masked"],
        "mae_global_masked": res["mae_global_masked"],
        "rmse_groups": res["rmse_groups"],
        "mae_groups": res["mae_groups"],
        "rmse_groups_masked": res["rmse_groups_masked"],
        "mae_groups_masked": res["mae_groups_masked"],
        "rmse_global_zspace": res["rmse_global_zspace"],
        "mae_global_zspace": res["mae_global_zspace"],
        "rmse_global_masked_zspace": res["rmse_global_masked_zspace"],
        "mae_global_masked_zspace": res["mae_global_masked_zspace"],
        "rmse_groups_zspace": res["rmse_groups_zspace"],
        "mae_groups_zspace": res["mae_groups_zspace"],
        "rmse_groups_masked_zspace": res["rmse_groups_masked_zspace"],
        "mae_groups_masked_zspace": res["mae_groups_masked_zspace"],
        "component_metrics": {
            "physical": {
                "dense": res["component_metrics"],
                "masked": res["component_metrics_masked"],
            },
            "zspace": {
                "dense": res["component_metrics_zspace"],
                "masked": res["component_metrics_masked_zspace"],
            },
        },
        "layout": {
            "schema_version": SEMANTIC_LAYOUT_SCHEMA_VERSION,
            "execution": build_execution_layout_metadata(cfg_model.y_in_idx),
            "semantic": build_semantic_layout_metadata(res["semantic_layout"]),
        },
        "supervision": {
            "schema_version": SUPERVISION_SCHEMA_VERSION,
            "dense_metrics": {
                "present": True,
            },
            "masked_metrics": {
                "present": True,
                "mask_name": "target_mask",
                "raw_mask_source": res["raw_mask_source"],
                "applies_to_groups": ["vel"],
                "group_source": res["semantic_layout"].source,
                "horizon_files": {
                    "rmse": "rmse_by_horizon_masked.csv",
                    "mae": "mae_by_horizon_masked.csv",
                },
            },
        },
        "control_readiness": {
            "schema_version": CONTROL_READINESS_SCHEMA_VERSION,
            "intended_use": "offline_screening_for_control",
            "closed_loop_proof": False,
            "physical": {
                "dense": res["control_readiness_dense"],
                "masked": res["control_readiness_masked"],
            },
        },
        "long_horizon_fit": {
            "schema_version": LONG_HORIZON_FIT_SCHEMA_VERSION,
            "intended_use": "offline_long_horizon_rollout_audit",
            "closed_loop_proof": False,
            "physical": {
                "dense": res["long_horizon_fit_dense"],
                "masked": res["long_horizon_fit_masked"],
            },
        },
        "artifacts": {
            "primary_samples": "pred_samples.npz",
            "auxiliary_samples_zspace": "pred_samples_zspace.npz",
            "sample_context": "pred_context.npz",
            "sample_manifest_csv": "pred_sample_manifest.csv",
            "prediction_trace_npz": "pred_trace.npz",
            "component_metrics_csv": "component_metrics.csv",
            "component_metrics_masked_csv": "component_metrics_masked.csv",
            "component_metrics_zspace_csv": "component_metrics_zspace.csv",
            "component_metrics_masked_zspace_csv": "component_metrics_masked_zspace.csv",
            "rmse_horizon_zspace_csv": "rmse_by_horizon_zspace.csv",
            "mae_horizon_zspace_csv": "mae_by_horizon_zspace.csv",
            "rmse_horizon_masked_zspace_csv": "rmse_by_horizon_masked_zspace.csv",
            "mae_horizon_masked_zspace_csv": "mae_by_horizon_masked_zspace.csv",
            "long_horizon_plot": "plots/long_horizon_fit_summary.png",
            "resolved_eval_yaml": "resolved_eval.yaml",
        },
        "cfg": to_snapshot_value(dict(cfg_snapshot), base_dir=path_root),
    }
    with open(out_dir / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)

    resolved_eval = {
        "schema_version": "eval_resolved_v1",
        "out_dir": relative_path_str(out_dir, base_dir=path_root),
        "cfg": to_snapshot_value(dict(cfg_snapshot), base_dir=path_root),
    }
    with open(out_dir / "resolved_eval.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_eval, f, sort_keys=False, allow_unicode=True)

    _write_csv_hd(out_dir / "rmse_by_horizon.csv", res["rmse_hd"], col_prefix="d")
    _write_csv_hd(out_dir / "mae_by_horizon.csv", res["mae_hd"], col_prefix="d")
    _write_csv_hd(out_dir / "rmse_by_horizon_masked.csv", res["rmse_hd_masked"], col_prefix="d")
    _write_csv_hd(out_dir / "mae_by_horizon_masked.csv", res["mae_hd_masked"], col_prefix="d")
    _write_csv_hd(out_dir / "rmse_by_horizon_zspace.csv", res["rmse_hd_zspace"], col_prefix="d")
    _write_csv_hd(out_dir / "mae_by_horizon_zspace.csv", res["mae_hd_zspace"], col_prefix="d")
    _write_csv_hd(out_dir / "rmse_by_horizon_masked_zspace.csv", res["rmse_hd_masked_zspace"], col_prefix="d")
    _write_csv_hd(out_dir / "mae_by_horizon_masked_zspace.csv", res["mae_hd_masked_zspace"], col_prefix="d")
    _write_component_metric_csv(out_dir / "component_metrics.csv", res["component_metrics"])
    _write_component_metric_csv(out_dir / "component_metrics_masked.csv", res["component_metrics_masked"])
    _write_component_metric_csv(out_dir / "component_metrics_zspace.csv", res["component_metrics_zspace"])
    _write_component_metric_csv(out_dir / "component_metrics_masked_zspace.csv", res["component_metrics_masked_zspace"])

    pred_npz = out_dir / "pred_samples.npz"
    np.savez_compressed(
        pred_npz,
        y_hat=res["samples"]["y_hat"],
        y_true=res["samples"]["y_true"],
        logvar=res["samples"]["logvar"],
    )
    pred_npz_z = out_dir / "pred_samples_zspace.npz"
    np.savez_compressed(
        pred_npz_z,
        y_hat=res["samples_zspace"]["y_hat"],
        y_true=res["samples_zspace"]["y_true"],
        logvar=res["samples_zspace"]["logvar"],
    )
    pred_context_npz = out_dir / "pred_context.npz"
    np.savez_compressed(
        pred_context_npz,
        target_mask=res["pred_context"]["target_mask"],
        sample_index=res["pred_context"]["sample_index"],
        component_labels=res["pred_context"]["component_labels"],
        component_display_labels=res["pred_context"]["component_display_labels"],
        component_units=res["pred_context"]["component_units"],
        sample_tag=res["pred_context"]["sample_tag"],
    )
    pred_trace_npz = out_dir / "pred_trace.npz"
    np.savez_compressed(
        pred_trace_npz,
        y_hat=res["pred_trace"]["y_hat"],
        y_true=res["pred_trace"]["y_true"],
        logvar=res["pred_trace"]["logvar"],
        target_mask=res["pred_trace"]["target_mask"],
        sample_index=res["pred_trace"]["sample_index"],
        t_s=res["pred_trace"]["t_s"],
        component_labels=res["pred_trace"]["component_labels"],
        component_display_labels=res["pred_trace"]["component_display_labels"],
        component_units=res["pred_trace"]["component_units"],
        horizon_index=res["pred_trace"]["horizon_index"],
        dt_s=res["pred_trace"]["dt_s"],
        requested_seconds=res["pred_trace"]["requested_seconds"],
    )
    _write_sample_manifest_csv(out_dir / "pred_sample_manifest.csv", res["sample_manifest_rows"])


# =============================================================================
# Main evaluation
# =============================================================================

@torch.no_grad()
def evaluate_once(*, cfg_eval: EvalConfig, cfg_model: S1PredictorConfig) -> Dict[str, Any]:
    """执行一次完整数值评估并返回待落盘的指标与样本 artifact。"""
    device = torch.device(cfg_eval.device)
    loaded = load_eval_artifacts(cfg_eval=cfg_eval, cfg_model=cfg_model)
    split_artifacts = slice_eval_artifacts(loaded, split_name=cfg_eval.split_name)

    model = S1Predictor(cfg_model).to(device)
    ckpt = _load_checkpoint(cfg_eval.ckpt, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    bs = int(cfg_eval.batch_size)
    n_eval = split_artifacts.X_scaled.shape[0]

    yhat_list: List[torch.Tensor] = []
    ytrue_list: List[torch.Tensor] = []
    logvar_list: List[torch.Tensor] = []

    # 批量 rollout 只负责“前向 + y0 恢复 + delta 累积”，
    # 所有指标都放到循环外统一计算，避免 batch 粒度聚合误差。
    for s in range(0, n_eval, bs):
        e = min(n_eval, s + bs)
        xb = torch.from_numpy(split_artifacts.X_scaled[s:e]).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(split_artifacts.Y_scaled[s:e]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(split_artifacts.target_mask[s:e]).to(device=device, dtype=torch.bool)

        dY, logvar = model(xb)

        if cfg_eval.mode != "delta_cumsum":
            raise NotImplementedError(f"Only mode='delta_cumsum' supported, got {cfg_eval.mode!r}")
        if cfg_eval.y0_source != "x_last_state":
            raise NotImplementedError(f"Only y0_source='x_last_state' supported, got {cfg_eval.y0_source!r}")
        y0 = extract_y0_from_x_last(xb, cfg_model.y_in_idx)
        y_hat = rollout_from_delta(y0, dY)

        yhat_list.append(y_hat.cpu())
        ytrue_list.append(yb.cpu())
        logvar_list.append(logvar.cpu())

    y_hat_all = torch.cat(yhat_list, dim=0)
    y_true_all = torch.cat(ytrue_list, dim=0)
    logvar_all = torch.cat(logvar_list, dim=0)

    y_hat_z_np = y_hat_all.numpy()
    y_true_z_np = y_true_all.numpy()
    logvar_z_np = logvar_all.numpy()
    return summarize_eval_predictions(
        split_artifacts=split_artifacts,
        y_hat_z_np=y_hat_z_np,
        y_true_z_np=y_true_z_np,
        logvar_z_np=logvar_z_np,
        save_samples=int(cfg_eval.save_samples),
        late_horizon_fraction=float(cfg_eval.long_horizon_fraction),
        trace_seconds=float(cfg_eval.trace_seconds),
        trace_dt_s=float(cfg_eval.trace_dt_s),
    )


def main() -> int:
    """纯数值评估入口，不直接承担绘图调度。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.eval.evaluate")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml (contains data/model/rollout config)")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.pth/.pt)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="which split to evaluate")

    ap.add_argument("--device", type=str, default=None, help="override device (cpu/cuda)")
    ap.add_argument("--batch_size", type=int, default=None, help="override batch size")
    ap.add_argument("--out_dir", type=str, default=None, help="override output directory")
    ap.add_argument("--save_samples", type=int, default=256, help="save first N samples to npz for viz")
    ap.add_argument("--trace_seconds", type=float, default=50.0, help="continuous prediction trace window length")
    ap.add_argument("--trace_dt", type=float, default=0.01, help="sample period for pred_trace time axis")

    # 仅保留解析以给出明确迁移提示；evaluate.py 不再承载绘图执行。
    ap.add_argument("--plots", action="store_true", help="deprecated: use cli/eval.py or cli/pipeline.py for plotting")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--n_plot_samples", type=int, default=8)

    args = ap.parse_args()

    if args.plots:
        print(
            "[EVAL] `--plots` 已弃用：`evaluate.py` 现在只负责数值评估与 artifact 落盘。"
            " 请改用 `python -m uwnav_dynamics.cli.eval ... --plots`"
            " 或 `python -m uwnav_dynamics.cli.pipeline ... --plots`。",
            file=sys.stderr,
        )
        return 2

    cfg_eval, cfg_model = build_eval_config(
        train_yaml=Path(args.yaml),
        ckpt=Path(args.ckpt),
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
        save_samples=int(args.save_samples),
        trace_seconds=float(args.trace_seconds),
        trace_dt_s=float(args.trace_dt),
    )

    _ensure_dir(cfg_eval.out_dir)
    res = evaluate_once(cfg_eval=cfg_eval, cfg_model=cfg_model)
    repo_root = infer_repo_root(Path(args.yaml))
    write_eval_outputs(
        out_dir=cfg_eval.out_dir,
        res=res,
        cfg_model=cfg_model,
        cfg_snapshot={
            "data_dir": cfg_eval.data_dir,
            "ckpt": cfg_eval.ckpt,
            "device": cfg_eval.device,
            "batch_size": cfg_eval.batch_size,
            "split_indices": cfg_eval.split_indices_path,
            "x_scaler": cfg_eval.x_scaler_path,
            "y_scaler": cfg_eval.y_scaler_path,
            "y0_source": cfg_eval.y0_source,
            "mode": cfg_eval.mode,
            "long_horizon_fraction": cfg_eval.long_horizon_fraction,
            "save_samples": cfg_eval.save_samples,
            "trace_seconds": cfg_eval.trace_seconds,
            "trace_dt_s": cfg_eval.trace_dt_s,
            "predictor": {
                "type": "neural",
                "kind": "s1_predictor",
                "ckpt": cfg_eval.ckpt,
            },
        },
        path_root=repo_root,
    )

    print(f"[EVAL] split={res['split']}  n_eval={res['n_eval']}")
    print(f"[EVAL] RMSE(global)={res['rmse_global']:.6f}  MAE(global)={res['mae_global']:.6f}")
    print(f"[EVAL] wrote: {cfg_eval.out_dir / 'metrics.yaml'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
