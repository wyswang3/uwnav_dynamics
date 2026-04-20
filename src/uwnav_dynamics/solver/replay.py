"""
模块名称：长序列 replay 验证

模块职责：
基于现有实验数据，把训练好的网络模型当作经验型状态求解器做
autoregressive 长序列 replay 验证，补上“固定窗口 rollout”之外的验证证据。

主要功能：
1. 从 `processed dataset + meta.yaml + base_csv` 恢复连续时间轴上的输入/目标序列。
2. 根据 split 对应的窗口起点 `idx0` 切出可重放的连续 segment。
3. 用求解器递推主状态、保留未来控制/上下文模板，执行长序列 autoregressive replay。
4. 支持“预测列 / 统一目标列 / 观测 mask”自定义评估口径，便于跨主线公平比较。
5. 汇总全局、segment 末步、误差增长、阈值生存、尾部误差与偏差指标，并落盘 artifact。

数据流：
data_dir/meta.yaml + base_csv + features.idx0 + split_indices
    ↓
replay segments on raw timeline
    ↓
transition solver autoregressive rollout
    ↓
    replay metrics / segment_metrics.csv / step_metrics.csv /
    representative pred_samples.npz / pred_context.npz / pred_sample_manifest.csv

依赖模块：
- numpy
- pandas
- yaml
- math
- uwnav_dynamics.dataset.split
- uwnav_dynamics.experiment.paths
- uwnav_dynamics.solver.transition_solver

备注：
- 当前 replay 验证默认把未来控制/上下文当作已知模板，
  只递推主状态槽位 `y_in_idx`。
- 若不同训练主线的代理状态定义不一致，可通过自定义评估口径，
  统一映射到 `base_csv` 中同一组观测列上再比较。
- 这一步用于验证“经验型状态求解器”的长序列可行性，
  不是闭环控制最终证明。
- 当前样例产物默认保存代表性 segment，而不是简单截取前 N 个片段。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.dataset.split import load_split_indices
from uwnav_dynamics.experiment.representative import RepresentativeRule, select_representative_rows
from uwnav_dynamics.experiment.paths import relative_path_str, to_snapshot_value
from uwnav_dynamics.solver.transition_solver import TrainedTransitionSolver


@dataclass(frozen=True)
class ReplayDataset:
    """replay 验证所需的连续基表与窗口索引。"""
    data_dir: Path
    base_csv: Path
    base_frame: pd.DataFrame
    hist_len: int
    pred_len: int
    input_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    inputs: np.ndarray
    targets: np.ndarray
    idx0: np.ndarray


@dataclass(frozen=True)
class ReplayEvalSpec:
    """
    replay 统一比较时的额外评估口径。

    设计目的：
    - 默认 replay 仍按各自数据集的 `target_cols` 评估；
    - 若需要跨不同训练主线做公平比较，可指定“从预测特征行里取哪些列，
      并与 base_csv 中哪组观测列对齐比较”。
    """

    name: str = "dataset_target"
    pred_source: str = "state"  # "state" | "feature_row"
    pred_cols: tuple[str, ...] = ()
    target_cols: tuple[str, ...] = ()
    mask_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayThresholdSpec:
    """
    长线拟合判据使用的默认阈值。

    备注：
    - 当前默认值优先服务“体速度对 DVL 观测”的 replay 口径；
    - 若未来改成其他状态量，建议在 launcher 中显式覆盖。
    """

    rmse_threshold: float = 0.05
    abs_error_threshold: float = 0.10


@dataclass(frozen=True)
class ReplaySegment:
    """单个可连续重放的时间段。"""
    segment_id: int
    start_idx0: int
    n_steps: int

    @property
    def stop_idx0_exclusive(self) -> int:
        return int(self.start_idx0 + self.n_steps)


@dataclass(frozen=True)
class ReplayResult:
    """长序列 replay 汇总结果。"""
    metrics: dict[str, Any]
    segment_rows: list[dict[str, Any]]
    sample_pred_y_hat: np.ndarray
    sample_pred_y_true: np.ndarray
    sample_pred_mask: np.ndarray
    sample_context: dict[str, np.ndarray]
    sample_manifest_rows: list[dict[str, Any]]


def replay_seconds_to_steps(seconds: float | int, *, dt_s: float = 0.01) -> int:
    """把 replay 时长秒数转换成向上取整的 step 数。"""
    seconds_f = float(seconds)
    dt_f = float(dt_s)
    if seconds_f <= 0.0:
        raise ValueError(f"replay seconds must be positive, got {seconds}")
    if dt_f <= 0.0:
        raise ValueError(f"replay dt_s must be positive, got {dt_s}")
    return max(1, int(math.ceil(seconds_f / dt_f)))


def resolve_replay_step_count(
    *,
    steps: int | None,
    seconds: float | int | None,
    dt_s: float,
    default_steps: int,
) -> int:
    """
    统一解析 replay 长度配置。

    若提供 `seconds`，优先按 `seconds / dt_s` 换算；
    否则使用显式 `steps`，再退回默认 step 数。
    """
    if seconds is not None:
        return replay_seconds_to_steps(seconds, dt_s=dt_s)
    if steps is None:
        return int(default_steps)
    steps_i = int(steps)
    if steps_i <= 0:
        raise ValueError(f"replay steps must be positive, got {steps}")
    return steps_i


def load_replay_dataset(data_dir: str | Path) -> ReplayDataset:
    """
    从 processed dataset 与 meta/base_csv 恢复 replay 所需连续数据。
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.yaml"
    feat_path = data_dir / "features.npz"
    label_path = data_dir / "labels.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing replay meta.yaml: {meta_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing replay features.npz: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing replay labels.npz: {label_path}")

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if not isinstance(meta, dict):
        raise TypeError(f"meta.yaml must be a dict: {meta_path}")

    base_csv = Path(str(meta.get("base_csv", ""))).expanduser().resolve()
    hist_len = int(meta.get("hist_len"))
    pred_len = int(meta.get("pred_len"))
    input_cols = tuple(str(v) for v in meta.get("input_cols", []))
    target_cols = tuple(str(v) for v in meta.get("target_cols", []))
    if not base_csv.exists():
        raise FileNotFoundError(f"Replay base_csv not found: {base_csv}")
    if not input_cols or not target_cols:
        raise ValueError(f"meta.yaml missing input_cols/target_cols: {meta_path}")

    base_df = pd.read_csv(base_csv)
    missing_inputs = [c for c in input_cols if c not in base_df.columns]
    missing_targets = [c for c in target_cols if c not in base_df.columns]
    if missing_inputs:
        raise KeyError(f"base_csv missing replay input columns: {missing_inputs}")
    if missing_targets:
        raise KeyError(f"base_csv missing replay target columns: {missing_targets}")

    with np.load(feat_path, allow_pickle=True) as feat_npz:
        if "idx0" not in feat_npz:
            raise KeyError(f"features.npz missing idx0: {feat_path}")
        idx0 = np.asarray(feat_npz["idx0"], dtype=np.int64)

    inputs = base_df.loc[:, list(input_cols)].to_numpy(dtype=np.float32, copy=True)
    targets = base_df.loc[:, list(target_cols)].to_numpy(dtype=np.float32, copy=True)
    return ReplayDataset(
        data_dir=data_dir,
        base_csv=base_csv,
        base_frame=base_df,
        hist_len=hist_len,
        pred_len=pred_len,
        input_cols=input_cols,
        target_cols=target_cols,
        inputs=inputs,
        targets=targets,
        idx0=idx0,
    )


def build_replay_segments(
    *,
    idx0: np.ndarray,
    split_window_indices: np.ndarray,
    min_steps: int,
    max_segments: int | None = None,
    max_steps_per_segment: int | None = None,
) -> list[ReplaySegment]:
    """
    把 split 中的窗口起点切成连续可重放的 segment。
    """
    split_idx = np.asarray(split_window_indices, dtype=np.int64)
    if split_idx.size == 0:
        return []
    start_rows = np.sort(np.asarray(idx0[split_idx], dtype=np.int64))
    segments: list[ReplaySegment] = []

    seg_start = int(start_rows[0])
    prev = int(start_rows[0])
    segment_id = 0

    def _flush(start: int, stop_inclusive: int) -> None:
        nonlocal segment_id, segments
        total_steps = int(stop_inclusive - start + 1)
        if total_steps < int(min_steps):
            return
        if max_steps_per_segment is None or total_steps <= int(max_steps_per_segment):
            segments.append(ReplaySegment(segment_id=segment_id, start_idx0=int(start), n_steps=int(total_steps)))
            segment_id += 1
            return

        offset = 0
        chunk = int(max_steps_per_segment)
        while offset < total_steps:
            remaining = total_steps - offset
            take = min(chunk, remaining)
            if take < int(min_steps):
                break
            segments.append(ReplaySegment(segment_id=segment_id, start_idx0=int(start + offset), n_steps=int(take)))
            segment_id += 1
            offset += take

    for cur in start_rows[1:]:
        cur_i = int(cur)
        if cur_i == prev + 1:
            prev = cur_i
            continue
        _flush(seg_start, prev)
        seg_start = cur_i
        prev = cur_i
    _flush(seg_start, prev)

    if max_segments is not None:
        return segments[: int(max_segments)]
    return segments


def _expand_eval_mask(mask: np.ndarray | None, *, shape: tuple[int, int]) -> np.ndarray:
    """把 1D/2D mask 统一广播成 `(T,D)` 布尔矩阵。"""
    t_steps, n_dim = int(shape[0]), int(shape[1])
    if mask is None:
        return np.ones((t_steps, n_dim), dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 1:
        if arr.shape[0] != t_steps:
            raise ValueError(f"mask length mismatch: {arr.shape[0]} vs {t_steps}")
        return np.repeat(arr.reshape(t_steps, 1).astype(bool), n_dim, axis=1)
    if arr.ndim == 2 and tuple(arr.shape) == (t_steps, 1):
        return np.repeat(arr.astype(bool), n_dim, axis=1)
    if arr.ndim == 2 and tuple(arr.shape) == (t_steps, n_dim):
        return arr.astype(bool, copy=False)
    raise ValueError(f"Unsupported mask shape {tuple(arr.shape)} for target shape {shape}")


def _valid_eval_mask(y_hat: np.ndarray, y_true: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    """构造“用户掩码 + 有限值”联合有效掩码。"""
    y_hat_arr = np.asarray(y_hat, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    if y_hat_arr.shape != y_true_arr.shape:
        raise ValueError(f"Prediction/target shape mismatch: {y_hat_arr.shape} vs {y_true_arr.shape}")
    eval_mask = _expand_eval_mask(mask, shape=y_hat_arr.shape)
    return eval_mask & np.isfinite(y_hat_arr) & np.isfinite(y_true_arr)


def _masked_values(values: np.ndarray, *, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(f"Value/mask shape mismatch: {arr.shape} vs {mask.shape}")
    return arr[mask]


def _global_rmse_mae(y_hat: np.ndarray, y_true: np.ndarray, *, mask: np.ndarray | None = None) -> tuple[float, float]:
    valid_mask = _valid_eval_mask(y_hat, y_true, mask=mask)
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    vals = _masked_values(err, valid_mask=valid_mask)
    if vals.size <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(vals * vals))), float(np.mean(np.abs(vals)))


def _edge_ratio(first: float, last: float) -> float:
    if not np.isfinite(first) or not np.isfinite(last):
        return float("nan")
    eps = 1e-12
    if abs(float(first)) <= eps:
        return 1.0 if abs(float(last)) <= eps else float("inf")
    return float(last / first)


def _nanmean_or_nan(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(np.mean(finite))


def _nanpercentile_or_nan(values: np.ndarray | Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _linear_slope(values: np.ndarray | Sequence[float]) -> float:
    """对 step-wise 曲线做最小二乘线性斜率估计。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(arr)
    if int(finite_mask.sum()) < 2:
        return float("nan")
    x = np.arange(1, arr.size + 1, dtype=np.float64)[finite_mask]
    y = arr[finite_mask]
    x_center = x - np.mean(x)
    denom = float(np.sum(x_center * x_center))
    if denom <= 0.0:
        return float("nan")
    y_center = y - np.mean(y)
    return float(np.sum(x_center * y_center) / denom)


def _first_threshold_breach_step(values: np.ndarray | Sequence[float], *, threshold: float) -> int | None:
    """返回首次超过阈值的 1-based step；未超过则返回 None。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    hit = np.where(np.isfinite(arr) & (arr > float(threshold)))[0]
    if hit.size <= 0:
        return None
    return int(hit[0] + 1)


def _tail_percentiles(values: np.ndarray, *, percentiles: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for q in percentiles:
        out[f"p{int(q)}"] = float(np.percentile(arr, q)) if arr.size > 0 else float("nan")
    return out


def _component_metric_rows(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    component_labels: Sequence[str],
    mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    valid_mask = _valid_eval_mask(y_hat, y_true, mask=mask)
    rows: list[dict[str, Any]] = []
    for dim, label in enumerate(component_labels):
        vals = err[:, dim][valid_mask[:, dim]]
        rows.append(
            {
                "component": str(label),
                "count": int(vals.size),
                "rmse": float(np.sqrt(np.mean(vals * vals))) if vals.size > 0 else float("nan"),
                "mae": float(np.mean(np.abs(vals))) if vals.size > 0 else float("nan"),
                "bias": float(np.mean(vals)) if vals.size > 0 else float("nan"),
            }
        )
    return rows


def _pad_sample_segments(segments: list[dict[str, Any]], *, dout: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not segments:
        return (
            np.zeros((0, 0, dout), dtype=np.float32),
            np.zeros((0, 0, dout), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
        )
    max_len = max(int(seg["n_steps"]) for seg in segments)
    n_seg = len(segments)
    y_hat = np.full((n_seg, max_len, dout), np.nan, dtype=np.float32)
    y_true = np.full((n_seg, max_len, dout), np.nan, dtype=np.float32)
    mask = np.zeros((n_seg, max_len), dtype=bool)
    for i, seg in enumerate(segments):
        seg_len = int(seg["n_steps"])
        y_hat[i, :seg_len, :] = np.asarray(seg["y_hat"], dtype=np.float32)
        y_true[i, :seg_len, :] = np.asarray(seg["y_true"], dtype=np.float32)
        mask[i, :seg_len] = True
    return y_hat, y_true, mask


def _resolve_eval_mask_rows(base_rows: pd.DataFrame, *, mask_cols: Sequence[str]) -> np.ndarray:
    """
    从 base_csv 行片段中恢复评估掩码。

    约定：
    - `mask_cols` 作为候选列列表，取第一列存在的列；
    - 若未提供 `mask_cols`，则视为全有效；
    - 若提供了候选列但都不存在，直接 fail-fast。
    """
    if len(mask_cols) <= 0:
        return np.ones(len(base_rows), dtype=bool)
    for col in mask_cols:
        if col in base_rows.columns:
            vals = pd.to_numeric(base_rows[col], errors="coerce").to_numpy(dtype=float)
            return np.isfinite(vals) & (vals > 0.5)
    raise KeyError(f"None of eval.mask_cols exist in base_csv: {list(mask_cols)}")


def _build_step_metric_rows(
    segment_curves: Sequence[dict[str, Any]],
    *,
    thresholds: ReplayThresholdSpec,
) -> list[dict[str, Any]]:
    """把各段逐步误差曲线汇总成 step-wise artifact。"""
    if len(segment_curves) <= 0:
        return []
    max_len = max(int(curve["n_steps"]) for curve in segment_curves)
    rows: list[dict[str, Any]] = []
    for step_idx in range(max_len):
        abs_chunks: list[np.ndarray] = []
        active_segments = 0
        rmse_survivors = 0
        abs_survivors = 0
        for curve in segment_curves:
            if step_idx >= int(curve["n_steps"]):
                continue
            step_abs = np.asarray(curve["abs_values"][step_idx], dtype=np.float64).reshape(-1)
            if step_abs.size <= 0:
                continue
            active_segments += 1
            abs_chunks.append(step_abs)
            rmse_breach_step = curve["rmse_breach_step"]
            abs_breach_step = curve["abs_breach_step"]
            if rmse_breach_step is None or int(rmse_breach_step) > (step_idx + 1):
                rmse_survivors += 1
            if abs_breach_step is None or int(abs_breach_step) > (step_idx + 1):
                abs_survivors += 1

        if abs_chunks:
            abs_vals = np.concatenate(abs_chunks, axis=0)
            rmse_global = float(np.sqrt(np.mean(abs_vals * abs_vals)))
            mae_global = float(np.mean(abs_vals))
            abs_p50 = float(np.percentile(abs_vals, 50.0))
            abs_p95 = float(np.percentile(abs_vals, 95.0))
        else:
            abs_vals = np.zeros((0,), dtype=np.float64)
            rmse_global = float("nan")
            mae_global = float("nan")
            abs_p50 = float("nan")
            abs_p95 = float("nan")

        rows.append(
            {
                "step": int(step_idx + 1),
                "active_segments": int(active_segments),
                "value_count": int(abs_vals.size),
                "rmse_global": rmse_global,
                "mae_global": mae_global,
                "abs_p50_global": abs_p50,
                "abs_p95_global": abs_p95,
                "rmse_survival_rate": (
                    float(rmse_survivors / active_segments) if active_segments > 0 else float("nan")
                ),
                "abs_survival_rate": (
                    float(abs_survivors / active_segments) if active_segments > 0 else float("nan")
                ),
                "rmse_threshold": float(thresholds.rmse_threshold),
                "abs_error_threshold": float(thresholds.abs_error_threshold),
            }
        )
    return rows


def _resolve_eval_arrays(
    *,
    replay_dataset: ReplayDataset,
    rollout_rows: np.ndarray,
    predicted_states: np.ndarray,
    hist_stop: int,
    target_stop: int,
    eval_spec: ReplayEvalSpec | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...], dict[str, Any]]:
    """
    根据评估口径恢复本段 `y_hat / y_true / mask`。
    """
    if eval_spec is None:
        return (
            np.asarray(predicted_states, dtype=np.float32),
            np.asarray(replay_dataset.targets[hist_stop:target_stop, :], dtype=np.float32),
            None,
            replay_dataset.target_cols,
            {
                "name": "dataset_target",
                "pred_source": "state",
                "pred_cols": list(replay_dataset.target_cols),
                "target_cols": list(replay_dataset.target_cols),
                "mask_cols": [],
                "last_step_policy": "segment_last_step",
            },
        )

    spec = ReplayEvalSpec(
        name=str(eval_spec.name or "custom_eval"),
        pred_source=str(eval_spec.pred_source or "state"),
        pred_cols=tuple(str(v) for v in eval_spec.pred_cols),
        target_cols=tuple(str(v) for v in eval_spec.target_cols),
        mask_cols=tuple(str(v) for v in eval_spec.mask_cols),
    )
    if spec.pred_source not in {"state", "feature_row"}:
        raise ValueError(f"Unsupported eval.pred_source={spec.pred_source!r}; expect 'state' or 'feature_row'")
    if len(spec.target_cols) <= 0:
        raise ValueError("eval.target_cols must be non-empty when eval spec is provided")

    base_rows = replay_dataset.base_frame.iloc[hist_stop:target_stop, :]
    missing_targets = [c for c in spec.target_cols if c not in base_rows.columns]
    if missing_targets:
        raise KeyError(f"base_csv missing eval.target_cols: {missing_targets}")
    y_true = base_rows.loc[:, list(spec.target_cols)].to_numpy(dtype=np.float32, copy=True)
    mask = _resolve_eval_mask_rows(base_rows, mask_cols=spec.mask_cols)

    if spec.pred_source == "state":
        y_hat = np.asarray(predicted_states, dtype=np.float32)
        if len(spec.pred_cols) > 0 and tuple(spec.pred_cols) != tuple(replay_dataset.target_cols):
            raise ValueError("eval.pred_cols is only supported with pred_source='feature_row'")
        if y_hat.shape[1] != len(spec.target_cols):
            raise ValueError(
                "state prediction dim does not match eval.target_cols: "
                f"{y_hat.shape[1]} vs {len(spec.target_cols)}"
            )
        pred_cols = tuple(replay_dataset.target_cols)
    else:
        if len(spec.pred_cols) <= 0:
            raise ValueError("eval.pred_cols must be non-empty when pred_source='feature_row'")
        pred_idx: list[int] = []
        for col in spec.pred_cols:
            if col not in replay_dataset.input_cols:
                raise KeyError(f"eval.pred_cols column not found in input_cols: {col}")
            pred_idx.append(int(replay_dataset.input_cols.index(col)))
        y_hat = np.asarray(rollout_rows[:, pred_idx], dtype=np.float32)
        pred_cols = tuple(spec.pred_cols)

    if y_hat.shape != y_true.shape:
        raise ValueError(f"eval prediction/target shape mismatch: {y_hat.shape} vs {y_true.shape}")
    return (
        y_hat,
        y_true,
        mask,
        tuple(str(v) for v in spec.target_cols),
        {
            "name": spec.name,
            "pred_source": spec.pred_source,
            "pred_cols": list(pred_cols),
            "target_cols": list(spec.target_cols),
            "mask_cols": list(spec.mask_cols),
            "last_step_policy": "last_valid_observation" if len(spec.mask_cols) > 0 else "segment_last_step",
        },
    )


def _select_representative_segment_rows(
    *,
    rows: Sequence[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    return select_representative_rows(
        rows,
        id_key="segment_id",
        max_items=max_items,
        rules=(
            RepresentativeRule(tag="best_rmse", metric="rmse_global", mode="min"),
            RepresentativeRule(tag="median_rmse", metric="rmse_global", mode="median"),
            RepresentativeRule(tag="worst_rmse", metric="rmse_global", mode="max"),
            RepresentativeRule(tag="worst_final_step", metric="final_step_rmse", mode="max"),
            RepresentativeRule(tag="worst_growth", metric="rmse_last_over_first", mode="max"),
        ),
    )


def run_transition_replay(
    *,
    replay_dataset: ReplayDataset,
    split_indices_path: str | Path,
    split_name: str,
    solver: TrainedTransitionSolver,
    min_steps: int,
    max_segments: int | None = None,
    max_steps_per_segment: int | None = None,
    save_samples: int = 8,
    eval_spec: ReplayEvalSpec | None = None,
    thresholds: ReplayThresholdSpec | None = None,
) -> ReplayResult:
    """
    执行长序列 autoregressive replay，并汇总数值结果。
    """
    split_indices = load_split_indices(split_indices_path)
    if split_name not in split_indices:
        raise KeyError(f"split '{split_name}' not found in split artifact")

    segments = build_replay_segments(
        idx0=replay_dataset.idx0,
        split_window_indices=np.asarray(split_indices[split_name], dtype=np.int64),
        min_steps=int(min_steps),
        max_segments=max_segments,
        max_steps_per_segment=max_steps_per_segment,
    )
    if not segments:
        raise RuntimeError(
            f"No replay segments built for split={split_name!r}. "
            f"Try lowering min_steps (current={min_steps})."
        )

    resolved_thresholds = thresholds or ReplayThresholdSpec()
    all_hat: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    segment_rows: list[dict[str, Any]] = []
    segment_curves: list[dict[str, Any]] = []
    segment_payloads: list[dict[str, Any]] = []
    nonfinite_trigger_count = 0
    eval_meta: dict[str, Any] | None = None
    component_labels: tuple[str, ...] = replay_dataset.target_cols
    final_step_abs_segments: list[np.ndarray] = []

    for segment in segments:
        start = int(segment.start_idx0)
        hist_stop = start + int(replay_dataset.hist_len)
        target_stop = hist_stop + int(segment.n_steps)

        history = replay_dataset.inputs[start:hist_stop, :]
        future_templates = replay_dataset.inputs[hist_stop:target_stop, :]
        y_true = replay_dataset.targets[hist_stop:target_stop, :]
        if history.shape[0] != int(replay_dataset.hist_len):
            raise ValueError(
                f"segment {segment.segment_id} initial history length mismatch: "
                f"{history.shape[0]} vs {replay_dataset.hist_len}"
            )
        if future_templates.shape[0] != int(segment.n_steps) or y_true.shape[0] != int(segment.n_steps):
            raise ValueError(
                f"segment {segment.segment_id} step length mismatch: "
                f"template={future_templates.shape[0]}, target={y_true.shape[0]}, expected={segment.n_steps}"
            )

        rollout = solver.rollout_with_feature_templates(
            initial_history_physical=history,
            future_feature_templates_physical=future_templates,
        )
        y_hat, y_true_eval, eval_mask, eval_labels, segment_eval_meta = _resolve_eval_arrays(
            replay_dataset=replay_dataset,
            rollout_rows=rollout.replay_rows,
            predicted_states=rollout.predicted_states,
            hist_stop=hist_stop,
            target_stop=target_stop,
            eval_spec=eval_spec,
        )
        if eval_meta is None:
            eval_meta = dict(segment_eval_meta)
            component_labels = tuple(eval_labels)
        if y_hat.shape != y_true_eval.shape:
            raise ValueError(f"segment {segment.segment_id} prediction shape mismatch: {y_hat.shape} vs {y_true_eval.shape}")

        nonfinite_trigger_count += int(rollout.nonfinite_trigger_count)
        all_hat.append(y_hat)
        all_true.append(y_true_eval)
        all_masks.append(_expand_eval_mask(eval_mask, shape=y_hat.shape))

        seg_valid_mask = _valid_eval_mask(y_hat, y_true_eval, mask=eval_mask)
        seg_rmse, seg_mae = _global_rmse_mae(y_hat, y_true_eval, mask=eval_mask)
        seg_abs = np.abs(np.asarray(y_hat - y_true_eval, dtype=np.float64))
        step_rmse_curve = np.full(int(segment.n_steps), np.nan, dtype=np.float64)
        step_mae_curve = np.full(int(segment.n_steps), np.nan, dtype=np.float64)
        step_abs_max_curve = np.full(int(segment.n_steps), np.nan, dtype=np.float64)
        step_abs_values: list[np.ndarray] = []
        for step_idx in range(int(segment.n_steps)):
            row_mask = seg_valid_mask[step_idx:step_idx + 1, :]
            row_abs = _masked_values(seg_abs[step_idx:step_idx + 1, :], valid_mask=row_mask)
            step_abs_values.append(np.asarray(row_abs, dtype=np.float64))
            if row_abs.size <= 0:
                continue
            step_rmse_curve[step_idx] = float(np.sqrt(np.mean(row_abs * row_abs)))
            step_mae_curve[step_idx] = float(np.mean(row_abs))
            step_abs_max_curve[step_idx] = float(np.max(row_abs))
        valid_rows = np.where(seg_valid_mask.any(axis=1))[0]
        first_idx = int(valid_rows[0]) if valid_rows.size > 0 else None
        last_idx = int(valid_rows[-1]) if valid_rows.size > 0 else None
        first_rmse, first_mae = (
            _global_rmse_mae(y_hat[first_idx:first_idx + 1, :], y_true_eval[first_idx:first_idx + 1, :], mask=eval_mask[first_idx:first_idx + 1] if eval_mask is not None else None)
            if first_idx is not None
            else (float("nan"), float("nan"))
        )
        final_rmse, final_mae = (
            _global_rmse_mae(y_hat[last_idx:last_idx + 1, :], y_true_eval[last_idx:last_idx + 1, :], mask=eval_mask[last_idx:last_idx + 1] if eval_mask is not None else None)
            if last_idx is not None
            else (float("nan"), float("nan"))
        )
        if last_idx is not None:
            final_step_abs_segments.append(
                _masked_values(
                    np.abs(np.asarray(y_hat[last_idx:last_idx + 1, :] - y_true_eval[last_idx:last_idx + 1, :], dtype=np.float64)),
                    valid_mask=_valid_eval_mask(
                        y_hat[last_idx:last_idx + 1, :],
                        y_true_eval[last_idx:last_idx + 1, :],
                        mask=eval_mask[last_idx:last_idx + 1] if eval_mask is not None else None,
                    ),
                )
            )
        rmse_breach_step = _first_threshold_breach_step(
            step_rmse_curve,
            threshold=float(resolved_thresholds.rmse_threshold),
        )
        abs_breach_step = _first_threshold_breach_step(
            step_abs_max_curve,
            threshold=float(resolved_thresholds.abs_error_threshold),
        )
        segment_rows.append(
            {
                "segment_id": int(segment.segment_id),
                "start_idx0": int(segment.start_idx0),
                "n_steps": int(segment.n_steps),
                "eval_steps": int(seg_valid_mask.any(axis=1).sum()),
                "rmse_global": seg_rmse,
                "mae_global": seg_mae,
                "final_step_rmse": final_rmse,
                "final_step_mae": final_mae,
                "rmse_last_over_first": _edge_ratio(first_rmse, final_rmse),
                "mae_last_over_first": _edge_ratio(first_mae, final_mae),
                "rmse_threshold_breach_step": float(rmse_breach_step) if rmse_breach_step is not None else float("nan"),
                "abs_error_threshold_breach_step": float(abs_breach_step) if abs_breach_step is not None else float("nan"),
            }
        )
        segment_curves.append(
            {
                "segment_id": int(segment.segment_id),
                "n_steps": int(segment.n_steps),
                "abs_values": step_abs_values,
                "rmse_curve": step_rmse_curve,
                "mae_curve": step_mae_curve,
                "abs_max_curve": step_abs_max_curve,
                "rmse_breach_step": rmse_breach_step,
                "abs_breach_step": abs_breach_step,
            }
        )
        segment_payloads.append(
            {
                "segment_id": int(segment.segment_id),
                "start_idx0": int(segment.start_idx0),
                "n_steps": int(segment.n_steps),
                "y_hat": y_hat,
                "y_true": y_true_eval,
                "valid_mask": _expand_eval_mask(eval_mask, shape=y_hat.shape),
            }
        )

    y_hat_all = np.concatenate(all_hat, axis=0)
    y_true_all = np.concatenate(all_true, axis=0)
    eval_mask_all = np.concatenate(all_masks, axis=0) if all_masks else None
    global_rmse, global_mae = _global_rmse_mae(y_hat_all, y_true_all, mask=eval_mask_all)
    final_step_vals = (
        np.concatenate([arr for arr in final_step_abs_segments if np.asarray(arr).size > 0], axis=0)
        if len(final_step_abs_segments) > 0
        else np.zeros((0,), dtype=np.float64)
    )

    valid_mask_all = _valid_eval_mask(y_hat_all, y_true_all, mask=eval_mask_all)
    all_abs_err = _masked_values(np.abs(np.asarray(y_hat_all - y_true_all, dtype=np.float64)), valid_mask=valid_mask_all)
    component_rows = _component_metric_rows(
        y_hat_all,
        y_true_all,
        component_labels=component_labels,
        mask=eval_mask_all,
    )
    step_metric_rows = _build_step_metric_rows(segment_curves, thresholds=resolved_thresholds)
    step_rmse_curve = np.asarray([float(row["rmse_global"]) for row in step_metric_rows], dtype=np.float64)
    step_rmse_curve_clipped = np.clip(step_rmse_curve, 1e-12, None)
    worst_component = None
    worst_abs_bias = float("nan")
    for row in component_rows:
        bias = float(row["bias"])
        if not np.isfinite(bias):
            continue
        abs_bias = abs(bias)
        if worst_component is None or abs_bias > worst_abs_bias:
            worst_component = str(row["component"])
            worst_abs_bias = float(abs_bias)

    growth_rmse = np.asarray([float(row["rmse_last_over_first"]) for row in segment_rows], dtype=np.float64)
    growth_mae = np.asarray([float(row["mae_last_over_first"]) for row in segment_rows], dtype=np.float64)
    rmse_breach_steps = np.asarray(
        [float(curve["rmse_breach_step"]) if curve["rmse_breach_step"] is not None else float("nan") for curve in segment_curves],
        dtype=np.float64,
    )
    abs_breach_steps = np.asarray(
        [float(curve["abs_breach_step"]) if curve["abs_breach_step"] is not None else float("nan") for curve in segment_curves],
        dtype=np.float64,
    )
    representative_segment_rows = _select_representative_segment_rows(
        rows=segment_rows,
        max_items=int(min(save_samples, len(segment_rows))),
    )
    payload_by_segment_id = {int(item["segment_id"]): item for item in segment_payloads}
    representative_payloads = [
        payload_by_segment_id[int(row["segment_id"])]
        for row in representative_segment_rows
        if int(row["segment_id"]) in payload_by_segment_id
    ]
    sample_y_hat, sample_y_true, sample_mask = _pad_sample_segments(
        representative_payloads,
        dout=int(y_hat_all.shape[1]),
    )
    sample_context = {
        "segment_id": np.asarray([int(row["segment_id"]) for row in representative_segment_rows], dtype=np.int64),
        "start_idx0": np.asarray([int(row["start_idx0"]) for row in representative_segment_rows], dtype=np.int64),
        "n_steps": np.asarray([int(row["n_steps"]) for row in representative_segment_rows], dtype=np.int64),
        "sample_tag": np.asarray([str(row["representative_tag"]) for row in representative_segment_rows], dtype=str),
        "component_labels": np.asarray(component_labels, dtype=str),
    }
    segment_lengths = np.asarray([int(row["n_steps"]) for row in segment_rows], dtype=np.float64)
    eval_steps = int(valid_mask_all.any(axis=1).sum())
    eval_values = int(valid_mask_all.sum())
    segment_success_count = int(len(segment_rows))

    metrics = {
        "schema_version": "transition_replay_v1",
        "intended_use": "autoregressive_replay_for_transition_solver",
        "closed_loop_proof": False,
        "split": str(split_name),
        "solver_step_semantics": str(solver.step_semantics),
        "model_pred_len": int(solver.cfg_model.pred_len),
        "eval": eval_meta or {
            "name": "dataset_target",
            "pred_source": "state",
            "pred_cols": list(replay_dataset.target_cols),
            "target_cols": list(replay_dataset.target_cols),
            "mask_cols": [],
            "last_step_policy": "segment_last_step",
        },
        "segment_count": int(len(segment_rows)),
        "total_steps": int(y_hat_all.shape[0]),
        "eval_steps": int(eval_steps),
        "eval_values": int(eval_values),
        "nonfinite_trigger_count": int(nonfinite_trigger_count),
        "rmse_global": global_rmse,
        "mae_global": global_mae,
        "final_step": {
            "rmse_global_mean": float(np.mean([float(row["final_step_rmse"]) for row in segment_rows])),
            "mae_global_mean": float(np.mean([float(row["final_step_mae"]) for row in segment_rows])),
            "abs_p95_global": _tail_percentiles(final_step_vals, percentiles=(95.0,))["p95"],
        },
        "rollout_growth": {
            "rmse_last_over_first_mean": float(np.nanmean(growth_rmse)),
            "rmse_last_over_first_p95": float(np.nanpercentile(growth_rmse, 95.0)),
            "mae_last_over_first_mean": float(np.nanmean(growth_mae)),
            "mae_last_over_first_p95": float(np.nanpercentile(growth_mae, 95.0)),
        },
        "tail_error": {
            "abs_p95_global": _tail_percentiles(all_abs_err, percentiles=(95.0,))["p95"],
            "abs_p99_global": _tail_percentiles(all_abs_err, percentiles=(99.0,))["p99"],
            "final_step_abs_p95_global": _tail_percentiles(final_step_vals, percentiles=(95.0,))["p95"],
        },
        "long_horizon": {
            "schema_version": "transition_replay_long_horizon_v1",
            "rmse_slope": _linear_slope(step_rmse_curve),
            "log_rmse_slope": _linear_slope(np.log(step_rmse_curve_clipped)),
            "thresholds": {
                "rmse": float(resolved_thresholds.rmse_threshold),
                "abs_error": float(resolved_thresholds.abs_error_threshold),
            },
            "time_to_threshold": {
                "rmse": {
                    "failure_rate": float(np.isfinite(rmse_breach_steps).sum() / max(len(segment_curves), 1)),
                    "survival_rate": float(1.0 - (np.isfinite(rmse_breach_steps).sum() / max(len(segment_curves), 1))),
                    "breach_step_mean": _nanmean_or_nan(rmse_breach_steps),
                    "breach_step_p95": _nanpercentile_or_nan(rmse_breach_steps, 95.0),
                },
                "abs_error": {
                    "failure_rate": float(np.isfinite(abs_breach_steps).sum() / max(len(segment_curves), 1)),
                    "survival_rate": float(1.0 - (np.isfinite(abs_breach_steps).sum() / max(len(segment_curves), 1))),
                    "breach_step_mean": _nanmean_or_nan(abs_breach_steps),
                    "breach_step_p95": _nanpercentile_or_nan(abs_breach_steps, 95.0),
                },
            },
        },
        "bias": {
            "worst_component": worst_component,
            "worst_abs_bias": worst_abs_bias,
        },
        "robustness": {
            "finite_pass_rate": float(segment_success_count / max(int(len(segment_rows)), 1)),
            "segment_success_count": int(segment_success_count),
            "segment_failure_count": int(len(segment_rows) - segment_success_count),
        },
        "segment_stats": {
            "n_steps_mean": float(np.mean(segment_lengths)),
            "n_steps_p95": float(np.percentile(segment_lengths, 95.0)),
        },
        "component_metrics": component_rows,
        "step_metrics": step_metric_rows,
    }
    return ReplayResult(
        metrics=metrics,
        segment_rows=segment_rows,
        sample_pred_y_hat=sample_y_hat,
        sample_pred_y_true=sample_y_true,
        sample_pred_mask=sample_mask,
        sample_context=sample_context,
        sample_manifest_rows=representative_segment_rows,
    )


def _write_segment_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = (
        "segment_id,start_idx0,n_steps,rmse_global,mae_global,"
        "final_step_rmse,final_step_mae,rmse_last_over_first,mae_last_over_first,"
        "rmse_threshold_breach_step,abs_error_threshold_breach_step"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["segment_id"])),
                    str(int(row["start_idx0"])),
                    str(int(row["n_steps"])),
                    f"{float(row['rmse_global']):.8f}",
                    f"{float(row['mae_global']):.8f}",
                    f"{float(row['final_step_rmse']):.8f}",
                    f"{float(row['final_step_mae']):.8f}",
                    f"{float(row['rmse_last_over_first']):.8f}",
                    f"{float(row['mae_last_over_first']):.8f}",
                    f"{float(row['rmse_threshold_breach_step']):.8f}",
                    f"{float(row['abs_error_threshold_breach_step']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_component_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = "component,count,rmse,mae,bias"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["component"]),
                    str(int(row["count"])),
                    f"{float(row['rmse']):.8f}",
                    f"{float(row['mae']):.8f}",
                    f"{float(row['bias']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_step_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = (
        "step,active_segments,value_count,rmse_global,mae_global,abs_p50_global,abs_p95_global,"
        "rmse_survival_rate,abs_survival_rate,rmse_threshold,abs_error_threshold"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["step"])),
                    str(int(row["active_segments"])),
                    str(int(row["value_count"])),
                    f"{float(row['rmse_global']):.8f}",
                    f"{float(row['mae_global']):.8f}",
                    f"{float(row['abs_p50_global']):.8f}",
                    f"{float(row['abs_p95_global']):.8f}",
                    f"{float(row['rmse_survival_rate']):.8f}",
                    f"{float(row['abs_survival_rate']):.8f}",
                    f"{float(row['rmse_threshold']):.8f}",
                    f"{float(row['abs_error_threshold']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sample_manifest_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = (
        "segment_id,start_idx0,n_steps,representative_tag,representative_metric,representative_mode,"
        "representative_value,eval_steps,rmse_global,mae_global,final_step_rmse,final_step_mae,rmse_last_over_first"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row["segment_id"])),
                    str(int(row["start_idx0"])),
                    str(int(row["n_steps"])),
                    str(row["representative_tag"]),
                    str(row["representative_metric"]),
                    str(row["representative_mode"]),
                    f"{float(row['representative_value']):.8f}",
                    str(int(row["eval_steps"])),
                    f"{float(row['rmse_global']):.8f}",
                    f"{float(row['mae_global']):.8f}",
                    f"{float(row['final_step_rmse']):.8f}",
                    f"{float(row['final_step_mae']):.8f}",
                    f"{float(row['rmse_last_over_first']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_replay_outputs(
    *,
    out_dir: str | Path,
    replay_result: ReplayResult,
    cfg_snapshot: dict[str, Any],
    path_root: str | Path,
) -> None:
    """
    把 replay 验证结果落盘到统一 artifact 目录。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_payload = dict(replay_result.metrics)
    metrics_payload["artifacts"] = {
        "segment_metrics_csv": "segment_metrics.csv",
        "component_metrics_csv": "component_metrics.csv",
        "step_metrics_csv": "step_metrics.csv",
        "pred_samples": "pred_samples.npz",
        "pred_context": "pred_context.npz",
        "sample_manifest_csv": "pred_sample_manifest.csv",
        "resolved_replay_yaml": "resolved_replay.yaml",
    }
    metrics_payload["cfg"] = to_snapshot_value(cfg_snapshot, base_dir=path_root)
    with open(out_path / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_payload, f, sort_keys=False, allow_unicode=True)

    resolved = {
        "schema_version": "transition_replay_resolved_v1",
        "out_dir": relative_path_str(out_path, base_dir=path_root),
        "cfg": to_snapshot_value(cfg_snapshot, base_dir=path_root),
    }
    with open(out_path / "resolved_replay.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)

    _write_segment_metric_csv(out_path / "segment_metrics.csv", replay_result.segment_rows)
    _write_component_metric_csv(out_path / "component_metrics.csv", replay_result.metrics["component_metrics"])
    _write_step_metric_csv(out_path / "step_metrics.csv", replay_result.metrics["step_metrics"])
    np.savez_compressed(
        out_path / "pred_samples.npz",
        y_hat=replay_result.sample_pred_y_hat,
        y_true=replay_result.sample_pred_y_true,
        valid_mask=replay_result.sample_pred_mask,
    )
    np.savez_compressed(
        out_path / "pred_context.npz",
        segment_id=replay_result.sample_context["segment_id"],
        start_idx0=replay_result.sample_context["start_idx0"],
        n_steps=replay_result.sample_context["n_steps"],
        sample_tag=replay_result.sample_context["sample_tag"],
        component_labels=replay_result.sample_context["component_labels"],
    )
    _write_sample_manifest_csv(out_path / "pred_sample_manifest.csv", replay_result.sample_manifest_rows)
