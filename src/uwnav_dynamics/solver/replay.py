"""
模块名称：长序列 replay 验证

模块职责：
基于现有实验数据，把训练好的网络模型当作经验型状态求解器做
autoregressive 长序列 replay 验证，补上“固定窗口 rollout”之外的验证证据。

主要功能：
1. 从 `processed dataset + meta.yaml + base_csv` 恢复连续时间轴上的输入/目标序列。
2. 根据 split 对应的窗口起点 `idx0` 切出可重放的连续 segment。
3. 用求解器递推主状态、保留未来控制/上下文模板，执行长序列 autoregressive replay。
4. 汇总全局、segment 末步、误差增长、尾部误差与偏差指标，并落盘 artifact。

数据流：
data_dir/meta.yaml + base_csv + features.idx0 + split_indices
    ↓
replay segments on raw timeline
    ↓
transition solver autoregressive rollout
    ↓
replay metrics / segment_metrics.csv / pred_samples.npz

依赖模块：
- numpy
- pandas
- yaml
- uwnav_dynamics.dataset.split
- uwnav_dynamics.experiment.paths
- uwnav_dynamics.solver.transition_solver

备注：
- 当前 replay 验证默认把未来控制/上下文当作已知模板，
  只递推主状态槽位 `y_in_idx`。
- 这一步用于验证“经验型状态求解器”的长序列可行性，
  不是闭环控制最终证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.dataset.split import load_split_indices
from uwnav_dynamics.experiment.paths import relative_path_str, to_snapshot_value
from uwnav_dynamics.solver.transition_solver import TrainedTransitionSolver


@dataclass(frozen=True)
class ReplayDataset:
    """replay 验证所需的连续基表与窗口索引。"""
    data_dir: Path
    base_csv: Path
    hist_len: int
    pred_len: int
    input_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    inputs: np.ndarray
    targets: np.ndarray
    idx0: np.ndarray


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


def _global_rmse_mae(y_hat: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    vals = err.reshape(-1)
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
) -> list[dict[str, Any]]:
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for dim, label in enumerate(component_labels):
        vals = err[:, dim]
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

    all_hat: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    segment_rows: list[dict[str, Any]] = []
    sample_segments: list[dict[str, Any]] = []
    nonfinite_trigger_count = 0

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
        y_hat = np.asarray(rollout.predicted_states, dtype=np.float32)
        if y_hat.shape != y_true.shape:
            raise ValueError(f"segment {segment.segment_id} prediction shape mismatch: {y_hat.shape} vs {y_true.shape}")

        nonfinite_trigger_count += int(rollout.nonfinite_trigger_count)
        all_hat.append(y_hat)
        all_true.append(y_true)

        seg_rmse, seg_mae = _global_rmse_mae(y_hat, y_true)
        first_rmse, first_mae = _global_rmse_mae(y_hat[:1, :], y_true[:1, :])
        final_rmse, final_mae = _global_rmse_mae(y_hat[-1:, :], y_true[-1:, :])
        segment_rows.append(
            {
                "segment_id": int(segment.segment_id),
                "start_idx0": int(segment.start_idx0),
                "n_steps": int(segment.n_steps),
                "rmse_global": seg_rmse,
                "mae_global": seg_mae,
                "final_step_rmse": final_rmse,
                "final_step_mae": final_mae,
                "rmse_last_over_first": _edge_ratio(first_rmse, final_rmse),
                "mae_last_over_first": _edge_ratio(first_mae, final_mae),
            }
        )
        if len(sample_segments) < int(save_samples):
            sample_segments.append(
                {
                    "segment_id": int(segment.segment_id),
                    "start_idx0": int(segment.start_idx0),
                    "n_steps": int(segment.n_steps),
                    "y_hat": y_hat,
                    "y_true": y_true,
                }
            )

    y_hat_all = np.concatenate(all_hat, axis=0)
    y_true_all = np.concatenate(all_true, axis=0)
    global_rmse, global_mae = _global_rmse_mae(y_hat_all, y_true_all)
    final_step_err = np.asarray(
        [
            np.abs(np.asarray(seg["y_hat"][-1], dtype=np.float64) - np.asarray(seg["y_true"][-1], dtype=np.float64))
            for seg in sample_segments + []  # keep type stable
        ],
        dtype=np.float64,
    )
    if len(sample_segments) < len(segment_rows):
        final_step_err = np.asarray(
            [
                np.abs(all_hat[i][-1].astype(np.float64) - all_true[i][-1].astype(np.float64))
                for i in range(len(segment_rows))
            ],
            dtype=np.float64,
        )

    all_abs_err = np.abs(np.asarray(y_hat_all - y_true_all, dtype=np.float64))
    component_rows = _component_metric_rows(
        y_hat_all,
        y_true_all,
        component_labels=replay_dataset.target_cols,
    )
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
    sample_y_hat, sample_y_true, sample_mask = _pad_sample_segments(sample_segments, dout=int(y_hat_all.shape[1]))
    segment_lengths = np.asarray([int(row["n_steps"]) for row in segment_rows], dtype=np.float64)
    finite_segments = int(nonfinite_trigger_count == 0)

    metrics = {
        "schema_version": "transition_replay_v1",
        "intended_use": "autoregressive_replay_for_transition_solver",
        "closed_loop_proof": False,
        "split": str(split_name),
        "solver_step_semantics": str(solver.step_semantics),
        "model_pred_len": int(solver.cfg_model.pred_len),
        "segment_count": int(len(segment_rows)),
        "total_steps": int(y_hat_all.shape[0]),
        "nonfinite_trigger_count": int(nonfinite_trigger_count),
        "rmse_global": global_rmse,
        "mae_global": global_mae,
        "final_step": {
            "rmse_global_mean": float(np.mean([float(row["final_step_rmse"]) for row in segment_rows])),
            "mae_global_mean": float(np.mean([float(row["final_step_mae"]) for row in segment_rows])),
            "abs_p95_global": _tail_percentiles(final_step_err, percentiles=(95.0,))["p95"],
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
            "final_step_abs_p95_global": _tail_percentiles(final_step_err, percentiles=(95.0,))["p95"],
        },
        "bias": {
            "worst_component": worst_component,
            "worst_abs_bias": worst_abs_bias,
        },
        "robustness": {
            "finite_pass_rate": float(finite_segments / max(int(len(segment_rows)), 1)),
            "segment_success_count": int(finite_segments),
            "segment_failure_count": int(len(segment_rows) - finite_segments),
        },
        "segment_stats": {
            "n_steps_mean": float(np.mean(segment_lengths)),
            "n_steps_p95": float(np.percentile(segment_lengths, 95.0)),
        },
        "component_metrics": component_rows,
    }
    return ReplayResult(
        metrics=metrics,
        segment_rows=segment_rows,
        sample_pred_y_hat=sample_y_hat,
        sample_pred_y_true=sample_y_true,
        sample_pred_mask=sample_mask,
    )


def _write_segment_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    header = (
        "segment_id,start_idx0,n_steps,rmse_global,mae_global,"
        "final_step_rmse,final_step_mae,rmse_last_over_first,mae_last_over_first"
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
        "pred_samples": "pred_samples.npz",
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
    np.savez_compressed(
        out_path / "pred_samples.npz",
        y_hat=replay_result.sample_pred_y_hat,
        y_true=replay_result.sample_pred_y_true,
        valid_mask=replay_result.sample_pred_mask,
    )
