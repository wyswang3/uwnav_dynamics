"""
模块名称：经典基线运行器

模块职责：
负责在与主模型完全一致的数据划分、归一化与评估契约下，
运行 trivial baseline 与 classical linear baseline，
并把结果写成与神经网络评估一致的 artifact。

主要功能：
1. 复用 train yaml、split/scaler 与 eval artifact 合同。
2. 提供 `trivial_last` 与 `classical_ridge` 两类 baseline。
3. 为每个 baseline 生成独立 run_dir、state snapshot、resolved config 与 eval 产物。
4. 汇总所有 baseline 的指标到 `summary.csv`，并可选输出 compare 图。

数据流：
train yaml
    ↓
prepare_train_data() 生成 split/scaler
    ↓
load_eval_artifacts() / slice_eval_artifacts()
    ↓
baseline fit / predict
    ↓
summarize_eval_predictions()
    ↓
write_eval_outputs()
    ↓
manifest.yaml / summary.csv / compare plots

依赖模块：
- numpy
- yaml
- uwnav_dynamics.eval.evaluate
- uwnav_dynamics.train.data_pipeline
- uwnav_dynamics.experiment.reporting

备注：
- baseline runner 不依赖 checkpoint，所有预测器状态都直接写成 `baseline_state.npz`。
- `classical_ridge` 使用 train split 拟合、val split 选超参，不会把 test 信息泄漏回模型选择阶段。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from uwnav_dynamics.eval.config import EvalConfig
from uwnav_dynamics.eval.evaluate import (
    load_eval_artifacts,
    slice_eval_artifacts,
    summarize_eval_predictions,
    write_eval_outputs,
)
from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.experiment.paths import relative_path_str
from uwnav_dynamics.experiment.reporting import EVAL_SUMMARY_FIELDS, flatten_eval_metrics
from uwnav_dynamics.train.config import TrainYamlConfig, load_train_config
from uwnav_dynamics.train.data_pipeline import PreparedTrainData, prepare_train_data
from uwnav_dynamics.train.runtime import set_global_seed


_BASELINE_CHOICES = ("trivial_last", "classical_ridge")


@dataclass(frozen=True)
class BaselineOutcome:
    """单个 baseline 运行完成后的摘要。"""
    name: str
    label: str
    kind: str
    run_dir: Path
    eval_dir: Path
    state_path: Path
    resolved_config_path: Path
    selected_alpha: float | None
    val_rmse_zspace: float | None
    split_strategy: str
    dropped_window_count: int


def _copy_source_train_yaml(source_yaml: Path, run_dir: Path) -> Path:
    dst = run_dir / "source_train.yaml"
    shutil.copyfile(source_yaml, dst)
    return dst


def _write_resolved_baseline_yaml(
    path: Path,
    *,
    cfg_train: TrainYamlConfig,
    baseline_kind: str,
    eval_split: str,
    source_train_yaml: Path,
    state_path: Path,
    prepared: PreparedTrainData,
    selected_alpha: float | None,
    val_rmse_zspace: float | None,
    path_root: Path,
) -> Path:
    payload = {
        "schema_version": "baseline_resolved_v1",
        "baseline": {
            "kind": baseline_kind,
            "eval_split": eval_split,
            "selected_alpha": selected_alpha,
            "val_rmse_zspace": val_rmse_zspace,
        },
        "run": {
            "name": cfg_train.run.name,
            "seed": cfg_train.run.seed,
            "out_dir": relative_path_str(path.parent.parent, base_dir=path_root),
            "variant": path.parent.name,
        },
        "data": {
            "data_dir": relative_path_str(cfg_train.data.data_dir, base_dir=path_root),
            "batch_size": cfg_train.data.batch_size,
            "split_strategy": prepared.split_strategy,
            "split_sizes": {str(k): int(v) for k, v in prepared.split_sizes.items()},
            "dropped_window_count": int(prepared.dropped_window_count),
        },
        "artifacts": {
            "source_train_yaml": relative_path_str(source_train_yaml, base_dir=path_root),
            "baseline_state": relative_path_str(state_path, base_dir=path_root),
            "split_indices": relative_path_str(path.parent / "split_indices.npz", base_dir=path_root),
            "x_scaler": relative_path_str(path.parent / "scalers" / "x_scaler.npz", base_dir=path_root),
            "y_scaler": relative_path_str(path.parent / "scalers" / "y_scaler.npz", base_dir=path_root),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path


def _repeat_last_state(split_data, y_in_idx: Sequence[int], pred_len: int) -> np.ndarray:
    last_state = np.asarray(split_data.X_scaled[:, -1, list(y_in_idx)], dtype=np.float32)
    return np.repeat(last_state[:, None, :], int(pred_len), axis=1)


def _flatten_history(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)


def _flatten_target(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)


def _solve_ridge_multioutput(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x2 = np.asarray(x, dtype=np.float64)
    y2 = np.asarray(y, dtype=np.float64)
    if x2.ndim != 2 or y2.ndim != 2 or x2.shape[0] != y2.shape[0]:
        raise ValueError(f"x/y shape mismatch: {x2.shape} vs {y2.shape}")

    x_aug = np.concatenate([x2, np.ones((x2.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(alpha) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y2
    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    weight = coeff[:-1]
    bias = coeff[-1]
    return weight.astype(np.float64, copy=False), bias.astype(np.float64, copy=False)


def _predict_ridge(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)


def _masked_rmse_np(y_hat: np.ndarray, y_true: np.ndarray, target_mask: np.ndarray) -> float:
    err = np.asarray(y_hat - y_true, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    vals = err[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals * vals)))


def _estimate_constant_logvar(
    *,
    y_hat: np.ndarray,
    y_true: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    err2 = np.square(np.asarray(y_hat - y_true, dtype=np.float64))
    if target_mask is None:
        var = np.mean(err2, axis=0)
    else:
        mask = np.asarray(target_mask, dtype=bool)
        valid = mask.sum(axis=0)
        var = np.full(err2.shape[1:], 1e-6, dtype=np.float64)
        good = valid > 0
        if np.any(good):
            var[good] = (err2 * mask).sum(axis=0)[good] / valid[good]
    var = np.maximum(var, 1e-6)
    return np.log(var).astype(np.float32, copy=False)


def _broadcast_logvar(logvar_hd: np.ndarray, *, n_samples: int) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(logvar_hd, dtype=np.float32)[None, :, :],
        (int(n_samples),) + tuple(logvar_hd.shape),
    ).copy()


def _fit_trivial_last(*, train_split, cfg_train: TrainYamlConfig) -> dict[str, Any]:
    y_hat_train = _repeat_last_state(train_split, cfg_train.model.y_in_idx, cfg_train.model.pred_len)
    logvar_hd = _estimate_constant_logvar(
        y_hat=y_hat_train,
        y_true=train_split.Y_scaled,
        target_mask=train_split.target_mask,
    )
    return {
        "kind": "trivial_last",
        "logvar_hd": logvar_hd,
        "selected_alpha": None,
        "val_rmse_zspace": None,
        "state_payload": {
            "kind": np.asarray("trivial_last"),
            "logvar_hd": logvar_hd,
        },
    }


def _fit_classical_ridge(
    *,
    train_split,
    val_split,
    alpha_grid: Sequence[float],
) -> dict[str, Any]:
    x_train = _flatten_history(train_split.X_scaled)
    y_train = _flatten_target(train_split.Y_scaled)
    x_val = _flatten_history(val_split.X_scaled)

    best_alpha: float | None = None
    best_rmse = float("inf")
    best_weight: np.ndarray | None = None
    best_bias: np.ndarray | None = None
    for alpha in alpha_grid:
        weight, bias = _solve_ridge_multioutput(x_train, y_train, float(alpha))
        pred_val = _predict_ridge(x_val, weight, bias).reshape(val_split.Y_scaled.shape)
        rmse = _masked_rmse_np(pred_val, val_split.Y_scaled, val_split.target_mask)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_alpha = float(alpha)
            best_weight = weight
            best_bias = bias

    if best_weight is None or best_bias is None or best_alpha is None:
        raise RuntimeError("classical_ridge failed to select a valid alpha")

    pred_train = _predict_ridge(x_train, best_weight, best_bias).reshape(train_split.Y_scaled.shape)
    logvar_hd = _estimate_constant_logvar(
        y_hat=pred_train,
        y_true=train_split.Y_scaled,
        target_mask=train_split.target_mask,
    )
    return {
        "kind": "classical_ridge",
        "weight": best_weight.astype(np.float32, copy=False),
        "bias": best_bias.astype(np.float32, copy=False),
        "logvar_hd": logvar_hd,
        "selected_alpha": best_alpha,
        "val_rmse_zspace": best_rmse,
        "state_payload": {
            "kind": np.asarray("classical_ridge"),
            "selected_alpha": np.asarray(best_alpha, dtype=np.float32),
            "val_rmse_zspace": np.asarray(best_rmse, dtype=np.float32),
            "weight": best_weight.astype(np.float32, copy=False),
            "bias": best_bias.astype(np.float32, copy=False),
            "logvar_hd": logvar_hd,
        },
    }


def _predict_baseline(model_state: dict[str, Any], *, split_data, cfg_train: TrainYamlConfig) -> tuple[np.ndarray, np.ndarray]:
    kind = str(model_state["kind"])
    if kind == "trivial_last":
        y_hat = _repeat_last_state(split_data, cfg_train.model.y_in_idx, cfg_train.model.pred_len)
    elif kind == "classical_ridge":
        x_eval = _flatten_history(split_data.X_scaled)
        y_hat = _predict_ridge(x_eval, model_state["weight"], model_state["bias"]).reshape(split_data.Y_scaled.shape)
        y_hat = y_hat.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported baseline kind={kind!r}")

    logvar = _broadcast_logvar(model_state["logvar_hd"], n_samples=y_hat.shape[0])
    return np.asarray(y_hat, dtype=np.float32), logvar


def _baseline_specs(baselines: Iterable[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for baseline in baselines:
        kind = str(baseline).strip().lower()
        if kind not in _BASELINE_CHOICES:
            raise ValueError(f"Unsupported baseline={kind!r}; allowed={list(_BASELINE_CHOICES)}")
        label = "Trivial Last" if kind == "trivial_last" else "Classical Ridge"
        specs.append((kind, label))
    return specs


def _write_manifest(path: Path, *, source_yaml: Path, work_dir: Path, outcomes: Sequence[BaselineOutcome], path_root: Path) -> None:
    payload = {
        "schema_version": "baseline_manifest_v1",
        "source_train_yaml": relative_path_str(source_yaml, base_dir=path_root),
        "work_dir": relative_path_str(work_dir, base_dir=path_root),
        "runs": [
            {
                "name": outcome.name,
                "label": outcome.label,
                "kind": outcome.kind,
                "run_dir": relative_path_str(outcome.run_dir, base_dir=path_root),
                "eval_dir": relative_path_str(outcome.eval_dir, base_dir=path_root),
                "state_path": relative_path_str(outcome.state_path, base_dir=path_root),
                "resolved_config_path": relative_path_str(outcome.resolved_config_path, base_dir=path_root),
            }
            for outcome in outcomes
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_summary(path: Path, *, outcomes: Sequence[BaselineOutcome], path_root: Path) -> None:
    fieldnames = [
        "name",
        "label",
        "kind",
        "run_dir",
        "eval_dir",
        "state_path",
        "resolved_config_path",
        "split_strategy",
        "dropped_window_count",
        "selected_alpha",
        "val_rmse_zspace",
    ] + EVAL_SUMMARY_FIELDS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in outcomes:
            row = {
                "name": outcome.name,
                "label": outcome.label,
                "kind": outcome.kind,
                "run_dir": relative_path_str(outcome.run_dir, base_dir=path_root),
                "eval_dir": relative_path_str(outcome.eval_dir, base_dir=path_root),
                "state_path": relative_path_str(outcome.state_path, base_dir=path_root),
                "resolved_config_path": relative_path_str(outcome.resolved_config_path, base_dir=path_root),
                "split_strategy": outcome.split_strategy,
                "dropped_window_count": outcome.dropped_window_count,
                "selected_alpha": "" if outcome.selected_alpha is None else outcome.selected_alpha,
                "val_rmse_zspace": "" if outcome.val_rmse_zspace is None else outcome.val_rmse_zspace,
            }
            row.update({field: "" for field in EVAL_SUMMARY_FIELDS})
            metrics_path = outcome.eval_dir / "metrics.yaml"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    metrics = yaml.safe_load(mf) or {}
                row.update(flatten_eval_metrics(metrics))
            writer.writerow(row)


def _maybe_write_compare_plots(
    *,
    outcomes: Sequence[BaselineOutcome],
    work_dir: Path,
    split: str,
    dt: float,
    x_axis: str,
    plot_fmt: str,
) -> None:
    if len(outcomes) < 2:
        return
    from uwnav_dynamics.viz.eval.plot_control_readiness import ControlReadinessPlotCfg, plot_control_readiness
    from uwnav_dynamics.viz.eval.plot_model_compare import ModelCompareCfg, plot_horizon_compare

    eval_dirs = [outcome.eval_dir for outcome in outcomes]
    labels = [outcome.label for outcome in outcomes]
    roles = ["baseline"] * len(outcomes)
    compare_dir = work_dir / f"compare_{split}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("rmse", "mae"):
        plot_horizon_compare(
            eval_dirs=eval_dirs,
            labels=labels,
            roles=roles,
            out_dir=compare_dir,
            cfg=ModelCompareCfg(
                dt_s=float(dt),
                use_seconds=(x_axis == "sec"),
                metric=metric,
                fmt=plot_fmt,
            ),
        )
    plot_control_readiness(
        eval_dirs=eval_dirs,
        labels=labels,
        roles=roles,
        out_dir=compare_dir,
        cfg=ControlReadinessPlotCfg(fmt=plot_fmt),
    )


def run_baseline_suite(
    *,
    train_yaml: Path,
    baselines: Sequence[str],
    split: str,
    work_dir: Path | None,
    alpha_grid: Sequence[float],
    save_samples: int,
    compare: bool,
    dt: float,
    x_axis: str,
    plot_fmt: str,
) -> Path:
    """运行一组 baseline，并返回 `summary.csv` 路径。"""
    cfg_train = load_train_config(train_yaml)
    resolved_work_dir = Path(work_dir) if work_dir is not None else (Path(cfg_train.run.out_dir) / "baseline_runs")
    runs_root = resolved_work_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    specs = _baseline_specs(baselines)
    outcomes: list[BaselineOutcome] = []
    for baseline_kind, baseline_label in specs:
        set_global_seed(cfg_train.run.seed)
        variant = f"{cfg_train.run.variant}__{baseline_kind}"
        run_layout = RunLayout(out_dir=runs_root, variant=variant)
        run_dir = run_layout.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        source_snapshot = _copy_source_train_yaml(train_yaml, run_dir)

        prepared = prepare_train_data(cfg_train.data, run_layout)
        cfg_eval = EvalConfig(
            data_dir=Path(cfg_train.data.data_dir),
            ckpt=run_dir / "baseline_state.npz",
            out_dir=run_layout.eval_dir(split),
            device="cpu",
            batch_size=int(cfg_train.data.batch_size),
            split_name=split,
            split_indices_path=run_layout.split_indices_path,
            x_scaler_path=run_layout.x_scaler_path,
            y_scaler_path=run_layout.y_scaler_path,
            y0_source=str(cfg_train.rollout.y0_source),
            mode=str(cfg_train.rollout.mode),
            save_samples=int(save_samples),
        )
        loaded = load_eval_artifacts(cfg_eval=cfg_eval, cfg_model=cfg_train.model)
        train_split = slice_eval_artifacts(loaded, split_name="train")
        val_split = slice_eval_artifacts(loaded, split_name="val")
        eval_split = slice_eval_artifacts(loaded, split_name=split)

        if baseline_kind == "trivial_last":
            model_state = _fit_trivial_last(train_split=train_split, cfg_train=cfg_train)
        elif baseline_kind == "classical_ridge":
            model_state = _fit_classical_ridge(
                train_split=train_split,
                val_split=val_split,
                alpha_grid=tuple(float(v) for v in alpha_grid),
            )
        else:
            raise ValueError(f"Unsupported baseline kind={baseline_kind!r}")

        state_path = run_dir / "baseline_state.npz"
        np.savez_compressed(state_path, **model_state["state_payload"])
        resolved_cfg_path = _write_resolved_baseline_yaml(
            run_dir / "resolved_baseline.yaml",
            cfg_train=cfg_train,
            baseline_kind=baseline_kind,
            eval_split=split,
            source_train_yaml=source_snapshot,
            state_path=state_path,
            prepared=prepared,
            selected_alpha=model_state["selected_alpha"],
            val_rmse_zspace=model_state["val_rmse_zspace"],
            path_root=resolved_work_dir,
        )

        y_hat_eval, logvar_eval = _predict_baseline(model_state, split_data=eval_split, cfg_train=cfg_train)
        res = summarize_eval_predictions(
            split_artifacts=eval_split,
            y_hat_z_np=y_hat_eval,
            y_true_z_np=np.asarray(eval_split.Y_scaled, dtype=np.float32),
            logvar_z_np=logvar_eval,
            save_samples=int(save_samples),
            late_horizon_fraction=float(cfg_eval.long_horizon_fraction),
            trace_seconds=float(cfg_eval.trace_seconds),
            trace_dt_s=float(cfg_eval.trace_dt_s),
        )
        write_eval_outputs(
            out_dir=cfg_eval.out_dir,
            res=res,
            cfg_model=cfg_train.model,
            cfg_snapshot={
                "data_dir": cfg_eval.data_dir,
                "ckpt": "",
                "device": cfg_eval.device,
                "batch_size": cfg_eval.batch_size,
                "split_indices": cfg_eval.split_indices_path,
                "x_scaler": cfg_eval.x_scaler_path,
                "y_scaler": cfg_eval.y_scaler_path,
                "y0_source": cfg_eval.y0_source,
                "mode": cfg_eval.mode,
                "late_horizon_fraction": cfg_eval.long_horizon_fraction,
                "trace_seconds": cfg_eval.trace_seconds,
                "trace_dt_s": cfg_eval.trace_dt_s,
                "predictor": {
                    "type": "baseline",
                    "kind": baseline_kind,
                    "state_path": state_path,
                    "selected_alpha": model_state["selected_alpha"],
                },
            },
            path_root=resolved_work_dir,
        )
        outcomes.append(
            BaselineOutcome(
                name=variant,
                label=baseline_label,
                kind=baseline_kind,
                run_dir=run_dir,
                eval_dir=cfg_eval.out_dir,
                state_path=state_path,
                resolved_config_path=resolved_cfg_path,
                selected_alpha=model_state["selected_alpha"],
                val_rmse_zspace=model_state["val_rmse_zspace"],
                split_strategy=prepared.split_strategy,
                dropped_window_count=prepared.dropped_window_count,
            )
        )

    summary_path = resolved_work_dir / "summary.csv"
    _write_manifest(
        resolved_work_dir / "manifest.yaml",
        source_yaml=train_yaml,
        work_dir=resolved_work_dir,
        outcomes=outcomes,
        path_root=resolved_work_dir,
    )
    _write_summary(summary_path, outcomes=outcomes, path_root=resolved_work_dir)
    if compare:
        _maybe_write_compare_plots(
            outcomes=outcomes,
            work_dir=resolved_work_dir,
            split=split,
            dt=float(dt),
            x_axis=str(x_axis),
            plot_fmt=str(plot_fmt),
        )
    return summary_path
