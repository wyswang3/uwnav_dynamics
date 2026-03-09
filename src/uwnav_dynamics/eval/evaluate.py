"""
模块名称：离线数值评估主程序

模块职责：
负责加载训练阶段产出的数据划分、归一化器与 checkpoint，
执行纯数值 rollout 评估，并把“物理量纲主输出 + z-space 辅助输出”稳定落盘。

主要功能：
1. 复用训练阶段的 split / scaler artifact，执行确定性的离线评估。
2. 计算全局、group 与 horizon 级 RMSE / MAE 指标，并并行写出 dense / masked artifact。
3. 以物理量纲写出主 `pred_samples.npz`，同时保留 `pred_samples_zspace.npz` 供调参与排障。
4. 额外导出 `pred_context.npz` 与 `component_metrics*.csv`，供 component/residual 可视化复用。
5. 对经过 scaler 后仍残留在输入 `X` 中的非有限值做最小清洗，与训练消费端保持一致。

数据流：
train yaml + checkpoint + run_dir artifacts
    ↓
EvalConfig / S1PredictorConfig
    ↓
加载 features.npz / labels.npz / split_indices.npz / scalers
    ↓
model rollout(z-space) + inverse_transform(physical) + target_mask-aware metric aggregation
    ↓
metrics.yaml(metric_space + layout + supervision) +
rmse/mae_by_horizon*.csv +
component_metrics*.csv +
pred_samples.npz + pred_samples_zspace.npz + pred_context.npz
    ↓
cli/eval.py 或 cli/pipeline.py 再调起 viz 层出图

依赖模块：
- uwnav_dynamics.eval.config
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
- uwnav_dynamics.models.nets.s1_predictor

备注：
- 本模块只负责数值评估与 artifact 落盘。
- 正式用户入口为 `cli/eval.py` 与 `cli/pipeline.py`；`evaluate.py` 不直接编排绘图。
- 主指标与主样例产物使用物理量纲；z-space 指标只作为并行辅助信息保留。
- runtime mask 的唯一执行真源是评估 batch 中的 `target_mask`。
- 输入 `X` 中由稀疏辅助通道保留的 NaN 不会回写 dataset artifact，
  仅在评估消费端被清洗为 0.0（对应 z-score 后的 train 均值）。
"""

# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, List, Sequence

import numpy as np
import torch
import yaml

from uwnav_dynamics.dataset.normalize import inverse_transform, load_scaler, transform
from uwnav_dynamics.dataset.split import load_split_indices
from uwnav_dynamics.eval.config import EvalConfig, build_eval_config
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.models.utils.execution_layout import (
    build_execution_layout_metadata,
    extract_y0_from_x_last,
)
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.utils.semantic_output_layout import (
    SEMANTIC_LAYOUT_SCHEMA_VERSION,
    SemanticOutputLayout,
    build_semantic_layout_metadata,
    resolve_semantic_output_layout,
)
from uwnav_dynamics.supervision_mask import build_dense_target_mask, build_target_mask_from_dvl_mask


SUPERVISION_SCHEMA_VERSION = "supervision_v1"
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


# =============================================================================
# Main evaluation
# =============================================================================

@torch.no_grad()
def evaluate_once(*, cfg_eval: EvalConfig, cfg_model: S1PredictorConfig) -> Dict[str, Any]:
    """执行一次完整数值评估并返回待落盘的指标与样本 artifact。"""
    device = torch.device(cfg_eval.device)

    # 先完整恢复 features / labels，再在评估侧按 split 做切片；
    # 这样能严格复用训练阶段保存的 split/scaler 真源。
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

    n, L, Din = X.shape
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
    if cfg_eval.split_name not in split_indices:
        raise KeyError(f"split '{cfg_eval.split_name}' not found in {cfg_eval.split_indices_path}")
    idx = np.asarray(split_indices[cfg_eval.split_name], dtype=np.int64)
    if idx.size == 0:
        raise RuntimeError(f"Split {cfg_eval.split_name} is empty. Check ratios.")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"Split indices out of range for n={n}: {cfg_eval.split_indices_path}")

    x_scaler = load_scaler(cfg_eval.x_scaler_path)
    y_scaler = load_scaler(cfg_eval.y_scaler_path)

    # 评估与训练一样在 z-space 中执行模型前向；`copy()` 仅用于避开只读 numpy view 的 warning。
    Xs = transform(np.array(X[idx], copy=True), x_scaler).astype(np.float32, copy=False)
    Xs = _sanitize_scaled_inputs_for_eval(Xs)
    Ys = transform(np.array(Y[idx], copy=True), y_scaler).astype(np.float32, copy=False)
    target_mask_np = np.asarray(target_mask_all_np[idx], dtype=bool)

    model = S1Predictor(cfg_model).to(device)
    ckpt = _load_checkpoint(cfg_eval.ckpt, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    bs = int(cfg_eval.batch_size)
    n_eval = Xs.shape[0]

    yhat_list: List[torch.Tensor] = []
    ytrue_list: List[torch.Tensor] = []
    logvar_list: List[torch.Tensor] = []
    target_mask_list: List[torch.Tensor] = []

    # 批量 rollout 只负责“前向 + y0 恢复 + delta 累积”，
    # 所有指标都放到循环外统一计算，避免 batch 粒度聚合误差。
    for s in range(0, n_eval, bs):
        e = min(n_eval, s + bs)
        xb = torch.from_numpy(Xs[s:e]).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(Ys[s:e]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(target_mask_np[s:e]).to(device=device, dtype=torch.bool)

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
        target_mask_list.append(mb.cpu())

    y_hat_all = torch.cat(yhat_list, dim=0)
    y_true_all = torch.cat(ytrue_list, dim=0)
    logvar_all = torch.cat(logvar_list, dim=0)
    target_mask_all = torch.cat(target_mask_list, dim=0)

    y_hat_z_np = y_hat_all.numpy()
    y_true_z_np = y_true_all.numpy()
    logvar_z_np = logvar_all.numpy()
    target_mask_np_eval = target_mask_all.numpy()

    # 物理量纲指标是主输出；z-space 指标只作为训练/调参的并行辅助信息保留。
    y_hat_phys_np = inverse_transform(y_hat_z_np, y_scaler).astype(np.float32, copy=False)
    y_true_phys_np = inverse_transform(y_true_z_np, y_scaler).astype(np.float32, copy=False)
    logvar_phys_np = _convert_logvar_to_physical(logvar_z_np, y_scaler).astype(np.float32, copy=False)

    y_hat_phys = torch.from_numpy(y_hat_phys_np)
    y_true_phys = torch.from_numpy(y_true_phys_np)

    rmse_hd, mae_hd = _rmse_mae_by_horizon(y_hat_phys, y_true_phys)
    rmse_hd_masked, mae_hd_masked = _rmse_mae_by_horizon_masked(y_hat_phys, y_true_phys, target_mask_all)
    rmse_groups = _aggregate_group_curve(rmse_hd, semantic_layout.group_indices)
    mae_groups = _aggregate_group_curve(mae_hd, semantic_layout.group_indices)
    rmse_groups_masked = _aggregate_group_curve(rmse_hd_masked, semantic_layout.group_indices)
    mae_groups_masked = _aggregate_group_curve(mae_hd_masked, semantic_layout.group_indices)
    rmse_global, mae_global = _global_rmse_mae(y_hat_phys, y_true_phys)
    rmse_global_masked, mae_global_masked = _global_rmse_mae(y_hat_phys, y_true_phys, target_mask_all)

    rmse_hd_z, mae_hd_z = _rmse_mae_by_horizon(y_hat_all, y_true_all)
    rmse_hd_masked_z, mae_hd_masked_z = _rmse_mae_by_horizon_masked(y_hat_all, y_true_all, target_mask_all)
    rmse_groups_z = _aggregate_group_curve(rmse_hd_z, semantic_layout.group_indices)
    mae_groups_z = _aggregate_group_curve(mae_hd_z, semantic_layout.group_indices)
    rmse_groups_masked_z = _aggregate_group_curve(rmse_hd_masked_z, semantic_layout.group_indices)
    mae_groups_masked_z = _aggregate_group_curve(mae_hd_masked_z, semantic_layout.group_indices)
    rmse_global_z, mae_global_z = _global_rmse_mae(y_hat_all, y_true_all)
    rmse_global_masked_z, mae_global_masked_z = _global_rmse_mae(y_hat_all, y_true_all, target_mask_all)

    component_labels, component_display_labels, component_units = _component_metadata(semantic_layout)
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

    # 样本类 artifact 只保留前 `save_samples` 个窗口，避免评估目录无限膨胀。
    n_samp = int(min(cfg_eval.save_samples, y_hat_all.shape[0]))
    samp = {
        "y_hat": y_hat_phys_np[:n_samp],
        "y_true": y_true_phys_np[:n_samp],
        "logvar": logvar_phys_np[:n_samp],
    }
    samp_z = {
        "y_hat": y_hat_z_np[:n_samp],
        "y_true": y_true_z_np[:n_samp],
        "logvar": logvar_z_np[:n_samp],
    }
    pred_context = {
        "target_mask": target_mask_np_eval[:n_samp].astype(bool, copy=False),
        "sample_index": np.asarray(idx[:n_samp], dtype=np.int64),
        "component_labels": np.asarray(component_labels, dtype=str),
        "component_display_labels": np.asarray(component_display_labels, dtype=str),
        "component_units": np.asarray(component_units, dtype=str),
    }

    return {
        "n_total": int(n),
        "n_eval": int(n_eval),
        "split": cfg_eval.split_name,
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
        "semantic_layout": semantic_layout,
        "raw_mask_source": raw_mask_source,
        "samples": samp,
        "samples_zspace": samp_z,
        "pred_context": pred_context,
    }


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
    )

    _ensure_dir(cfg_eval.out_dir)
    res = evaluate_once(cfg_eval=cfg_eval, cfg_model=cfg_model)

    # `metrics.yaml` 是评估目录的总索引；CSV/NPZ 则承载大体量数组 artifact。
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
        "artifacts": {
            "primary_samples": "pred_samples.npz",
            "auxiliary_samples_zspace": "pred_samples_zspace.npz",
            "sample_context": "pred_context.npz",
            "component_metrics_csv": "component_metrics.csv",
            "component_metrics_masked_csv": "component_metrics_masked.csv",
            "component_metrics_zspace_csv": "component_metrics_zspace.csv",
            "component_metrics_masked_zspace_csv": "component_metrics_masked_zspace.csv",
            "rmse_horizon_zspace_csv": "rmse_by_horizon_zspace.csv",
            "mae_horizon_zspace_csv": "mae_by_horizon_zspace.csv",
            "rmse_horizon_masked_zspace_csv": "rmse_by_horizon_masked_zspace.csv",
            "mae_horizon_masked_zspace_csv": "mae_by_horizon_masked_zspace.csv",
        },
        "cfg": {
            "data_dir": str(cfg_eval.data_dir),
            "ckpt": str(cfg_eval.ckpt),
            "device": cfg_eval.device,
            "batch_size": cfg_eval.batch_size,
            "split_indices": str(cfg_eval.split_indices_path),
            "x_scaler": str(cfg_eval.x_scaler_path),
            "y_scaler": str(cfg_eval.y_scaler_path),
            "y0_source": cfg_eval.y0_source,
            "mode": cfg_eval.mode,
        },
    }
    with open(cfg_eval.out_dir / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)

    # horizon / component CSV 保持“物理量纲主、z-space 并行保留”的双轨产物结构。
    _write_csv_hd(cfg_eval.out_dir / "rmse_by_horizon.csv", res["rmse_hd"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "mae_by_horizon.csv", res["mae_hd"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "rmse_by_horizon_masked.csv", res["rmse_hd_masked"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "mae_by_horizon_masked.csv", res["mae_hd_masked"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "rmse_by_horizon_zspace.csv", res["rmse_hd_zspace"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "mae_by_horizon_zspace.csv", res["mae_hd_zspace"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "rmse_by_horizon_masked_zspace.csv", res["rmse_hd_masked_zspace"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "mae_by_horizon_masked_zspace.csv", res["mae_hd_masked_zspace"], col_prefix="d")
    _write_component_metric_csv(cfg_eval.out_dir / "component_metrics.csv", res["component_metrics"])
    _write_component_metric_csv(cfg_eval.out_dir / "component_metrics_masked.csv", res["component_metrics_masked"])
    _write_component_metric_csv(cfg_eval.out_dir / "component_metrics_zspace.csv", res["component_metrics_zspace"])
    _write_component_metric_csv(cfg_eval.out_dir / "component_metrics_masked_zspace.csv", res["component_metrics_masked_zspace"])

    # 样本级 NPZ 分成 physical / zspace / context 三类，供后续不同绘图脚本按需读盘。
    pred_npz = cfg_eval.out_dir / "pred_samples.npz"
    np.savez_compressed(
        pred_npz,
        y_hat=res["samples"]["y_hat"],
        y_true=res["samples"]["y_true"],
        logvar=res["samples"]["logvar"],
    )
    pred_npz_z = cfg_eval.out_dir / "pred_samples_zspace.npz"
    np.savez_compressed(
        pred_npz_z,
        y_hat=res["samples_zspace"]["y_hat"],
        y_true=res["samples_zspace"]["y_true"],
        logvar=res["samples_zspace"]["logvar"],
    )
    pred_context_npz = cfg_eval.out_dir / "pred_context.npz"
    np.savez_compressed(
        pred_context_npz,
        target_mask=res["pred_context"]["target_mask"],
        sample_index=res["pred_context"]["sample_index"],
        component_labels=res["pred_context"]["component_labels"],
        component_display_labels=res["pred_context"]["component_display_labels"],
        component_units=res["pred_context"]["component_units"],
    )

    print(f"[EVAL] split={res['split']}  n_eval={res['n_eval']}")
    print(f"[EVAL] RMSE(global)={res['rmse_global']:.6f}  MAE(global)={res['mae_global']:.6f}")
    print(f"[EVAL] wrote: {cfg_eval.out_dir / 'metrics.yaml'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
