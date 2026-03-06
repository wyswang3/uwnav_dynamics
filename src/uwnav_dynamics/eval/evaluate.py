"""
模块名称：离线数值评估主程序

模块职责：
负责加载训练阶段产出的数据划分、归一化器与 checkpoint，
执行纯数值 rollout 评估并将指标与样例产物落盘。

主要功能：
1. 复用训练阶段的 split / scaler artifact，执行确定性的离线评估。
2. 计算全局与 horizon 级 RMSE / MAE 指标并写出 metrics 与 CSV。
3. 导出供 viz 层复用的 `pred_samples.npz`，并在 `metrics.yaml` 中写入最小 layout metadata。

数据流：
train yaml + checkpoint + run_dir artifacts
    ↓
EvalConfig / S1PredictorConfig
    ↓
加载 features.npz / labels.npz / split_indices.npz / scalers
    ↓
model rollout + metric aggregation
    ↓
metrics.yaml(layout.execution + layout.semantic) + rmse_by_horizon.csv + mae_by_horizon.csv + pred_samples.npz
    ↓
cli/eval.py 或 cli/pipeline.py 再调起 viz 层出图

依赖模块：
- uwnav_dynamics.eval.config
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.dataset.split
- uwnav_dynamics.models.nets.s1_predictor

备注：
- 本模块只负责数值评估与 artifact 落盘。
- 旧 `--plots` 路径已显式弃用，正式用户入口为 `cli/eval.py` 与 `cli/pipeline.py`。
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

from uwnav_dynamics.dataset.normalize import load_scaler, transform
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
    canonical_semantic_output_layout,
    validate_target_cols_against_semantic_layout,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _np_load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


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


def _aggregate_group_curve(metric_hd: np.ndarray, groups: Dict[str, Sequence[int]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key, indices in groups.items():
        out[key] = metric_hd[:, list(indices)].mean(axis=1).tolist()
    return out


def _write_csv_hd(path: Path, hd: np.ndarray, col_prefix: str = "d") -> None:
    H, D = hd.shape
    header = ",".join(["h"] + [f"{col_prefix}{i}" for i in range(D)])
    lines = [header]
    for h in range(H):
        row = ",".join([str(h + 1)] + [f"{hd[h, d]:.8f}" for d in range(D)])
        lines.append(row)
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_target_cols(label_npz: Dict[str, Any]) -> tuple[str, ...] | None:
    if "target_cols" not in label_npz:
        return None
    target_cols = np.asarray(label_npz["target_cols"])
    if target_cols.ndim == 0:
        return (str(target_cols.item()),)
    return tuple(str(x) for x in target_cols.tolist())


def _resolve_semantic_layout(label_npz: Dict[str, Any], dout: int) -> SemanticOutputLayout:
    layout = canonical_semantic_output_layout(dout)
    target_cols = _extract_target_cols(label_npz)
    if target_cols is None:
        return layout
    return validate_target_cols_against_semantic_layout(target_cols, layout)


# =============================================================================
# Main evaluation
# =============================================================================

@torch.no_grad()
def evaluate_once(*, cfg_eval: EvalConfig, cfg_model: S1PredictorConfig) -> Dict[str, Any]:
    device = torch.device(cfg_eval.device)

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

    # copy()：避免 torch.from_numpy 的只读 warning
    Xs = transform(np.array(X[idx], copy=True), x_scaler).astype(np.float32, copy=False)
    Ys = transform(np.array(Y[idx], copy=True), y_scaler).astype(np.float32, copy=False)

    model = S1Predictor(cfg_model).to(device)
    ckpt = _load_checkpoint(cfg_eval.ckpt, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    bs = int(cfg_eval.batch_size)
    n_eval = Xs.shape[0]

    yhat_list: List[torch.Tensor] = []
    ytrue_list: List[torch.Tensor] = []
    logvar_list: List[torch.Tensor] = []

    for s in range(0, n_eval, bs):
        e = min(n_eval, s + bs)
        xb = torch.from_numpy(Xs[s:e]).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(Ys[s:e]).to(device=device, dtype=torch.float32)

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

    rmse_hd, mae_hd = _rmse_mae_by_horizon(y_hat_all, y_true_all)
    rmse_groups = _aggregate_group_curve(rmse_hd, semantic_layout.group_indices)
    mae_groups = _aggregate_group_curve(mae_hd, semantic_layout.group_indices)
    rmse_global = float(rmse_hd.mean())
    mae_global = float(mae_hd.mean())

    n_samp = int(min(cfg_eval.save_samples, y_hat_all.shape[0]))
    samp = {
        "y_hat": y_hat_all[:n_samp].numpy(),
        "y_true": y_true_all[:n_samp].numpy(),
        "logvar": logvar_all[:n_samp].numpy(),
    }

    return {
        "n_total": int(n),
        "n_eval": int(n_eval),
        "split": cfg_eval.split_name,
        "rmse_global": rmse_global,
        "mae_global": mae_global,
        "rmse_hd": rmse_hd,
        "mae_hd": mae_hd,
        "rmse_groups": rmse_groups,
        "mae_groups": mae_groups,
        "semantic_layout": semantic_layout,
        "samples": samp,
    }


def main() -> int:
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

    # ---- write metrics.yaml ----
    metrics = {
        "split": res["split"],
        "n_total": res["n_total"],
        "n_eval": res["n_eval"],
        "rmse_global": res["rmse_global"],
        "mae_global": res["mae_global"],
        "rmse_groups": res["rmse_groups"],
        "mae_groups": res["mae_groups"],
        "layout": {
            "schema_version": SEMANTIC_LAYOUT_SCHEMA_VERSION,
            "execution": build_execution_layout_metadata(cfg_model.y_in_idx),
            "semantic": build_semantic_layout_metadata(res["semantic_layout"]),
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

    # ---- write csv ----
    _write_csv_hd(cfg_eval.out_dir / "rmse_by_horizon.csv", res["rmse_hd"], col_prefix="d")
    _write_csv_hd(cfg_eval.out_dir / "mae_by_horizon.csv", res["mae_hd"], col_prefix="d")

    # ---- write samples ----
    pred_npz = cfg_eval.out_dir / "pred_samples.npz"
    np.savez_compressed(
        pred_npz,
        y_hat=res["samples"]["y_hat"],
        y_true=res["samples"]["y_true"],
        logvar=res["samples"]["logvar"],
    )

    print(f"[EVAL] split={res['split']}  n_eval={res['n_eval']}")
    print(f"[EVAL] RMSE(global)={res['rmse_global']:.6f}  MAE(global)={res['mae_global']:.6f}")
    print(f"[EVAL] wrote: {cfg_eval.out_dir / 'metrics.yaml'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
