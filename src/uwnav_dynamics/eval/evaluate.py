# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping

import numpy as np
import torch
import yaml

from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data import build_loaders, DataConfig
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.rollout import extract_y0_from_x_last, rollout_from_delta

# viz（可选）
from uwnav_dynamics.viz.eval.plot_horizon_metrics import plot_groups_vs_horizon, HorizonPlotCfg
from uwnav_dynamics.viz.eval.plot_rollout_samples import plot_rollout_samples_from_npz


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class EvalConfig:
    data_dir: Path
    ckpt: Path
    out_dir: Path

    device: str = "cpu"
    batch_size: int = 512
    split_name: str = "test"  # "train" | "val" | "test"

    save_samples: int = 256

    # plotting
    make_plots: bool = False
    plot_fmt: str = "png"      # "png" | "pdf" | "both"
    dt_s: float = 0.01
    x_axis: str = "sec"        # "sec" | "step"
    n_plot_samples: int = 8


# =============================================================================
# helpers
# =============================================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def _write_csv_hd(path: Path, hd: np.ndarray, col_prefix: str = "d") -> None:
    H, D = hd.shape
    header = ",".join(["h"] + [f"{col_prefix}{i}" for i in range(D)])
    lines = [header]
    for h in range(H):
        row = ",".join([str(h + 1)] + [f"{hd[h, d]:.8f}" for d in range(D)])
        lines.append(row)
    path.write_text("\n".join(lines), encoding="utf-8")


def _group_slices(dout: int) -> Dict[str, slice]:
    # 你当前 y = [acc(3), gyro(3), vel(3)] => 9
    if dout != 9:
        raise ValueError(f"grouping assumes dout=9, got {dout}")
    return {"acc": slice(0, 3), "gyro": slice(3, 6), "vel": slice(6, 9)}


def _aggregate_group_curve(metric_hd: np.ndarray, groups: Dict[str, slice]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k, sl in groups.items():
        out[k] = metric_hd[:, sl].mean(axis=1).tolist()
    return out


BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Mapping[str, torch.Tensor],
]


def _unpack_batch(batch: BatchType) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    兼容：
      - dict: {"X","Y","YM"(optional)}
      - tuple: (X,Y) or (X,Y,XM,YM)
    返回：X, Y, YM(optional)
    """
    if isinstance(batch, Mapping):
        X = batch["X"]
        Y = batch["Y"]
        YM = batch.get("YM", None)
        return X, Y, YM

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            X, Y = batch
            return X, Y, None
        if len(batch) == 4:
            X, Y, _XM, YM = batch
            return X, Y, YM

    raise TypeError(f"Unsupported batch type: {type(batch)}")


# =============================================================================
# masked metrics (accumulate across batches)
# =============================================================================

@torch.no_grad()
def _accum_masked_err_stats(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_samples: int,
) -> Dict[str, Any]:
    """
    计算 masked RMSE/MAE by horizon:
      对每个 (h,d)：
        mse = sum(mask*e^2) / sum(mask)
        mae = sum(mask*|e|) / sum(mask)

    若 YM 缺失，则 mask=1（全监督）。
    """
    model.eval()

    sum_e2: Optional[torch.Tensor] = None   # (H,D)
    sum_abs: Optional[torch.Tensor] = None  # (H,D)
    sum_m: Optional[torch.Tensor] = None    # (H,D)

    samp_yhat: List[np.ndarray] = []
    samp_ytrue: List[np.ndarray] = []
    samp_logvar: List[np.ndarray] = []
    samp_ym: List[np.ndarray] = []

    n_collected = 0

    for batch in loader:
        X, Y, YM = _unpack_batch(batch)
        X = X.to(device=device, dtype=torch.float32)
        Y = Y.to(device=device, dtype=torch.float32)

        if YM is None:
            m = torch.ones_like(Y, dtype=torch.float32, device=device)
        else:
            m = YM.to(device=device, dtype=torch.float32)
            # 兼容 (B,H) / (B,H,1)
            if m.ndim == 2:
                m = m.unsqueeze(-1).expand_as(Y)
            elif m.ndim == 3 and m.shape[-1] == 1:
                m = m.expand_as(Y)
            elif m.shape != Y.shape:
                raise ValueError(f"YM shape {tuple(m.shape)} must be broadcastable to Y {tuple(Y.shape)}")

        # forward: dY, logvar
        dY, logvar = model(X)
        y0 = extract_y0_from_x_last(X)
        y_hat = rollout_from_delta(y0, dY)

        # shape checks
        if y_hat.shape != Y.shape:
            raise ValueError(f"y_hat{tuple(y_hat.shape)} != Y{tuple(Y.shape)}")
        if logvar.shape != Y.shape:
            raise ValueError(f"logvar{tuple(logvar.shape)} != Y{tuple(Y.shape)}")

        # masked accumulate
        err = (y_hat - Y)
        e2 = err * err
        aerr = torch.abs(err)

        # reduce over batch dimension -> (H,D)
        se2 = torch.sum(m * e2, dim=0)
        sab = torch.sum(m * aerr, dim=0)
        sm = torch.sum(m, dim=0)

        if sum_e2 is None:
            sum_e2 = se2.detach().cpu()
            sum_abs = sab.detach().cpu()
            sum_m = sm.detach().cpu()
        else:
            sum_e2 += se2.detach().cpu()
            sum_abs += sab.detach().cpu()
            sum_m += sm.detach().cpu()

        # save samples
        if n_collected < save_samples:
            take = min(int(X.shape[0]), save_samples - n_collected)
            samp_yhat.append(y_hat[:take].detach().cpu().numpy())
            samp_ytrue.append(Y[:take].detach().cpu().numpy())
            samp_logvar.append(logvar[:take].detach().cpu().numpy())
            samp_ym.append(m[:take].detach().cpu().numpy())
            n_collected += take

    assert sum_e2 is not None and sum_abs is not None and sum_m is not None

    eps = 1e-12
    mse = (sum_e2.numpy() / (sum_m.numpy() + eps))
    rmse = np.sqrt(mse)
    mae = (sum_abs.numpy() / (sum_m.numpy() + eps))

    # global: 按 mask 加权平均（更符合异步监督语义）
    rmse_global = float(np.sum(sum_e2.numpy()) ** 0.5 / (np.sum(sum_m.numpy()) + eps) ** 0.5)
    mae_global = float(np.sum(sum_abs.numpy()) / (np.sum(sum_m.numpy()) + eps))

    samples = {
        "y_hat": np.concatenate(samp_yhat, axis=0) if samp_yhat else np.zeros((0,)),
        "y_true": np.concatenate(samp_ytrue, axis=0) if samp_ytrue else np.zeros((0,)),
        "logvar": np.concatenate(samp_logvar, axis=0) if samp_logvar else np.zeros((0,)),
        "y_mask": np.concatenate(samp_ym, axis=0) if samp_ym else np.zeros((0,)),
    }

    return {
        "rmse_hd": rmse,     # (H,D)
        "mae_hd": mae,       # (H,D)
        "rmse_global": rmse_global,
        "mae_global": mae_global,
        "sum_mask_hd": sum_m.numpy(),  # (H,D) 便于诊断：每个 horizon/dim 实际监督量
        "samples": samples,
    }


# =============================================================================
# main eval
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.eval.evaluate")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="configs/train/*.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.pth/.pt)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--save_samples", type=int, default=256)

    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--n_plot_samples", type=int, default=8)

    args = ap.parse_args()

    cfg = load_train_config(Path(args.yaml))

    # ---- override data/run ----
    data_cfg: DataConfig = cfg.data
    if args.data_dir is not None:
        data_cfg = DataConfig(**{**data_cfg.__dict__, "data_dir": Path(args.data_dir)})

    if args.batch_size is not None:
        data_cfg = DataConfig(**{**data_cfg.__dict__, "batch_size": int(args.batch_size)})

    dev = str(args.device) if args.device is not None else str(cfg.run.device)

    # ---- out_dir ----
    variant = getattr(cfg.run, "variant", "default")
    default_out = Path(cfg.run.out_dir) / str(variant) / f"eval_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out

    eval_cfg = EvalConfig(
        data_dir=Path(data_cfg.data_dir),
        ckpt=Path(args.ckpt),
        out_dir=out_dir,
        device=dev,
        batch_size=int(data_cfg.batch_size),
        split_name=str(args.split),
        save_samples=int(args.save_samples),
        make_plots=bool(args.plots),
        plot_fmt=str(args.plot_fmt),
        dt_s=float(args.dt),
        x_axis=str(args.x_axis),
        n_plot_samples=int(args.n_plot_samples),
    )

    _ensure_dir(eval_cfg.out_dir)

    # ---- device resolve ----
    dev_l = eval_cfg.device.lower()
    if dev_l.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        dev_l = "cpu"
    device = torch.device(dev_l)

    # ---- loaders (split exactly same as training) ----
    # 关键：build_loaders 的 split 逻辑与你训练一致，避免“评估切分不同步”
    train_loader, val_loader, test_loader = build_loaders(data_cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[eval_cfg.split_name]

    # ---- model ----
    model = S1Predictor(cfg.model).to(device)
    ckpt = _load_checkpoint(eval_cfg.ckpt, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ---- metrics (masked) ----
    res = _accum_masked_err_stats(
        model=model,
        loader=loader,
        device=device,
        save_samples=eval_cfg.save_samples,
    )

    rmse_hd = res["rmse_hd"]
    mae_hd = res["mae_hd"]
    rmse_global = float(res["rmse_global"])
    mae_global = float(res["mae_global"])
    sum_mask_hd = res["sum_mask_hd"]

    H, D = rmse_hd.shape
    groups = _group_slices(D)
    rmse_groups = _aggregate_group_curve(rmse_hd, groups)
    mae_groups = _aggregate_group_curve(mae_hd, groups)

    # ---- write outputs ----
    metrics = {
        "split": eval_cfg.split_name,
        "rmse_global": rmse_global,
        "mae_global": mae_global,
        "rmse_groups": rmse_groups,
        "mae_groups": mae_groups,
        "mask_sum_hd": sum_mask_hd.tolist(),  # 诊断：每个 (h,d) 实际监督量
        "cfg": {
            "data_dir": str(eval_cfg.data_dir),
            "ckpt": str(eval_cfg.ckpt),
            "device": str(device),
            "batch_size": int(eval_cfg.batch_size),
            "dt_s": float(eval_cfg.dt_s),
            "x_axis": str(eval_cfg.x_axis),
            "save_samples": int(eval_cfg.save_samples),
        },
        "model": {
            "din": int(cfg.model.din),
            "dout": int(cfg.model.dout),
            "pred_len": int(cfg.model.pred_len),
        },
    }

    with open(eval_cfg.out_dir / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)

    _write_csv_hd(eval_cfg.out_dir / "rmse_by_horizon.csv", rmse_hd, col_prefix="d")
    _write_csv_hd(eval_cfg.out_dir / "mae_by_horizon.csv", mae_hd, col_prefix="d")

    pred_npz = eval_cfg.out_dir / "pred_samples.npz"
    np.savez_compressed(
        pred_npz,
        y_hat=res["samples"]["y_hat"],
        y_true=res["samples"]["y_true"],
        logvar=res["samples"]["logvar"],
        y_mask=res["samples"]["y_mask"],
    )

    print(f"[EVAL] split={eval_cfg.split_name}  device={device}")
    print(f"[EVAL] RMSE(global)={rmse_global:.6f}  MAE(global)={mae_global:.6f}")
    print(f"[EVAL] wrote: {eval_cfg.out_dir / 'metrics.yaml'}")

    # ---- optional plots ----
    if eval_cfg.make_plots:
        plots_dir = eval_cfg.out_dir / "plots"
        _ensure_dir(plots_dir)

        fmt_list = [eval_cfg.plot_fmt]
        if eval_cfg.plot_fmt == "both":
            fmt_list = ["png", "pdf"]

        for fmt in fmt_list:
            for metric_name in ("rmse", "mae"):
                hp = HorizonPlotCfg(
                    dt_s=eval_cfg.dt_s,
                    use_seconds=(eval_cfg.x_axis == "sec"),
                    metric=metric_name,
                    out_name=f"{metric_name}_horizon_groups",
                    fmt=fmt,
                )
                plot_groups_vs_horizon(
                    eval_dirs=[eval_cfg.out_dir],
                    labels=[eval_cfg.split_name],
                    out_dir=plots_dir,
                    cfg=hp,
                )

            plot_rollout_samples_from_npz(
                pred_npz=pred_npz,
                out_dir=plots_dir,
                dt_s=eval_cfg.dt_s,
                n=eval_cfg.n_plot_samples,
                fmt=fmt,
            )

        print(f"[EVAL] plots saved under: {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
