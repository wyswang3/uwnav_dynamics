# src/uwnav_dynamics/eval/evaluate.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import yaml

from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig

# 复用你已经写好的绘图模块（horizon 曲线）
from uwnav_dynamics.viz.eval.plot_horizon_metrics import (
    plot_groups_vs_horizon,
    HorizonPlotCfg,
)

# rollout 样例图我们这里内置一个轻量函数（后续可重构到 viz 脚本里）
from uwnav_dynamics.viz.style.sci_style import setup_mpl


# =============================================================================
# Config
# =============================================================================

@dataclass
class EvalConfig:
    """
    评估配置（升级版）

    输出工件（供 viz 消费）：
      - metrics.yaml
      - rmse_by_horizon.csv
      - mae_by_horizon.csv
      - pred_samples.npz

    可选（由 evaluate.py 直接触发）：
      - plots/rmse_horizon_groups.(png/pdf)
      - plots/mae_horizon_groups.(png/pdf)
      - plots/rollout_sample_*.png/pdf
    """
    data_dir: Path
    ckpt: Path
    out_dir: Path

    device: str = "cpu"
    batch_size: int = 512

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_name: str = "test"  # "train" | "val" | "test"

    y0_source: str = "x_last_state"
    mode: str = "delta_cumsum"

    save_samples: int = 256

    # plotting
    make_plots: bool = False
    plot_fmt: str = "png"         # "png" | "pdf" | "both"
    dt_s: float = 0.01
    x_axis: str = "sec"           # "sec" | "step"
    n_plot_samples: int = 8       # 画多少个 rollout sample


# =============================================================================
# IO helpers
# =============================================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _np_load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _select_split_indices(n: int, train_ratio: float, val_ratio: float, split_name: str) -> np.ndarray:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))

    if split_name == "train":
        return np.arange(0, n_train, dtype=np.int64)
    if split_name == "val":
        return np.arange(n_train, n_train + n_val, dtype=np.int64)
    if split_name == "test":
        return np.arange(n_train + n_val, n, dtype=np.int64)
    raise ValueError(f"Unknown split_name={split_name!r}")


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


# =============================================================================
# Rollout
# =============================================================================

def rollout_delta_cumsum(
    *,
    x: torch.Tensor,
    dY: torch.Tensor,
    cfg_model: S1PredictorConfig,
    y0_source: str,
) -> torch.Tensor:
    if y0_source != "x_last_state":
        raise NotImplementedError(f"Only y0_source='x_last_state' supported, got {y0_source!r}")

    idx = torch.as_tensor(list(cfg_model.y_in_idx), device=x.device, dtype=torch.long)
    y0 = x[:, -1, :].index_select(dim=-1, index=idx)  # (B,9)
    y0 = y0.unsqueeze(1)  # (B,1,9)
    y_hat = y0 + torch.cumsum(dY, dim=1)  # (B,H,9)
    return y_hat


# =============================================================================
# Metrics
# =============================================================================

def _rmse_mae_by_horizon(y_hat: torch.Tensor, y_true: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    err = (y_hat - y_true)  # (N,H,D)
    mse = torch.mean(err * err, dim=0)        # (H,D)
    rmse = torch.sqrt(mse)                    # (H,D)
    mae = torch.mean(torch.abs(err), dim=0)   # (H,D)
    return rmse.detach().cpu().numpy(), mae.detach().cpu().numpy()


def _group_indices(dout: int) -> Dict[str, slice]:
    if dout != 9:
        raise ValueError("Current grouping assumes dout=9")
    return {"acc": slice(0, 3), "gyro": slice(3, 6), "vel": slice(6, 9)}


def _aggregate_group_curve(metric_hd: np.ndarray, groups: Dict[str, slice]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k, sl in groups.items():
        out[k] = metric_hd[:, sl].mean(axis=1).tolist()
    return out


def _write_csv_hd(path: Path, hd: np.ndarray, col_prefix: str = "d") -> None:
    H, D = hd.shape
    header = ",".join(["h"] + [f"{col_prefix}{i}" for i in range(D)])
    lines = [header]
    for h in range(H):
        row = ",".join([str(h + 1)] + [f"{hd[h, d]:.8f}" for d in range(D)])
        lines.append(row)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Plotting helpers (rollout samples)
# =============================================================================

def _norm3(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=-1))


def _plot_rollout_samples_from_npz(
    pred_npz: Path,
    out_dir: Path,
    *,
    dt_s: float,
    n: int,
    fmt: str,
) -> None:
    """
    画检查图：||Acc||, ||Gyro||, ||Vel|| 的 true vs pred

    业务逻辑：
      - 对 rollout 稳定性/发散/相位滞后很敏感；
      - 作为 baseline sanity check 很有用。
    """
    import matplotlib.pyplot as plt

    z = np.load(pred_npz)
    y_hat = z["y_hat"]   # (N,H,9)
    y_true = z["y_true"] # (N,H,9)

    setup_mpl()

    N, H, D = y_hat.shape
    nplot = min(int(n), int(N))
    t = np.arange(1, H + 1) * float(dt_s)

    for i in range(nplot):
        yh = y_hat[i]
        yt = y_true[i]

        acc_h = _norm3(yh[:, 0:3]); acc_t = _norm3(yt[:, 0:3])
        gyr_h = _norm3(yh[:, 3:6]); gyr_t = _norm3(yt[:, 3:6])
        vel_h = _norm3(yh[:, 6:9]); vel_t = _norm3(yt[:, 6:9])

        fig, ax = plt.subplots(1, 1)
        ax.plot(t, acc_t, label="||Acc|| true")
        ax.plot(t, acc_h, label="||Acc|| pred")
        ax.plot(t, gyr_t, label="||Gyro|| true")
        ax.plot(t, gyr_h, label="||Gyro|| pred")
        ax.plot(t, vel_t, label="||Vel|| true")
        ax.plot(t, vel_h, label="||Vel|| pred")

        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel("Magnitude")
        ax.grid(False)
        ax.legend(loc="best")

        stem = f"rollout_sample_{i:03d}"
        if fmt in ("png", "both"):
            fig.savefig(out_dir / f"{stem}.png")
        if fmt in ("pdf", "both"):
            fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)


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

    idx = _select_split_indices(n, cfg_eval.train_ratio, cfg_eval.val_ratio, cfg_eval.split_name)
    if idx.size == 0:
        raise RuntimeError(f"Split {cfg_eval.split_name} is empty. Check ratios.")

    # copy()：避免 torch.from_numpy 的只读 warning
    Xs = np.array(X[idx], copy=True)
    Ys = np.array(Y[idx], copy=True)

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
        y_hat = rollout_delta_cumsum(x=xb, dY=dY, cfg_model=cfg_model, y0_source=cfg_eval.y0_source)

        yhat_list.append(y_hat.cpu())
        ytrue_list.append(yb.cpu())
        logvar_list.append(logvar.cpu())

    y_hat_all = torch.cat(yhat_list, dim=0)
    y_true_all = torch.cat(ytrue_list, dim=0)
    logvar_all = torch.cat(logvar_list, dim=0)

    rmse_hd, mae_hd = _rmse_mae_by_horizon(y_hat_all, y_true_all)
    groups = _group_indices(cfg_model.dout)
    rmse_groups = _aggregate_group_curve(rmse_hd, groups)
    mae_groups = _aggregate_group_curve(mae_hd, groups)
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

    # 新增：一键画图
    ap.add_argument("--plots", action="store_true", help="generate plots under <out_dir>/plots")
    ap.add_argument("--plot_fmt", type=str, default="png", choices=["png", "pdf", "both"])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--x_axis", type=str, default="sec", choices=["sec", "step"])
    ap.add_argument("--n_plot_samples", type=int, default=8)

    args = ap.parse_args()

    cfg_y = _load_yaml(Path(args.yaml))

    data_dir = Path(cfg_y["data"]["data_dir"])
    split = cfg_y.get("data", {}).get("split", {})
    train_ratio = float(split.get("train_ratio", 0.70))
    val_ratio = float(split.get("val_ratio", 0.15))

    rollout = cfg_y.get("rollout", {})
    y0_source = str(rollout.get("y0_source", "x_last_state"))
    mode = str(rollout.get("mode", "delta_cumsum"))

    run = cfg_y.get("run", {})
    default_out_dir = Path(run.get("out_dir", "out/eval")) / "eval"
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir

    device = args.device if args.device is not None else str(run.get("device", "cpu"))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg_y["data"].get("batch_size", 512))

    cfg_eval = EvalConfig(
        data_dir=data_dir,
        ckpt=Path(args.ckpt),
        out_dir=out_dir,
        device=device,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_name=args.split,
        y0_source=y0_source,
        mode=mode,
        save_samples=int(args.save_samples),
        make_plots=bool(args.plots),
        plot_fmt=str(args.plot_fmt),
        dt_s=float(args.dt),
        x_axis=str(args.x_axis),
        n_plot_samples=int(args.n_plot_samples),
    )

    m = cfg_y["model"]
    cfg_model = S1PredictorConfig(
        din=int(m["din"]),
        dout=int(m["dout"]),
        pred_len=int(m["pred_len"]),
        rnn_hidden=int(m["rnn_hidden"]),
        rnn_layers=int(m["rnn_layers"]),
        dropout=float(m.get("dropout", 0.0)),
        u_in_idx=tuple(m.get("u_in_idx", list(range(0, 8)))),
        y_in_idx=tuple(m.get("y_in_idx", list(range(8, 17)))),
        use_thruster_as_replacement=bool(m.get("use_thruster_as_replacement", True)),
        use_hydro_feat=bool(m.get("use_hydro_feat", True)),
        # blocks 的详细配置你可以后面再补；B0 不影响评估
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
        "cfg": {
            "data_dir": str(cfg_eval.data_dir),
            "ckpt": str(cfg_eval.ckpt),
            "device": cfg_eval.device,
            "batch_size": cfg_eval.batch_size,
            "train_ratio": cfg_eval.train_ratio,
            "val_ratio": cfg_eval.val_ratio,
            "y0_source": cfg_eval.y0_source,
            "mode": cfg_eval.mode,
            "dt_s": cfg_eval.dt_s,
            "x_axis": cfg_eval.x_axis,
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

    # ---- optional plots ----
    if cfg_eval.make_plots:
        plots_dir = cfg_eval.out_dir / "plots"
        _ensure_dir(plots_dir)

        # 1) horizon 曲线：RMSE + MAE（复用 plot_horizon_metrics.py）
        for metric_name in ("rmse", "mae"):
            hp = HorizonPlotCfg(
                dt_s=cfg_eval.dt_s,
                use_seconds=(cfg_eval.x_axis == "sec"),
                metric=metric_name,
                out_name=f"{metric_name}_horizon_groups",
                fmt=cfg_eval.plot_fmt,
            )
            plot_groups_vs_horizon(
                eval_dirs=[cfg_eval.out_dir],
                labels=[Path(cfg_eval.out_dir).name],
                out_dir=plots_dir,
                cfg=hp,
            )

        # 2) rollout 样例（内置轻量画法，后续可 refactor 到 plot_rollout_samples.py）
        _plot_rollout_samples_from_npz(
            pred_npz,
            plots_dir,
            dt_s=cfg_eval.dt_s,
            n=cfg_eval.n_plot_samples,
            fmt=cfg_eval.plot_fmt,
        )

        print(f"[EVAL] plots saved under: {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
