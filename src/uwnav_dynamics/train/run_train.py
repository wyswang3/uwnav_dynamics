from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data import build_loaders  # updated: returns dict batches
from uwnav_dynamics.train.trainer import fit
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.rollout import extract_y0_from_x_last, rollout_from_delta
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag, gaussian_nll_diag_masked
from uwnav_dynamics.models.losses.metrics import masked_regression_metrics


# =============================================================================
# Seed
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Loss builder (async supervision aware)
# =============================================================================
def build_loss_fn(
    logvar_clip_min: float,
    logvar_clip_max: float,
    *,
    min_valid_count: int = 1,
    print_mask_stats: bool = False,
):
    """
    Async-supervision loss (returns (loss, aux)):

      model(X) -> dY, logvar
      y0 = extract_y0_from_x_last(X)
      y_hat = rollout_from_delta(y0, dY)
      loss = masked diag NLL(y_hat, Y, logvar, YM)

    aux (for trainer logging / ckpt):
      - rmse, mae (masked)
      - valid_ratio (from YM)
      - nll (same as loss.item(), for readability)
      - skipped (if supervision empty)
    """

    def _loss(model: torch.nn.Module, batch: Any):
        # -------- parse batch --------
        if isinstance(batch, (tuple, list)):
            X = batch[0]
            Y = batch[1]
            YM = batch[2] if len(batch) >= 3 else None
        elif isinstance(batch, dict):
            X = batch["X"]
            Y = batch["Y"]
            YM = batch.get("YM", None)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # -------- forward / rollout --------
        dY, logvar = model(X)
        y0 = extract_y0_from_x_last(X)
        y_hat = rollout_from_delta(y0, dY)

        logvar = torch.clamp(logvar, min=logvar_clip_min, max=logvar_clip_max)

        # -------- unmasked fallback --------
        if YM is None:
            loss = gaussian_nll_diag(y_hat, Y, logvar)
            # 指标：此时等价于全监督
            met = masked_regression_metrics(y_hat, Y, torch.ones_like(Y, dtype=torch.uint8, device=Y.device))
            aux = {
                "nll": float(loss.detach().item()),
                "rmse": float(met["rmse"]),
                "mae": float(met["mae"]),
                "valid_ratio": 1.0,
                "skipped": 0.0,
            }
            return loss, aux

        # -------- masked NLL + stats --------
        loss, stats = gaussian_nll_diag_masked(
            y_hat,
            Y,
            logvar,
            YM,
            min_count=min_valid_count,
            return_stats=True,
        )

        # stats: valid_ratio/valid_count/skipped...
        valid_ratio = float(stats.get("valid_ratio", 0.0))
        skipped = bool(stats.get("skipped", False))

        # -------- masked regression metrics (more intuitive) --------
        # 如果 batch 被 skip（监督为空），这里的指标没有意义；我们统一输出 NaN 或 0
        if skipped:
            rmse = float("nan")
            mae = float("nan")
        else:
            met = masked_regression_metrics(y_hat, Y, YM)
            rmse = float(met["rmse"])
            mae = float(met["mae"])

        aux = {
            "nll": float(loss.detach().item()),
            "rmse": rmse,
            "mae": mae,
            "valid_ratio": valid_ratio,
            "valid_count": float(stats.get("valid_count", 0.0)),
            "skipped": 1.0 if skipped else 0.0,
        }

        # 轻量 debug：只在发生 skipped 时提示，避免刷屏
        if print_mask_stats and skipped:
            print(f"[LOSS][MASK] skipped empty supervision batch: {aux}")

        return loss, aux

    return _loss

# =============================================================================
# CLI helpers
# =============================================================================

def _str2bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid bool string: {s!r} (use true/false)")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.train.run_train")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="configs/train/*.yaml")

    # ---- CLI override: keep your pipeline stable ----
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)  # cpu/cuda/cuda:0
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--pin_memory", type=str, default=None, choices=["true", "false"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    # async-supervision / debug toggles (optional but useful)
    ap.add_argument("--min_valid_count", type=int, default=None)
    ap.add_argument("--print_mask_stats", action="store_true")

    args = ap.parse_args()

    cfg = load_train_config(Path(args.yaml))

    # ------------------------------
    # Apply overrides (frozen dataclasses -> replace)
    # ------------------------------
    if args.amp and args.no_amp:
        raise ValueError("Cannot set both --amp and --no_amp")

    run = cfg.run
    data = cfg.data
    train = cfg.train

    # ---- data overrides ----
    if args.data_dir is not None:
        data = replace(data, data_dir=Path(args.data_dir))

    if args.batch_size is not None:
        data = replace(data, batch_size=int(args.batch_size))

    if args.num_workers is not None:
        data = replace(data, num_workers=int(args.num_workers))

    if args.pin_memory is not None:
        data = replace(data, pin_memory=_str2bool(args.pin_memory))

    # ---- run/train overrides ----
    if args.device is not None:
        dev = str(args.device)
        run = replace(run, device=dev)
        train = replace(train, device=dev)

    if args.epochs is not None:
        train = replace(train, epochs=int(args.epochs))

    if args.out_dir is not None:
        od = Path(args.out_dir)
        run = replace(run, out_dir=od)
        train = replace(train, out_dir=od)

    if args.variant is not None:
        run = replace(run, variant=str(args.variant))

    if args.seed is not None:
        sd = int(args.seed)
        run = replace(run, seed=sd)
        data = replace(data, seed=sd)

    if args.amp:
        run = replace(run, amp=True)
        train = replace(train, amp=True)
    if args.no_amp:
        run = replace(run, amp=False)
        train = replace(train, amp=False)

    # ------------------------------
    # Resolve run_dir = out_dir/variant
    # trainer expects cfg.train.out_dir = final directory
    # ------------------------------
    variant = getattr(run, "variant", "default")
    run_dir = Path(run.out_dir) / str(variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = replace(train, out_dir=run_dir)

    # finalize cfg
    cfg = replace(cfg, run=run, data=data, train=train)

    print(f"[RUN] name={cfg.run.name}")
    print(f"[RUN] variant={variant}")
    print(f"[RUN] run_dir={run_dir}")
    print(f"[RUN] data_dir={cfg.data.data_dir}")
    print(f"[RUN] device={cfg.run.device} amp={cfg.run.amp}")
    print(f"[RUN] batch_size={cfg.data.batch_size} num_workers={cfg.data.num_workers} pin_memory={cfg.data.pin_memory}")

    # ------------------------------
    # Seed / device sanity
    # ------------------------------
    set_seed(cfg.run.seed)

    dev_str = str(cfg.run.device).lower()
    if dev_str.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        dev_str = "cpu"

    # CPU pin_memory off
    if dev_str == "cpu" and cfg.data.pin_memory:
        print("[INFO] CPU training: pin_memory=True is useless; auto set to False.")
        cfg = replace(cfg, data=replace(cfg.data, pin_memory=False))

    # ------------------------------
    # Data / model / loss
    # ------------------------------
    train_loader, val_loader, _test_loader = build_loaders(cfg.data)

    model = S1Predictor(cfg.model)

    min_valid_count = int(args.min_valid_count) if args.min_valid_count is not None else 1
    loss_fn = build_loss_fn(
        cfg.loss.logvar_clip_min,
        cfg.loss.logvar_clip_max,
        min_valid_count=min_valid_count,
        print_mask_stats=bool(args.print_mask_stats),
    )

    # ------------------------------
    # Fit
    # trainer.fit reads cfg.train.device/cfg.train.amp/cfg.train.out_dir
    # ------------------------------
    fit(model, train_loader, val_loader, cfg.train, loss_fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
