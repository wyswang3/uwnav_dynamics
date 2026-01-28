from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data import build_loaders
from uwnav_dynamics.train.trainer import fit
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.rollout import extract_y0_from_x_last, rollout_from_delta
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss_fn(logvar_clip_min: float, logvar_clip_max: float):
    """
    v0 loss:
      model(X) -> dY, logvar
      y0 = X_last_state
      y_hat = y0 + cumsum(dY)
      loss = diag NLL(y_hat, Y, logvar)
    """
    def _loss(model, X, Y):
        dY, logvar = model(X)
        y0 = extract_y0_from_x_last(X)
        y_hat = rollout_from_delta(y0, dY)
        logvar = torch.clamp(logvar, min=logvar_clip_min, max=logvar_clip_max)
        return gaussian_nll_diag(y_hat, Y, logvar)
    return _loss


def main() -> int:
    ap = argparse.ArgumentParser("uwnav_dynamics.train.run_train")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="configs/train/*.yaml")

    # ---- CLI override：让 pipeline 生效 ----
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)          # cpu/cuda/cuda:0
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--pin_memory", type=str, default=None, choices=["true", "false"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    args = ap.parse_args()

    cfg = load_train_config(Path(args.yaml))

    # ------------------------------
    # Apply overrides
    # ------------------------------
    if args.data_dir is not None:
        cfg.data.data_dir = Path(args.data_dir)
    if args.device is not None:
        cfg.run.device = str(args.device)
        cfg.train.device = str(args.device)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.data.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
    if args.pin_memory is not None:
        cfg.data.pin_memory = (args.pin_memory.lower() == "true")
    if args.out_dir is not None:
        cfg.run.out_dir = Path(args.out_dir)
        cfg.train.out_dir = Path(args.out_dir)
    if args.variant is not None:
        cfg.run.variant = str(args.variant)
    if args.seed is not None:
        cfg.run.seed = int(args.seed)
        # DataConfig 的 seed 来自 run.seed，通常也同步一下更稳
        cfg.data.seed = int(args.seed)

    if args.amp and args.no_amp:
        raise ValueError("Cannot set both --amp and --no_amp")
    if args.amp:
        cfg.run.amp = True
        cfg.train.amp = True
    if args.no_amp:
        cfg.run.amp = False
        cfg.train.amp = False

    # ------------------------------
    # Resolve run_dir = out_dir/variant
    # ------------------------------
    variant = getattr(cfg.run, "variant", "default")
    run_dir = Path(cfg.run.out_dir) / str(variant)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] name={cfg.run.name}")
    print(f"[RUN] variant={variant}")
    print(f"[RUN] run_dir={run_dir}")
    print(f"[RUN] data_dir={cfg.data.data_dir}")

    # ------------------------------
    # Seed / device
    # ------------------------------
    set_seed(cfg.run.seed)

    dev_str = str(cfg.run.device).lower()
    if dev_str.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        dev_str = "cpu"
    device = torch.device(dev_str)

    # CPU pin_memory 关闭，避免 warning
    if device.type == "cpu" and cfg.data.pin_memory:
        print("[INFO] CPU training: pin_memory=True is useless; auto set to False.")
        cfg.data.pin_memory = False

    # ------------------------------
    # Data / model / loss
    # ------------------------------
    train_loader, val_loader, _test_loader = build_loaders(cfg.data)

    model = S1Predictor(cfg.model)

    loss_fn = build_loss_fn(cfg.loss.logvar_clip_min, cfg.loss.logvar_clip_max)

    # ------------------------------
    # Fit (pipeline-style): pass device/run_dir/amp
    # ------------------------------
    fit(
        model,
        train_loader,
        val_loader,
        cfg.train,
        loss_fn,
        device=device,
        run_dir=run_dir,
        amp=bool(cfg.run.amp),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
