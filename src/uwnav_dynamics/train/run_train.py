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

        # Re-clamp here in case you later change gaussian_nll_diag implementation.
        logvar = torch.clamp(logvar, min=logvar_clip_min, max=logvar_clip_max)
        return gaussian_nll_diag(y_hat, Y, logvar)

    return _loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", type=str, required=True, help="e.g. configs/train/pooltest02_s1_lstm_v0.yaml")
    args = ap.parse_args()

    cfg = load_train_config(Path(args.yaml))

    print(f"[RUN] name={cfg.run.name}")
    print(f"[RUN] out_dir={cfg.run.out_dir}")
    print(f"[RUN] data_dir={cfg.data.data_dir}")

    set_seed(cfg.run.seed)

    train_loader, val_loader, _test_loader = build_loaders(cfg.data)

    model = S1Predictor(cfg.model)

    loss_fn = build_loss_fn(cfg.loss.logvar_clip_min, cfg.loss.logvar_clip_max)

    fit(model, train_loader, val_loader, cfg.train, loss_fn)


if __name__ == "__main__":
    main()
