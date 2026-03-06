"""
模块名称：训练主入口

模块职责：
负责加载训练配置、准备 run-scoped 数据 artifact、
构建模型与损失函数，并调用 trainer 执行训练。

主要功能：
1. 复用 canonical train parser 与 runtime override 生成最终训练配置。
2. 准备 split / scaler / DataLoader，并写出 `resolved_train.yaml`。
3. 基于 execution layout contract 构建 rollout loss，保证 train / eval 对 `y0` 解释一致。
4. 在 PR5 中支持 batch `target_mask` 驱动的 mask-aware supervision。

数据流：
train yaml + CLI override
    ↓
TrainYamlConfig / resolved_train.yaml
    ↓
prepare_train_data()
    ↓
S1Predictor + rollout loss
    ↓
fit()
    ↓
best.pth / last.pth

依赖模块：
- uwnav_dynamics.train.config
- uwnav_dynamics.train.data_pipeline
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout

备注：
- 本模块只接入 execution layout contract，不负责语义分组解释。
- `target_mask` 若存在，则是训练运行时唯一的监督有效性真源。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataclasses import replace
import torch

from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.dataset.split import DEFAULT_SPLIT_STRATEGY
from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data_pipeline import prepare_train_data
from uwnav_dynamics.train.runtime import (
    TrainCliOverrides,
    apply_train_overrides,
    reconcile_data_config_for_device,
    resolve_runtime_device,
    save_resolved_train_config,
    set_global_seed,
)
from uwnav_dynamics.train.trainer import fit
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag, gaussian_nll_diag_masked


def build_loss_fn(logvar_clip_min: float, logvar_clip_max: float, y_in_idx):
    """
    v0 loss:
      model(X) -> dY, logvar
      y0 = X_last_state
      y_hat = y0 + cumsum(dY)
      loss = diag NLL(y_hat, Y, logvar)
    """
    def _loss(model, X, Y, target_mask=None):
        dY, logvar = model(X)
        y0 = extract_y0_from_x_last(X, y_in_idx)
        y_hat = rollout_from_delta(y0, dY)
        logvar = torch.clamp(logvar, min=logvar_clip_min, max=logvar_clip_max)
        if target_mask is not None:
            return gaussian_nll_diag_masked(y_hat, Y, logvar, target_mask)
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

    if args.amp and args.no_amp:
        raise ValueError("Cannot set both --amp and --no_amp")

    amp_override = None
    if args.amp:
        amp_override = True
    if args.no_amp:
        amp_override = False

    # 配置处理顺序固定为：
    #   1) 原始 train yaml（canonical parser）
    #   2) CLI override（纯函数替换，不改原对象）
    #   3) runtime reconcile（例如 CPU 下关闭 pin_memory）
    #   4) resolved_train.yaml（记录最终实际执行配置）
    cli_overrides = TrainCliOverrides(
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory.lower() == "true") if args.pin_memory is not None else None,
        out_dir=args.out_dir,
        variant=args.variant,
        seed=args.seed,
        amp=amp_override,
    )

    cfg = apply_train_overrides(
        cfg,
        cli_overrides,
    )

    # ------------------------------
    # Resolve run_dir = out_dir/variant
    # ------------------------------
    run_layout = RunLayout(out_dir=Path(cfg.run.out_dir), variant=str(getattr(cfg.run, "variant", "default")))
    variant = run_layout.variant
    run_dir = run_layout.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] name={cfg.run.name}")
    print(f"[RUN] variant={variant}")
    print(f"[RUN] run_dir={run_dir}")
    print(f"[RUN] data_dir={cfg.data.data_dir}")

    # ------------------------------
    # Seed / device
    # ------------------------------
    set_global_seed(cfg.run.seed)
    device = resolve_runtime_device(cfg.run.device)
    data_cfg, pin_memory_note = reconcile_data_config_for_device(cfg.data, device)
    if pin_memory_note is not None:
        print(pin_memory_note)
        cfg = replace(cfg, data=data_cfg)
    print(f"[RUN] resolved_config={run_dir / 'resolved_train.yaml'}")
    save_resolved_train_config(
        run_dir / "resolved_train.yaml",
        cfg,
        source_yaml=args.yaml,
        cli_overrides=cli_overrides,
        requested_device=str(cfg.run.device),
        runtime_device=str(device),
        run_dir=run_dir,
        split_indices_path=run_layout.split_indices_path,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        x_scaler_path=run_layout.x_scaler_path,
        y_scaler_path=run_layout.y_scaler_path,
    )

    # ------------------------------
    # Data / split / scaler (no leakage)
    # ------------------------------
    prepared = prepare_train_data(cfg.data, run_layout)

    model = S1Predictor(cfg.model)

    loss_fn = build_loss_fn(cfg.loss.logvar_clip_min, cfg.loss.logvar_clip_max, cfg.model.y_in_idx)

    # ------------------------------
    # Fit (pipeline-style): pass device/run_dir/amp
    # ------------------------------
    fit(
        model,
        prepared.train_loader,
        prepared.val_loader,
        cfg.train,
        loss_fn,
        device=device,
        run_dir=run_dir,
        amp=bool(cfg.run.amp),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
