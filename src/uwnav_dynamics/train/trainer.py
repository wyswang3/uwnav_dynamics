"""
模块名称：训练执行器

模块职责：
负责执行 epoch 级训练与验证循环，
并将 DataLoader batch、loss_fn、optimizer、AMP、scheduler、
early stopping 与 checkpoint 保存串起来。

主要功能：
1. 支持 `(X, Y)` 与 `(X, Y, target_mask)` 两类 batch。
2. 在训练与验证阶段统一调用外部注入的 `loss_fn`。
3. 管理设备迁移、AMP、梯度裁剪、scheduler 与 best/last checkpoint 落盘。
4. 记录 epoch 级 train/val/lr 历史，供上层落盘 summary 与可视化。

数据流：
prepare_train_data() 产出的 DataLoader
    ↓
trainer._to_device()
    ↓
loss_fn(model, X, Y, optional target_mask)
    ↓
optimizer / checkpoint

依赖模块：
- torch
- torch.utils.data

备注：
- 本模块不构造 supervision mask，只负责按 batch 传递。
- `target_mask` 若存在，则是训练运行时唯一的 mask 执行真源。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    """
    训练超参（兼容旧版）

    说明：
      - 为了兼容你现在的 load_train_config，我们暂时保留 lr/weight_decay/device/amp/out_dir 在这里；
      - 但在“管线模式”下（run_train/cli 传参），fit() 会优先使用外部传入的 device/run_dir/amp。
      - 这些字段仍属于“纯配置值”，因此适合冻结；runtime override 一律走 replace()。
    """
    epochs: int = 30
    eval_every: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    device: str = "cuda"
    amp: bool = True
    out_dir: Path = Path("out/ckpts/s1_baseline")

    save_best: bool = True
    save_last: bool = True
    metric: str = "val_loss"
    scheduler_name: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


def _to_device(
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """
    小优化：
      - non_blocking 只有在 CUDA + pinned memory 下才真正有意义；
      - CPU 时开 non_blocking 也不会错，但容易造成“看起来很玄学”的性能误判。
    """
    if len(batch) not in (2, 3):
        raise ValueError(f"batch must be (X,Y) or (X,Y,target_mask), got len={len(batch)}")
    nb = (device.type == "cuda")
    return tuple(t.to(device, non_blocking=nb) for t in batch)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    amp_enabled: bool,
) -> float:
    """执行一个训练 epoch，并返回按样本数加权的平均 loss。"""
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        moved = _to_device(batch, device)
        X, Y = moved[0], moved[1]
        target_mask = moved[2] if len(moved) == 3 else None
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and amp_enabled and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss = loss_fn(model, X, Y, target_mask)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, X, Y, target_mask)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += float(loss.item()) * X.shape[0]
        n += X.shape[0]

    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader: DataLoader, loss_fn, device: torch.device) -> float:
    """执行一个验证 epoch，并返回按样本数加权的平均 loss。"""
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        moved = _to_device(batch, device)
        X, Y = moved[0], moved[1]
        target_mask = moved[2] if len(moved) == 3 else None
        loss = loss_fn(model, X, Y, target_mask)
        total += float(loss.item()) * X.shape[0]
        n += X.shape[0]

    return total / max(n, 1)


def _resolve_device(device: Optional[torch.device], cfg_device: str) -> torch.device:
    """统一解析运行设备；优先使用显式传入的 runtime device。"""
    if device is not None:
        return device

    dev = str(cfg_device).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_out_dir(run_dir: Optional[Path], cfg_out_dir: Path) -> Path:
    """统一解析训练输出目录；运行时 `run_dir` 优先级高于配置值。"""
    return Path(run_dir) if run_dir is not None else Path(cfg_out_dir)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """读取当前 optimizer 主学习率。"""
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """按配置构造训练期学习率调度器。"""
    scheduler_name = str(getattr(cfg, "scheduler_name", "none")).lower()
    if scheduler_name in {"", "none"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(cfg, "scheduler_factor", 0.5)),
            patience=int(getattr(cfg, "scheduler_patience", 5)),
            min_lr=float(getattr(cfg, "scheduler_min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler_name={scheduler_name!r}")


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    loss_fn,
    *,
    device: Optional[torch.device] = None,
    run_dir: Optional[Path] = None,
    amp: Optional[bool] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    执行完整训练循环，并返回供 pipeline/报告消费的训练摘要。

    返回值示例：
      {"best_val":..., "best_path":..., "last_path":..., "device":...}
    """
    out_dir = _resolve_out_dir(run_dir, cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, cfg.device)

    amp_enabled = bool(cfg.amp) if amp is None else bool(amp)
    if device.type != "cuda":
        amp_enabled = False

    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if amp_enabled and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_epoch = 0
    stopped_early = False
    epochs_without_improve = 0
    eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
    early_patience = max(int(getattr(cfg, "early_stopping_patience", 0)), 0)
    early_min_delta = max(float(getattr(cfg, "early_stopping_min_delta", 0.0)), 0.0)
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"
    history: list[dict[str, Any]] = []
    start_ts = time.perf_counter()

    for ep in range(1, cfg.epochs + 1):
        # train / val 分开调用，保证日志与 best-checkpoint 选择都基于独立验证集。
        tr = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device=device, scaler=scaler,
            grad_clip=cfg.grad_clip,
            amp_enabled=amp_enabled,
        )
        should_eval = (ep % eval_every == 0) or (ep == cfg.epochs)
        va = float("nan")
        improved = False
        if should_eval:
            va = eval_one_epoch(model, val_loader, loss_fn, device)
            if scheduler is not None:
                scheduler.step(va)

            if va < (best_val - early_min_delta):
                improved = True
                best_val = va
                best_epoch = ep
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

        lr_now = _current_lr(optimizer)
        val_disp = f"{va:.6f}" if should_eval else "SKIP"
        print(f"[EPOCH {ep:03d}] train_loss={tr:.6f}  val_loss={val_disp}  lr={lr_now:.6e}")

        if getattr(cfg, "save_last", True):
            # last checkpoint 记录“最近训练状态”，用于排查中断或继续人工分析。
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "val_loss": va,
                    "train_loss": tr,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "device": str(device),
                    "amp": amp_enabled,
                    "lr": lr_now,
                },
                last_path,
            )

        if getattr(cfg, "save_best", True) and improved:
            # best checkpoint 只按验证损失更新，不受 train loss 或其他指标影响。
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "val_loss": va,
                    "train_loss": tr,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "device": str(device),
                    "amp": amp_enabled,
                    "lr": lr_now,
                },
                best_path,
            )
            print(f"[CKPT] best -> {best_path} (val={best_val:.6f})")

        history.append(
            {
                "epoch": int(ep),
                "train_loss": float(tr),
                "val_loss": float(va),
                "lr": float(lr_now),
                "evaluated": bool(should_eval),
                "is_best": bool(improved),
            }
        )

        if should_eval and early_patience > 0 and epochs_without_improve >= early_patience:
            stopped_early = True
            print(
                "[EARLY_STOP] "
                f"epoch={ep} best_epoch={best_epoch} patience={early_patience} "
                f"min_delta={early_min_delta:.3e}"
            )
            break

    train_wall_time_sec = float(time.perf_counter() - start_ts)
    print(f"[DONE] best_val={best_val:.6f}  best_epoch={best_epoch}  ckpt={best_path}")

    return {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "best_path": str(best_path),
        "last_path": str(last_path),
        "device": str(device),
        "amp": bool(amp_enabled),
        "out_dir": str(out_dir),
        "epochs_ran": int(len(history)),
        "stopped_early": bool(stopped_early),
        "final_lr": float(_current_lr(optimizer)),
        "scheduler_name": str(getattr(cfg, "scheduler_name", "none")),
        "eval_every": int(eval_every),
        "train_wall_time_sec": train_wall_time_sec,
        "history": history,
    }
