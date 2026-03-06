"""
模块名称：训练执行器

模块职责：
负责执行 epoch 级训练与验证循环，
并将 DataLoader batch、loss_fn、optimizer、AMP 与 checkpoint 保存串起来。

主要功能：
1. 支持 `(X, Y)` 与 `(X, Y, target_mask)` 两类 batch。
2. 在训练与验证阶段统一调用外部注入的 `loss_fn`。
3. 管理设备迁移、AMP、梯度裁剪与 best/last checkpoint 落盘。

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
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    device: str = "cuda"
    amp: bool = True
    out_dir: Path = Path("out/ckpts/s1_baseline")

    save_best: bool = True
    save_last: bool = True
    metric: str = "val_loss"


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
    if device is not None:
        return device

    dev = str(cfg_device).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_out_dir(run_dir: Optional[Path], cfg_out_dir: Path) -> Path:
    return Path(run_dir) if run_dir is not None else Path(cfg_out_dir)


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
    返回训练摘要，方便 pipeline/报告消费：
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

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if amp_enabled and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device=device, scaler=scaler,
            grad_clip=cfg.grad_clip,
            amp_enabled=amp_enabled,
        )
        va = eval_one_epoch(model, val_loader, loss_fn, device)

        print(f"[EPOCH {ep:03d}] train_loss={tr:.6f}  val_loss={va:.6f}")

        if getattr(cfg, "save_last", True):
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "val_loss": va,
                    "train_loss": tr,
                    "device": str(device),
                    "amp": amp_enabled,
                },
                last_path,
            )

        if getattr(cfg, "save_best", True) and va < best_val:
            best_val = va
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "val_loss": va,
                    "train_loss": tr,
                    "device": str(device),
                    "amp": amp_enabled,
                },
                best_path,
            )
            print(f"[CKPT] best -> {best_path} (val={best_val:.6f})")

    print(f"[DONE] best_val={best_val:.6f}  ckpt={best_path}")

    return {
        "best_val": float(best_val),
        "best_path": str(best_path),
        "last_path": str(last_path),
        "device": str(device),
        "amp": bool(amp_enabled),
        "out_dir": str(out_dir),
    }
