from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """
    训练超参（兼容旧版）

    说明：
      - 为了兼容你现在的 load_train_config，我们暂时保留 lr/weight_decay/device/amp/out_dir 在这里；
      - 但在“管线模式”下（run_train/cli 传参），fit() 会优先使用外部传入的 device/run_dir/amp。
    """
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    device: str = "cuda"
    amp: bool = True
    out_dir: Path = Path("out/ckpts/s1_baseline")

    # 未来可扩展：save_best/save_last/metric 等
    save_best: bool = True
    save_last: bool = True
    metric: str = "val_loss"  # 预留


def _to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    小优化：
      - non_blocking 只有在 CUDA + pinned memory 下才真正有意义；
      - CPU 时开 non_blocking 也不会错，但容易造成“看起来很玄学”的性能误判。
    """
    nb = (device.type == "cuda")
    X, Y = batch
    return X.to(device, non_blocking=nb), Y.to(device, non_blocking=nb)


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

    for X, Y in loader:
        X, Y = _to_device((X, Y), device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and amp_enabled and device.type == "cuda":
            # torch.cuda.amp.autocast is fine; for torch>=2.0 也可用 torch.autocast("cuda")
            with torch.cuda.amp.autocast():
                loss = loss_fn(model, X, Y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, X, Y)
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

    for X, Y in loader:
        X, Y = _to_device((X, Y), device)
        loss = loss_fn(model, X, Y)
        total += float(loss.item()) * X.shape[0]
        n += X.shape[0]

    return total / max(n, 1)


def _resolve_device(device: Optional[torch.device], cfg_device: str) -> torch.device:
    """
    统一 device 决策：
      - 优先使用外部传入 device（管线）
      - 否则使用 cfg.device
      - 若请求 cuda 但不可用，回落 cpu
    """
    if device is not None:
        return device

    dev = str(cfg_device).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_out_dir(run_dir: Optional[Path], cfg_out_dir: Path) -> Path:
    """
    训练输出目录优先级：
      run_dir（管线） > cfg.out_dir（旧逻辑）
    """
    return Path(run_dir) if run_dir is not None else Path(cfg_out_dir)


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    loss_fn,
    *,
    # ---- 管线可选注入：让 run_train/cli 生效 ----
    device: Optional[torch.device] = None,
    run_dir: Optional[Path] = None,
    amp: Optional[bool] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    # 未来扩展：scheduler / callbacks / loggers
) -> Dict[str, Any]:
    """
    返回训练摘要，方便 pipeline/报告消费：
      {"best_val":..., "best_path":..., "last_path":..., "device":...}
    """
    out_dir = _resolve_out_dir(run_dir, cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, cfg.device)

    # AMP 决策：外部优先，其次 cfg.amp；但只有 CUDA 才启用 scaler
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
    # 建议统一 .pth（和你 evaluate/cli 更一致）；你现在 .pt 也可以，但尽量统一
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

        # save last
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

        # save best
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
