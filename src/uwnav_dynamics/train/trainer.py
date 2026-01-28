from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = True
    out_dir: Path = Path("out/ckpts/s1_baseline")


def _to_device(batch, device: torch.device):
    return [t.to(device, non_blocking=True) for t in batch]


def train_one_epoch(model, loader: DataLoader, optimizer, loss_fn, device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler], grad_clip: float) -> float:
    model.train()
    total = 0.0
    n = 0

    for X, Y in loader:
        X, Y = _to_device((X, Y), device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = loss_fn(model, X, Y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, X, Y)
            loss.backward()
            if grad_clip > 0:
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


def fit(model, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, loss_fn):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = None
    if cfg.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_path = cfg.out_dir / "best.pt"
    last_path = cfg.out_dir / "last.pt"

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, cfg.grad_clip)
        va = eval_one_epoch(model, val_loader, loss_fn, device)

        print(f"[EPOCH {ep:03d}] train_loss={tr:.6f}  val_loss={va:.6f}")

        # save last
        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "val_loss": va,
            },
            last_path,
        )

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "val_loss": va,
                },
                best_path,
            )
            print(f"[CKPT] best -> {best_path} (val={best_val:.6f})")

    print(f"[DONE] best_val={best_val:.6f}  ckpt={best_path}")
