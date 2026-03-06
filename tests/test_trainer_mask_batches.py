"""
模块名称：trainer mask batch 测试

模块职责：
验证训练执行器在 PR5 后同时兼容 dense batch 与 mask-aware batch，
并且会把 batch 中的 `target_mask` 继续传递给 `loss_fn`。

主要功能：
1. 验证 `(X, Y)` batch 不回归。
2. 验证 `(X, Y, target_mask)` batch 能进入 train / eval 主循环。
3. 验证 `loss_fn` 实际收到的 mask 与 DataLoader 中一致。

数据流：
TensorDataset
    ↓
trainer.train_one_epoch / eval_one_epoch
    ↓
loss_fn(model, X, Y, target_mask)
    ↓
mask forwarding assertions

依赖模块：
- torch
- torch.utils.data
- uwnav_dynamics.train.trainer
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from uwnav_dynamics.train.trainer import eval_one_epoch, train_one_epoch


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))


def _zero_grad_loss(model: torch.nn.Module) -> torch.Tensor:
    return model.scale * 0.0


def test_trainer_accepts_dense_and_mask_batches():
    model = _DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    X = torch.randn(4, 3, 5)
    Y = torch.randn(4, 2, 9)
    target_mask = torch.ones(4, 2, 9, dtype=torch.bool)
    target_mask[0, 1, 6:] = False

    dense_seen: list[torch.Tensor | None] = []
    masked_seen: list[torch.Tensor | None] = []

    def dense_loss_fn(model_obj, xb, yb, target_mask_batch=None):
        dense_seen.append(target_mask_batch)
        return _zero_grad_loss(model_obj)

    def masked_loss_fn(model_obj, xb, yb, target_mask_batch=None):
        masked_seen.append(None if target_mask_batch is None else target_mask_batch.detach().cpu().clone())
        return _zero_grad_loss(model_obj)

    dense_loader = DataLoader(TensorDataset(X, Y), batch_size=2, shuffle=False)
    masked_loader = DataLoader(TensorDataset(X, Y, target_mask), batch_size=2, shuffle=False)

    train_one_epoch(
        model,
        dense_loader,
        optimizer,
        dense_loss_fn,
        device=device,
        scaler=None,
        grad_clip=0.0,
        amp_enabled=False,
    )
    eval_one_epoch(model, dense_loader, dense_loss_fn, device)
    assert all(mask is None for mask in dense_seen)

    train_one_epoch(
        model,
        masked_loader,
        optimizer,
        masked_loss_fn,
        device=device,
        scaler=None,
        grad_clip=0.0,
        amp_enabled=False,
    )
    eval_one_epoch(model, masked_loader, masked_loss_fn, device)

    assert len(masked_seen) > 0
    assert all(mask is not None for mask in masked_seen)
    assert torch.equal(masked_seen[0], target_mask[:2])
