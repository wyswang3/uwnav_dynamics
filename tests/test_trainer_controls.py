"""
模块名称：训练控制逻辑测试

模块职责：
验证 trainer 中新增的 scheduler 与 early stopping
能够在最小可控场景下稳定触发并返回训练摘要。

主要功能：
1. 构造恒定验证损失场景。
2. 验证 `ReduceLROnPlateau` 会降低学习率。
3. 验证 early stopping 会提前结束训练并记录 history。
4. 验证自定义 monitor 可以驱动 best ckpt 选择与 early stopping。

数据流：
dummy model + synthetic dataloader
    ↓
trainer.fit()
    ↓
fit summary assertions
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from uwnav_dynamics.train.trainer import TrainConfig, fit


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        d_y = self.dummy.expand(batch, 2, 3)
        logvar = torch.zeros_like(d_y)
        return d_y, logvar


def _constant_loss(model, x, y, target_mask=None):
    del x, y, target_mask
    return model.dummy * 0.0 + torch.tensor(1.0, dtype=torch.float32)


def _constant_extra_metrics(model, x, y, target_mask=None):
    del model, x, y, target_mask
    return {
        "val_rmse_global_zspace": 0.5,
        "val_mae_global_zspace": 0.25,
    }


def test_fit_supports_scheduler_and_early_stopping(tmp_path: Path) -> None:
    x = torch.zeros((8, 4, 5), dtype=torch.float32)
    y = torch.zeros((8, 2, 3), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    cfg = TrainConfig(
        epochs=10,
        eval_every=1,
        lr=1.0e-3,
        weight_decay=0.0,
        grad_clip=0.0,
        device="cpu",
        amp=False,
        out_dir=Path(tmp_path),
        scheduler_name="reduce_on_plateau",
        scheduler_factor=0.5,
        scheduler_patience=0,
        scheduler_min_lr=1.0e-5,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )

    res = fit(
        _DummyModel(),
        loader,
        loader,
        cfg,
        _constant_loss,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        amp=False,
        extra_val_metrics_fn=_constant_extra_metrics,
    )

    assert res["stopped_early"] is True
    assert res["epochs_ran"] == 3
    assert res["best_epoch"] == 1
    assert res["final_lr"] < 1.0e-3
    assert res["best_val_rmse_global_zspace"] == 0.5
    assert res["best_val_mae_global_zspace"] == 0.25
    assert len(res["history"]) == 3
    assert res["history"][0]["val_rmse_global_zspace"] == 0.5
    assert res["history"][0]["val_mae_global_zspace"] == 0.25
    assert (tmp_path / "best.pth").exists()
    assert (tmp_path / "last.pth").exists()


def test_fit_can_select_best_epoch_by_custom_monitor(tmp_path: Path) -> None:
    x = torch.zeros((4, 4, 5), dtype=torch.float32)
    y = torch.zeros((4, 2, 3), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    cfg = TrainConfig(
        epochs=10,
        eval_every=1,
        lr=1.0e-3,
        weight_decay=0.0,
        grad_clip=0.0,
        device="cpu",
        amp=False,
        out_dir=Path(tmp_path),
        metric="val_transition_score",
        scheduler_name="reduce_on_plateau",
        scheduler_factor=0.5,
        scheduler_patience=0,
        scheduler_min_lr=1.0e-5,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
    )

    monitor_values = iter([3.0, 2.0, 2.5])

    def _monitor(model, x, y, target_mask=None):
        del model, x, y, target_mask
        return torch.tensor(next(monitor_values), dtype=torch.float32)

    res = fit(
        _DummyModel(),
        loader,
        loader,
        cfg,
        _constant_loss,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        amp=False,
        monitor_fn=_monitor,
        monitor_name="val_transition_score",
        extra_val_metrics_fn=_constant_extra_metrics,
    )

    assert res["stopped_early"] is True
    assert res["epochs_ran"] == 3
    assert res["best_epoch"] == 2
    assert res["monitor_name"] == "val_transition_score"
    assert res["best_monitor"] == 2.0
    assert res["selected_val_loss"] == 1.0
    assert res["best_val_loss"] == 1.0
    assert res["selected_val_rmse_global_zspace"] == 0.5
    assert res["selected_val_mae_global_zspace"] == 0.25
    assert res["final_lr"] < 1.0e-3
    assert res["history"][1]["is_best"] is True
