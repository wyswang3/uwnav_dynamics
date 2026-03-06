"""
模块名称：execution layout contract 测试

模块职责：
验证 PR4 中的 execution layout contract
已经成为 train / eval rollout 路径的唯一执行真源。

主要功能：
1. 验证 `extract_y0_from_x_last()` 只按 `cfg_model.y_in_idx` 提取状态。
2. 验证非法 execution layout 会尽早失败。
3. 验证训练 loss path 与共享 rollout helper 对同一 `y_in_idx` 的解释一致。

数据流：
synthetic X / y_in_idx
    ↓
execution_layout helper
    ↓
run_train.build_loss_fn
    ↓
rollout parity assertions

依赖模块：
- torch
- pytest
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout
- uwnav_dynamics.train.run_train
"""

from __future__ import annotations

import pytest
import torch

from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last, validate_execution_layout
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.train import run_train


def test_extract_y0_from_x_last_uses_cfg_model_y_in_idx_instead_of_hardcoded():
    x = torch.arange(2 * 3 * 20, dtype=torch.float32).view(2, 3, 20)
    y_in_idx = (1, 5, 9, 2, 6, 10, 3, 7, 11)

    y0 = extract_y0_from_x_last(x, y_in_idx)
    expected = x[:, -1, :][:, list(y_in_idx)]

    assert torch.equal(y0, expected)
    assert not torch.equal(y0, x[:, -1, 8:17])


def test_validate_execution_layout_rejects_duplicate_or_wrong_length_indices():
    with pytest.raises(ValueError, match="length mismatch"):
        validate_execution_layout((1, 2, 3), din=20, dout=9)

    with pytest.raises(ValueError, match="duplicate indices"):
        validate_execution_layout((1, 2, 3, 4, 5, 6, 7, 7, 8), din=20, dout=9)

    with pytest.raises(ValueError, match="out of range"):
        validate_execution_layout((1, 2, 3, 4, 5, 6, 7, 8, 20), din=20, dout=9)


def test_build_loss_fn_uses_same_execution_contract_as_rollout_helper(monkeypatch):
    y_in_idx = (1, 5, 9, 2, 6, 10, 3, 7, 11)
    x = torch.arange(2 * 4 * 20, dtype=torch.float32).view(2, 4, 20)
    y_true = torch.zeros(2, 3, 9, dtype=torch.float32)
    dY = torch.full((2, 3, 9), 0.25, dtype=torch.float32)
    logvar = torch.zeros_like(dY)
    captured: dict[str, torch.Tensor] = {}

    class DummyModel:
        def __call__(self, xb):
            assert torch.equal(xb, x)
            return dY, logvar

    def fake_nll(y_hat, y_ref, lv):
        captured["y_hat"] = y_hat.detach().clone()
        captured["y_ref"] = y_ref.detach().clone()
        captured["logvar"] = lv.detach().clone()
        return ((y_hat - y_ref) ** 2).mean()

    monkeypatch.setattr(run_train, "gaussian_nll_diag", fake_nll)

    loss_fn = run_train.build_loss_fn(-10.0, 6.0, y_in_idx)
    loss = loss_fn(DummyModel(), x, y_true)

    expected = rollout_from_delta(extract_y0_from_x_last(x, y_in_idx), dY)
    assert torch.allclose(captured["y_hat"], expected)
    assert torch.allclose(captured["y_ref"], y_true)
    assert torch.allclose(captured["logvar"], logvar)
    assert float(loss.item()) >= 0.0
