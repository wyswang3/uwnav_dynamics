"""
模块名称：masked NLL 测试

模块职责：
验证 PR5 新增的 `gaussian_nll_diag_masked`
满足 dense 等价、忽略 masked-out 位置和 fail-fast 三个核心契约。

主要功能：
1. mask 全 True 时与 dense NLL 行为一致。
2. mask=False 的位置改变目标值，不应影响 masked loss。
3. 有效监督元素数为 0 时显式报错。

数据流：
y_hat / y_true / logvar / target_mask
    ↓
gaussian_nll_diag / gaussian_nll_diag_masked
    ↓
loss contract assertions

依赖模块：
- pytest
- torch
- uwnav_dynamics.models.losses.nll
"""

from __future__ import annotations

import pytest
import torch

from uwnav_dynamics.models.losses.nll import gaussian_nll_diag, gaussian_nll_diag_masked


def test_masked_nll_matches_dense_when_mask_all_true():
    y_hat = torch.zeros(2, 3, 4, dtype=torch.float32)
    y_true = torch.ones_like(y_hat)
    logvar = torch.zeros_like(y_hat)
    target_mask = torch.ones_like(y_hat, dtype=torch.bool)

    dense = gaussian_nll_diag(y_hat, y_true, logvar)
    masked = gaussian_nll_diag_masked(y_hat, y_true, logvar, target_mask)

    assert torch.allclose(masked, dense)


def test_masked_nll_ignores_masked_out_positions():
    y_hat = torch.zeros(1, 2, 3, dtype=torch.float32)
    y_true = torch.zeros_like(y_hat)
    logvar = torch.zeros_like(y_hat)
    target_mask = torch.ones_like(y_hat, dtype=torch.bool)
    target_mask[:, 1, 2] = False

    baseline = gaussian_nll_diag_masked(y_hat, y_true, logvar, target_mask)
    y_true[:, 1, 2] = 123.0
    changed = gaussian_nll_diag_masked(y_hat, y_true, logvar, target_mask)

    assert torch.allclose(changed, baseline)


def test_masked_nll_fails_fast_when_no_valid_supervision():
    y_hat = torch.zeros(1, 2, 3, dtype=torch.float32)
    y_true = torch.zeros_like(y_hat)
    logvar = torch.zeros_like(y_hat)
    target_mask = torch.zeros_like(y_hat, dtype=torch.bool)

    with pytest.raises(ValueError, match="zero valid supervision"):
        gaussian_nll_diag_masked(y_hat, y_true, logvar, target_mask)
