"""
模块名称：P0 复合损失测试

模块职责：
验证 P0.1 中新增的 DVL 辅助监督损失，
以最小方式接入训练主路径时的接线与退化行为。

主要功能：
1. 验证 `build_loss_fn()` 优先使用 `forward_with_aux()`。
2. 验证 `dvl_obs_weight=0.0` 时退化为原主 state loss。
3. 验证 DVL auxiliary loss 只读 velocity semantic group。
4. 验证全 False velocity mask 时，辅助 DVL 稀疏监督返回 0.0 是合法情况。

数据流：
dummy model + X / Y / target_mask
    ↓
build_loss_fn()
    ↓
state loss + optional dvl auxiliary loss
    ↓
退化行为与隔离性断言

依赖模块：
- torch
- uwnav_dynamics.train.run_train
- uwnav_dynamics.models.losses.auxiliary
- uwnav_dynamics.models.losses.nll
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 本测试只覆盖 Patch C 的 loss 接线，不涉及 trainer / eval / viz。
"""

from __future__ import annotations

import torch

from uwnav_dynamics.models.losses.auxiliary import masked_huber_loss
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag_masked
from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout
from uwnav_dynamics.train.run_train import build_loss_fn


class _DummyDvlAuxCfg:
    def __init__(self, enabled: bool):
        self.enabled = enabled


class _DummyAuxHeadsCfg:
    def __init__(self, enabled: bool):
        self.dvl_obs = _DummyDvlAuxCfg(enabled)


class _DummyCfg:
    def __init__(self, enabled: bool):
        self.aux_heads = _DummyAuxHeadsCfg(enabled)


class _DummyModel:
    def __init__(
        self,
        *,
        forward_dY: torch.Tensor,
        forward_logvar: torch.Tensor,
        aux_dY: torch.Tensor,
        aux_logvar: torch.Tensor,
        dvl_obs: torch.Tensor | None,
        dvl_enabled: bool,
    ):
        self._forward_dY = forward_dY
        self._forward_logvar = forward_logvar
        self._aux_dY = aux_dY
        self._aux_logvar = aux_logvar
        self._dvl_obs = dvl_obs
        self.cfg = _DummyCfg(dvl_enabled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_dY.clone(), self._forward_logvar.clone()

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        dvl_obs = None if self._dvl_obs is None else self._dvl_obs.clone()
        return self._aux_dY.clone(), self._aux_logvar.clone(), {"dvl_obs": dvl_obs}


def _make_inputs(*, batch: int = 2, hist: int = 5, horizon: int = 4, din: int = 25, dout: int = 9):
    x = torch.zeros(batch, hist, din)
    y = torch.zeros(batch, horizon, dout)
    target_mask = torch.ones(batch, horizon, dout, dtype=torch.bool)
    y_in_idx = tuple(range(8, 17))
    return x, y, target_mask, y_in_idx


def _state_loss_only(model: _DummyModel, x: torch.Tensor, y: torch.Tensor, target_mask: torch.Tensor, y_in_idx) -> torch.Tensor:
    dY, logvar, _aux = model.forward_with_aux(x)
    y0 = extract_y0_from_x_last(x, y_in_idx)
    y_hat = rollout_from_delta(y0, dY)
    return gaussian_nll_diag_masked(y_hat, y, logvar, target_mask)


def test_composite_loss_uses_forward_with_aux_and_matches_state_loss_when_dvl_weight_zero():
    x, y, target_mask, y_in_idx = _make_inputs()
    aux_dY = torch.zeros(2, 4, 9)
    aux_logvar = torch.zeros(2, 4, 9)
    forward_dY = torch.full((2, 4, 9), 3.0)
    forward_logvar = torch.zeros(2, 4, 9)
    y[:, :, 0] = 1.0

    model = _DummyModel(
        forward_dY=forward_dY,
        forward_logvar=forward_logvar,
        aux_dY=aux_dY,
        aux_logvar=aux_logvar,
        dvl_obs=torch.zeros(2, 4, 3),
        dvl_enabled=True,
    )

    loss_fn = build_loss_fn(
        -10.0,
        6.0,
        y_in_idx,
        dout=9,
        dvl_obs_weight=0.0,
        dvl_obs_delta=1.0,
    )
    got = loss_fn(model, x, y, target_mask)
    expected = _state_loss_only(model, x, y, target_mask, y_in_idx)

    assert torch.allclose(got, expected)


def test_composite_loss_non_velocity_dimensions_do_not_change_dvl_aux_component():
    x, y, target_mask, y_in_idx = _make_inputs()
    vel_indices = canonical_semantic_output_layout(9).group_indices["vel"]
    dvl_obs = torch.zeros(2, 4, len(vel_indices))
    model = _DummyModel(
        forward_dY=torch.zeros(2, 4, 9),
        forward_logvar=torch.zeros(2, 4, 9),
        aux_dY=torch.zeros(2, 4, 9),
        aux_logvar=torch.zeros(2, 4, 9),
        dvl_obs=dvl_obs,
        dvl_enabled=True,
    )
    loss_fn = build_loss_fn(
        -10.0,
        6.0,
        y_in_idx,
        dout=9,
        dvl_obs_weight=0.2,
        dvl_obs_delta=1.0,
    )

    y_changed = y.clone()
    y_changed[:, :, 0] = 9.0
    y_changed[:, :, 4] = -3.0

    base_total = loss_fn(model, x, y, target_mask)
    base_state = _state_loss_only(model, x, y, target_mask, y_in_idx)
    changed_total = loss_fn(model, x, y_changed, target_mask)
    changed_state = _state_loss_only(model, x, y_changed, target_mask, y_in_idx)

    assert torch.allclose(base_total - base_state, changed_total - changed_state)


def test_composite_loss_masked_out_velocity_positions_do_not_change_dvl_aux_component():
    x, y, target_mask, y_in_idx = _make_inputs()
    vel_indices = canonical_semantic_output_layout(9).group_indices["vel"]
    dvl_obs = torch.zeros(2, 4, len(vel_indices))
    model = _DummyModel(
        forward_dY=torch.zeros(2, 4, 9),
        forward_logvar=torch.zeros(2, 4, 9),
        aux_dY=torch.zeros(2, 4, 9),
        aux_logvar=torch.zeros(2, 4, 9),
        dvl_obs=dvl_obs,
        dvl_enabled=True,
    )
    loss_fn = build_loss_fn(
        -10.0,
        6.0,
        y_in_idx,
        dout=9,
        dvl_obs_weight=0.2,
        dvl_obs_delta=1.0,
    )

    target_mask_masked = target_mask.clone()
    target_mask_masked[:, :, vel_indices[0]] = False
    target_mask_masked[:, :, vel_indices[1]] = False
    target_mask_masked[:, :, vel_indices[2]] = False

    y_changed = y.clone()
    y_changed[:, :, vel_indices[0]] = 7.0
    y_changed[:, :, vel_indices[1]] = -2.0
    y_changed[:, :, vel_indices[2]] = 1.5

    base_total = loss_fn(model, x, y, target_mask_masked)
    base_state = _state_loss_only(model, x, y, target_mask_masked, y_in_idx)
    changed_total = loss_fn(model, x, y_changed, target_mask_masked)
    changed_state = _state_loss_only(model, x, y_changed, target_mask_masked, y_in_idx)

    assert torch.allclose(base_total - base_state, changed_total - changed_state)


def test_masked_huber_loss_all_false_velocity_mask_returns_zero_for_legal_sparse_dvl_case():
    # 这里显式说明：对辅助 DVL 稀疏监督，prediction horizon 内完全没有有效观测是合法情况，
    # 因此全 False mask 返回 0.0，而不是像主 state masked NLL 那样 fail-fast。
    y_hat = torch.randn(2, 4, 3)
    y_true = torch.randn(2, 4, 3)
    target_mask = torch.zeros(2, 4, 3, dtype=torch.bool)

    loss = masked_huber_loss(y_hat, y_true, target_mask, delta=1.0)

    assert loss.item() == 0.0


def test_composite_loss_raises_when_dvl_head_enabled_but_aux_output_missing():
    x, y, target_mask, y_in_idx = _make_inputs()
    model = _DummyModel(
        forward_dY=torch.zeros(2, 4, 9),
        forward_logvar=torch.zeros(2, 4, 9),
        aux_dY=torch.zeros(2, 4, 9),
        aux_logvar=torch.zeros(2, 4, 9),
        dvl_obs=None,
        dvl_enabled=True,
    )
    loss_fn = build_loss_fn(
        -10.0,
        6.0,
        y_in_idx,
        dout=9,
        dvl_obs_weight=0.2,
        dvl_obs_delta=1.0,
    )

    try:
        loss_fn(model, x, y, target_mask)
    except ValueError as exc:
        assert "aux['dvl_obs'] is None" in str(exc)
    else:
        raise AssertionError("expected ValueError when dvl auxiliary path is enabled but aux['dvl_obs'] is None")
