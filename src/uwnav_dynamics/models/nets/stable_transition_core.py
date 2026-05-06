"""
模块名称：稳定状态转移内核

模块职责：
定义不依赖旧 optional blocks 的 `StableTransitionCore`，
将物理耗散、有界控制输入和有界神经残差写入状态转移方程本身。

主要功能：
1. 从历史输入序列中编码控制与状态上下文。
2. 对 `gyro / vel` 使用指数耗散或半隐式耗散更新。
3. 用有界控制项和有界残差项生成下一步状态。
4. 保持 `(dY, logvar)` 输出契约，兼容现有 train / eval / replay。

数据流：
X(B,L,Din)
    ↓
slice u_seq / y_seq
    ↓
LSTM context encoder
    ↓
bounded control + dissipative transition + bounded residual
    ↓
y_next
    ↓
dY = y_next - y_current, logvar

系统级数据流：
IMU / DVL / PWM / Power
    ↓
alignment / KF-ESKF proxy state
    ↓
dataset build
    ↓
train split scaler
    ↓
StableTransitionCore
    ↓
transition-model validation
    ↓
controller / simulator loop

依赖模块：
- torch
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 本模块第一版在 z-score 空间中执行稳定转移；物理耗散约束体现为归一化状态空间的结构性收缩。
- 不 import `uwnav_dynamics.models.blocks.*`，避免延续旧模块组合路线。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig
from uwnav_dynamics.models.utils.execution_layout import validate_execution_layout, validate_feature_indices
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout


_SUPPORTED_CORE_TYPES = {
    "stable_diag_damp",
    "implicit_euler",
    "control_affine",
    "residual_budget",
    "energy_budget",
}


class StableTransitionCore(nn.Module):
    """
    带耗散、有界控制和有界残差的状态转移模型。

    输出契约与 `S1Predictor` 保持一致：
      - `dY: (B,H,Dout)`
      - `logvar: (B,H,Dout)`
    """

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = self._resolve_cfg(cfg)
        cfg = self.cfg
        if str(cfg.core_type) not in _SUPPORTED_CORE_TYPES:
            raise ValueError(
                f"Unsupported stable core_type={cfg.core_type!r}; "
                f"supported={sorted(_SUPPORTED_CORE_TYPES)}"
            )
        if int(cfg.dout) != 9:
            raise ValueError(f"StableTransitionCore v1 expects dout=9, got {cfg.dout}")
        if int(cfg.core_hidden) <= 0:
            raise ValueError(f"model.core_hidden must be > 0, got {cfg.core_hidden}")
        if float(cfg.dt) <= 0.0:
            raise ValueError(f"model.dt must be > 0, got {cfg.dt}")
        if float(cfg.damping_min) < 0.0 or float(cfg.damping_max) <= float(cfg.damping_min):
            raise ValueError(
                "model.damping_min/max must satisfy 0 <= min < max, "
                f"got {cfg.damping_min}/{cfg.damping_max}"
            )
        if float(cfg.residual_bound) <= 0.0 or float(cfg.control_bound) <= 0.0:
            raise ValueError(
                "model.residual_bound and model.control_bound must be > 0, "
                f"got {cfg.residual_bound}/{cfg.control_bound}"
            )

        validate_feature_indices(cfg.u_in_idx, upper_bound=cfg.din, name="model.u_in_idx")
        validate_execution_layout(cfg.y_in_idx, din=cfg.din, dout=cfg.dout)
        semantic_layout = canonical_semantic_output_layout(cfg.dout)
        self._acc_idx = tuple(int(i) for i in semantic_layout.group_indices["acc"])
        self._gyro_idx = tuple(int(i) for i in semantic_layout.group_indices["gyro"])
        self._vel_idx = tuple(int(i) for i in semantic_layout.group_indices["vel"])

        self.register_buffer("_u_idx", torch.as_tensor(list(cfg.u_in_idx), dtype=torch.long), persistent=False)
        self.register_buffer("_y_idx", torch.as_tensor(list(cfg.y_in_idx), dtype=torch.long), persistent=False)
        self.u_in_dim = int(len(cfg.u_in_idx))
        self.y_in_dim = int(len(cfg.y_in_idx))
        self.z_dim = 6

        self.enc = nn.LSTM(
            input_size=int(cfg.din),
            hidden_size=int(cfg.rnn_hidden),
            num_layers=int(cfg.rnn_layers),
            batch_first=True,
            dropout=float(cfg.dropout) if int(cfg.rnn_layers) > 1 else 0.0,
            bidirectional=False,
        )
        core_in = int(cfg.rnn_hidden) + self.y_in_dim + self.u_in_dim
        self.context = nn.Sequential(
            nn.Linear(core_in, int(cfg.core_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cfg.core_hidden), int(cfg.core_hidden)),
            nn.ReLU(inplace=True),
        )
        step_in = int(cfg.core_hidden) + self.y_in_dim + self.u_in_dim
        self.control_head = nn.Linear(step_in, self.z_dim)
        self.damping_head = nn.Linear(step_in, self.z_dim)
        self.residual_head = nn.Linear(step_in, int(cfg.dout))
        self.acc_head = nn.Linear(step_in + self.z_dim, 3)
        self.logvar_head = nn.Linear(int(cfg.core_hidden), int(cfg.pred_len) * int(cfg.dout))
        self.affine_head = nn.Linear(step_in, self.z_dim * self.u_in_dim)

    @staticmethod
    def _resolve_cfg(cfg: S1PredictorConfig | Any) -> S1PredictorConfig:
        if isinstance(cfg, S1PredictorConfig):
            return cfg
        if hasattr(cfg, "model") and isinstance(getattr(cfg, "model"), S1PredictorConfig):
            return getattr(cfg, "model")
        raise TypeError("StableTransitionCore expects S1PredictorConfig or TrainYamlConfig-like object")

    def _slice_u(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=-1, index=self._u_idx)

    def _slice_y(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=-1, index=self._y_idx)

    def _z_from_y(self, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([y[:, self._gyro_idx], y[:, self._vel_idx]], dim=-1)

    def _assemble_y(self, acc_next: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        y_next = torch.empty(
            (acc_next.shape[0], int(self.cfg.dout)),
            device=acc_next.device,
            dtype=acc_next.dtype,
        )
        y_next[:, self._acc_idx] = acc_next
        y_next[:, self._gyro_idx] = z_next[:, :3]
        y_next[:, self._vel_idx] = z_next[:, 3:]
        return y_next

    def _step_context(self, base_context: torch.Tensor, y_curr: torch.Tensor, u_last: torch.Tensor) -> torch.Tensor:
        return torch.cat([base_context, y_curr, u_last], dim=-1)

    def _bounded_residual(self, step_feat: torch.Tensor) -> torch.Tensor:
        bound = float(self.cfg.residual_bound)
        if str(self.cfg.core_type) == "residual_budget":
            bound *= 0.5
        if str(self.cfg.core_type) == "energy_budget":
            bound *= 0.75
        return bound * torch.tanh(self.residual_head(step_feat))

    def _bounded_control(self, step_feat: torch.Tensor, u_last: torch.Tensor) -> torch.Tensor:
        if str(self.cfg.core_type) == "control_affine":
            B = torch.tanh(self.affine_head(step_feat)).view(step_feat.shape[0], self.z_dim, self.u_in_dim)
            u_eff = torch.tanh(u_last).unsqueeze(-1)
            return float(self.cfg.control_bound) * torch.bmm(B, u_eff).squeeze(-1)
        return float(self.cfg.control_bound) * torch.tanh(self.control_head(step_feat))

    def _positive_damping(self, step_feat: torch.Tensor) -> torch.Tensor:
        lo = float(self.cfg.damping_min)
        hi = float(self.cfg.damping_max)
        damping = lo + (hi - lo) * torch.sigmoid(self.damping_head(step_feat))
        if str(self.cfg.core_type) == "energy_budget":
            damping = torch.clamp(damping * 1.5, max=hi)
        return damping

    def _one_step(
        self,
        *,
        base_context: torch.Tensor,
        y_curr: torch.Tensor,
        u_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_feat = self._step_context(base_context, y_curr, u_last)
        z_curr = self._z_from_y(y_curr)
        damping = self._positive_damping(step_feat)
        control = self._bounded_control(step_feat, u_last)
        residual = self._bounded_residual(step_feat)
        residual_z = torch.cat([residual[:, self._gyro_idx], residual[:, self._vel_idx]], dim=-1)
        dt = float(self.cfg.dt)

        if str(self.cfg.core_type) == "implicit_euler":
            z_next = (z_curr + dt * (control + residual_z)) / (1.0 + dt * damping)
        else:
            decay = torch.exp(-dt * damping)
            z_next = decay * z_curr + dt * (control + residual_z)

        acc_raw = self.acc_head(torch.cat([step_feat, z_next], dim=-1))
        acc_next = float(self.cfg.control_bound) * torch.tanh(acc_raw) + residual[:, self._acc_idx]
        y_next = self._assemble_y(acc_next, z_next)
        return y_next, y_next - y_curr

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        if x.ndim != 3 or x.shape[-1] != int(self.cfg.din):
            raise ValueError(f"x must be (B,L,{self.cfg.din}), got {tuple(x.shape)}")

        u_seq = self._slice_u(x)
        y_seq = self._slice_y(x)
        _enc_out, (h_n, _c_n) = self.enc(x)
        h_last = h_n[-1]
        y_curr = y_seq[:, -1, :]
        u_last = u_seq[:, -1, :]
        base_context = self.context(torch.cat([h_last, y_curr, u_last], dim=-1))

        dy_steps = []
        for _ in range(int(self.cfg.pred_len)):
            y_curr, dY_step = self._one_step(
                base_context=base_context,
                y_curr=y_curr,
                u_last=u_last,
            )
            dy_steps.append(dY_step.unsqueeze(1))
        dY = torch.cat(dy_steps, dim=1)
        logvar = self.logvar_head(base_context).view(x.shape[0], int(self.cfg.pred_len), int(self.cfg.dout))
        return dY, logvar, {"dvl_obs": None}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dY, logvar, _aux = self.forward_with_aux(x)
        return dY, logvar


def supported_stable_core_types() -> tuple[str, ...]:
    """返回当前实现支持的稳定内核类型。"""
    return tuple(sorted(_SUPPORTED_CORE_TYPES))
