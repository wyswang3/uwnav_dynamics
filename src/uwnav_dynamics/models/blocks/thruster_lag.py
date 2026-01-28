from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ThrusterLagConfig:
    """
    推进器微动力单元（输入侧先验）

    业务逻辑：
      将原始 PWM/duty 输入 u_seq (B,L,8) 转换为“等效推力控制量” u_eff_seq (B,L,8)，
      显式包含推进器常见非理想特性，避免 RNN 在隐状态里“歪学”：

      1) PWM 映射（工程习惯）
         - 若输入是 duty in [5,10]，映射到 [-1,1]，中心 7.5 对应 0 thrust。
      2) deadzone（死区）
         - 小幅指令不产生推力：用可微 soft-shrink 近似。
      3) saturation（饱和）
         - 推力上限：用 tanh 近似平滑饱和。
      4) 一阶滞后（等效惯性）
         - u_eff[k] = a*u_eff[k-1] + (1-a)*u_nl[k]
         - a = exp(-dt/tau)，tau 可学习且为正，保证递推稳定。

    备注：
      - enabled=False 时退化为恒等映射：u_eff_seq = u_seq（或映射后的 u_norm_seq 取决于 normalize_input）。
    """
    enabled: bool = False

    # PWM 输入范围（若你的输入已标准化到 [-1,1]，可以把 normalize_input=False）
    normalize_input: bool = True
    pwm_center: float = 7.5
    pwm_half_range: float = 2.5  # (10-5)/2

    # deadzone：softshrink 的阈值（单位是 [-1,1] 归一化后）
    deadzone: float = 0.05

    # saturation：tanh 饱和的斜率（越大越接近 hard clip）
    sat_gain: float = 2.0

    # lag 时间常数：可学习 tau（秒）。用 softplus(param)+tau_min 保证正。
    learn_tau: bool = True
    tau_init: float = 0.08
    tau_min: float = 0.01

    # 是否为每个推进器独立 tau
    per_channel_tau: bool = True

    # 采样周期（s）
    dt: float = 0.01


class ThrusterLag(nn.Module):
    def __init__(self, cfg: ThrusterLagConfig, *, n_thrusters: int = 8):
        super().__init__()
        self.cfg = cfg
        self.n_thrusters = n_thrusters

        if cfg.learn_tau:
            # 用“可训练参数 -> softplus -> tau”确保 tau>0
            # 允许 per_channel 或共享一个 tau
            if cfg.per_channel_tau:
                init = torch.full((n_thrusters,), float(cfg.tau_init))
            else:
                init = torch.tensor([float(cfg.tau_init)])

            # 反解 softplus^-1：p = log(exp(tau-tau_min)-1)
            tau0 = torch.clamp(init - float(cfg.tau_min), min=1e-6)
            p0 = torch.log(torch.expm1(tau0))
            self._tau_param = nn.Parameter(p0)
        else:
            self.register_buffer("_tau_param", torch.empty(0), persistent=False)

    def _pwm_to_unit(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        将 duty in [5,10] 线性映射到 [-1,1]：
          u_unit = (u - center) / half_range
        """
        return (u_seq - self.cfg.pwm_center) / self.cfg.pwm_half_range

    def _deadzone(self, u_unit: torch.Tensor) -> torch.Tensor:
        """
        deadzone 的可微近似：
          softshrink(u, lambd) = sign(u)*max(|u|-lambd, 0)
        """
        if self.cfg.deadzone <= 0:
            return u_unit
        return F.softshrink(u_unit, lambd=float(self.cfg.deadzone))

    def _saturate(self, u_unit: torch.Tensor) -> torch.Tensor:
        """
        平滑饱和：tanh(gain*u)
        gain 越大越接近 hard clip。
        """
        g = float(self.cfg.sat_gain)
        if g <= 0:
            return torch.clamp(u_unit, -1.0, 1.0)
        return torch.tanh(g * u_unit)

    def _tau(self) -> torch.Tensor:
        """
        返回 tau（秒），形状：
          - per_channel: (8,)
          - shared:      (1,)
        """
        if not self.cfg.learn_tau:
            # 固定 tau
            t = torch.tensor([float(self.cfg.tau_init)], device=self._device())
            if self.cfg.per_channel_tau:
                return t.repeat(self.n_thrusters)
            return t
        # softplus 确保正
        tau = F.softplus(self._tau_param) + float(self.cfg.tau_min)
        if not self.cfg.per_channel_tau:
            # (1,) broadcast
            return tau
        return tau  # (8,)

    def _device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")

    @torch.no_grad()
    def extra_repr(self) -> str:
        return f"enabled={self.cfg.enabled}, normalize={self.cfg.normalize_input}, learn_tau={self.cfg.learn_tau}"

    def forward(
        self,
        u_seq: torch.Tensor,
        *,
        u0_eff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          u_seq:   (B, L, 8) 原始 PWM/duty（或已归一化）
          u0_eff: (B, 8) 初始滤波状态（可选）。不提供时默认用 u_nl[:,0] 初始化。

        Returns:
          u_eff_seq: (B, L, 8) 等效控制序列
        """
        if not self.cfg.enabled:
            # 不启用：保持 baseline 行为（你可以选择仍做 normalize，这里默认不做任何处理）
            return u_seq

        if u_seq.ndim != 3 or u_seq.shape[-1] != self.n_thrusters:
            raise ValueError(f"u_seq must be (B,L,{self.n_thrusters}), got {tuple(u_seq.shape)}")

        # 1) normalize to [-1,1]（若已标准化可关）
        u_unit = self._pwm_to_unit(u_seq) if self.cfg.normalize_input else u_seq

        # 2) deadzone + 3) saturation
        u_nl = self._saturate(self._deadzone(u_unit))

        # 4) one-pole lag filter（稳定递推）
        B, L, C = u_nl.shape
        tau = self._tau().to(u_nl.device)  # (C,) or (1,)
        # a = exp(-dt/tau) in (0,1)
        a = torch.exp(-float(self.cfg.dt) / tau).view(1, 1, -1)  # broadcast to (1,1,C)

        # 初始化
        if u0_eff is None:
            u_prev = u_nl[:, 0, :]  # (B,C)
        else:
            u_prev = u0_eff
            if u_prev.shape != (B, C):
                raise ValueError(f"u0_eff must be (B,{C}), got {tuple(u_prev.shape)}")

        outs = []
        for k in range(L):
            uk = u_nl[:, k, :]
            u_prev = a[:, 0, :] * u_prev + (1.0 - a[:, 0, :]) * uk
            outs.append(u_prev)

        u_eff = torch.stack(outs, dim=1)  # (B,L,C)
        return u_eff
