from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HydroSSMConfig:
    """
    Hydro-SSM 隐状态单元（流体记忆效应先验）

    业务逻辑：
      用一个“稳定递推”的隐状态 h_k 吸收尾流、附加质量、流体记忆等效应：
        h_k = λ ⊙ h_{k-1} + (1-λ) ⊙ φ(Wu*u_k + Wy*y_k + b)
      其中：
        - λ ∈ (0,1) 可学习（sigmoid），保证递推不会发散（核心先验）
        - φ 为非线性（tanh/gelu）
        - 该单元本质是一个稳定的“低通/记忆融合器”，让后续预测头更容易多步 rollout

    输出：
      - h_seq: (B,L,Hh) 隐状态序列
      - h_last: (B,Hh) 最后时刻隐状态（常用于预测头特征）
    """
    enabled: bool = False
    hidden_dim: int = 64

    # 输入维度：u_eff_dim 通常为 8；y_dim 为 9（a/ω/v）
    u_dim: int = 8
    y_dim: int = 9

    # 递推稳定门控 λ 的初始化（越大记忆越强）
    lambda_init: float = 0.85

    # 非线性
    act: str = "tanh"  # "tanh" | "gelu"

    # 是否对每个隐状态维度独立 λ
    per_dim_lambda: bool = True


class HydroSSMCell(nn.Module):
    def __init__(self, cfg: HydroSSMConfig):
        super().__init__()
        self.cfg = cfg

        self.fc = nn.Linear(cfg.u_dim + cfg.y_dim, cfg.hidden_dim)

        # 可学习 λ：用 sigmoid(param) 映射到 (0,1)
        if cfg.per_dim_lambda:
            init = torch.full((cfg.hidden_dim,), float(cfg.lambda_init))
        else:
            init = torch.tensor([float(cfg.lambda_init)])

        # sigmoid^-1(x) = log(x/(1-x))
        x = torch.clamp(init, 1e-4, 1 - 1e-4)
        p0 = torch.log(x / (1.0 - x))
        self._lam_param = nn.Parameter(p0)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act == "tanh":
            return torch.tanh(x)
        if self.cfg.act == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unknown act={self.cfg.act!r}")

    def _lam(self) -> torch.Tensor:
        lam = torch.sigmoid(self._lam_param)  # (Hh,) or (1,)
        if not self.cfg.per_dim_lambda:
            return lam.view(1, 1)  # (1,1)
        return lam.view(1, -1)    # (1,Hh)

    def forward(
        self,
        u_eff_seq: torch.Tensor,
        y_seq: torch.Tensor,
        *,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          u_eff_seq: (B,L,u_dim) 等效输入（来自 ThrusterLag 或直接 u）
          y_seq:     (B,L,y_dim) 状态观测序列（例如 a/ω/v_state）
          h0:        (B,Hh) 初始隐状态（可选），不提供则为 0

        Returns:
          h_seq:  (B,L,Hh)
          h_last: (B,Hh)
        """
        if not self.cfg.enabled:
            # 不启用：输出全零隐状态（对下游 concat 是“无贡献”）
            B, L, _ = y_seq.shape
            h_seq = torch.zeros((B, L, self.cfg.hidden_dim), device=y_seq.device, dtype=y_seq.dtype)
            h_last = h_seq[:, -1, :]
            return h_seq, h_last

        if u_eff_seq.shape[:2] != y_seq.shape[:2]:
            raise ValueError("u_eff_seq and y_seq must share (B,L)")

        B, L, _ = y_seq.shape
        if h0 is None:
            h_prev = torch.zeros((B, self.cfg.hidden_dim), device=y_seq.device, dtype=y_seq.dtype)
        else:
            if h0.shape != (B, self.cfg.hidden_dim):
                raise ValueError(f"h0 must be (B,{self.cfg.hidden_dim}), got {tuple(h0.shape)}")
            h_prev = h0

        lam = self._lam().to(y_seq.device)  # (1,Hh) or (1,1)

        hs = []
        for k in range(L):
            inp = torch.cat([u_eff_seq[:, k, :], y_seq[:, k, :]], dim=-1)
            z = self._act(self.fc(inp))  # (B,Hh)

            if self.cfg.per_dim_lambda:
                h_prev = lam * h_prev + (1.0 - lam) * z
            else:
                # shared scalar lambda
                h_prev = lam[:, 0:1] * h_prev + (1.0 - lam[:, 0:1]) * z

            hs.append(h_prev)

        h_seq = torch.stack(hs, dim=1)  # (B,L,Hh)
        h_last = h_prev
        return h_seq, h_last
