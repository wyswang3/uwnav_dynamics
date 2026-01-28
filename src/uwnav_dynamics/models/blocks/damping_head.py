# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DampingHeadConfig:
    """
    阻尼一致性输出头（耗散先验）

    业务逻辑：
      显式构造“速度相关耗散项”，提升多步 rollout 稳定性。
      最小实现先只对速度 3 维（v_state）做阻尼项：
        Δv_damp = - d(v) ⊙ v
      其中 d(v) >= 0，通过 softplus 保证非负。

    使用方式：
      - 输入 y_last (B, 9)，内部取 v_last = y_last[..., v_slice]
      - 输出 dY_damp (B, H, 9)，仅速度维度非零，其余为 0
      - enabled=False 时返回全零（对 baseline 无影响）
    """
    enabled: bool = False

    # y 的最后 3 维是 v_state（默认符合你现在的 target_cols）
    v_start: int = 6
    v_dim: int = 3

    # 预测步长 H（为了输出 dY_damp (B,H,9)）
    pred_len: int = 10
    y_dim: int = 9

    # 阻尼模型复杂度：simple=每轴一个系数；mlp=随速度变化
    mode: str = "simple"  # "simple" | "mlp"

    # simple 模式：每轴一个可学习系数（softplus）
    d_init: float = 0.5

    # mlp 模式：用 |v| 和 v^2 做特征生成 d(v)
    mlp_hidden: int = 16


class DampingHead(nn.Module):
    def __init__(self, cfg: DampingHeadConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.mode == "simple":
            # per-axis damping coefficient params
            init = torch.full((cfg.v_dim,), float(cfg.d_init))
            # softplus^-1: p=log(exp(x)-1)
            x = torch.clamp(init, min=1e-6)
            p0 = torch.log(torch.expm1(x))
            self._d_param = nn.Parameter(p0)
            self.mlp = None
        elif cfg.mode == "mlp":
            self._d_param = None
            in_dim = cfg.v_dim * 2  # |v| + v^2
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, cfg.mlp_hidden),
                nn.ReLU(),
                nn.Linear(cfg.mlp_hidden, cfg.v_dim),
            )
        else:
            raise ValueError(f"Unknown mode={cfg.mode!r}")

    def _d_coeff(self, v: torch.Tensor) -> torch.Tensor:
        """
        生成非负阻尼系数 d >= 0
        v: (B, v_dim)
        return: (B, v_dim)
        """
        if self.cfg.mode == "simple":
            d = F.softplus(self._d_param).view(1, -1)  # (1,v_dim)
            return d.expand(v.shape[0], -1)
        # mlp
        feat = torch.cat([v.abs(), v * v], dim=-1)
        d = F.softplus(self.mlp(feat))
        return d

    def forward(self, y_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y_last: (B, y_dim) 当前时刻状态（最后 3 维为速度）

        Returns:
          dY_damp: (B, H, y_dim) 阻尼贡献的“增量项”
        """
        B, D = y_last.shape
        if D != self.cfg.y_dim:
            raise ValueError(f"y_last must be (B,{self.cfg.y_dim}), got (B,{D})")

        if not self.cfg.enabled:
            return torch.zeros((B, self.cfg.pred_len, self.cfg.y_dim), device=y_last.device, dtype=y_last.dtype)

        v0 = y_last[:, self.cfg.v_start : self.cfg.v_start + self.cfg.v_dim]  # (B,3)
        d = self._d_coeff(v0)  # (B,3)

        # Δv_damp = - d ⊙ v
        dv = -(d * v0)  # (B,3)

        # 组装到 9 维增量上
        dY = torch.zeros((B, self.cfg.pred_len, self.cfg.y_dim), device=y_last.device, dtype=y_last.dtype)
        dY[:, :, self.cfg.v_start : self.cfg.v_start + self.cfg.v_dim] = dv.unsqueeze(1).expand(-1, self.cfg.pred_len, -1)
        return dY
