# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class UncertaintyHeadConfig:
    """
    不确定度输出头（U1：对角异方差）

    业务逻辑：
      输出对角协方差的 log-variance（logvar），用于异方差 NLL：
        y ~ N(μ, diag(exp(logvar)))
      你当前策略是“先做对角协方差”，这是 U1 的合理起点。

    设计要点：
      - logvar 必须裁剪到合理范围避免数值发散（训练更稳）
      - enabled=False 时返回常数 logvar（等价固定方差，退化为 MSE 的尺度变种）
    """
    enabled: bool = False

    pred_len: int = 10
    y_dim: int = 9

    feat_dim: int = 128
    hidden: int = 64

    logvar_min: float = -10.0
    logvar_max: float = 6.0


class UncertaintyHead(nn.Module):
    def __init__(self, cfg: UncertaintyHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.pred_len * cfg.y_dim),
        )

        # disabled 时的常数 logvar（可理解为固定噪声水平）
        self.register_buffer("_const_logvar", torch.zeros((1, cfg.pred_len, cfg.y_dim)), persistent=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
          feat: (B, feat_dim) 用于预测不确定度的特征（通常是 [h_last, y_last, u_last] 拼接后投影）

        Returns:
          logvar: (B, H, y_dim) 对角 log-variance
        """
        B, F = feat.shape
        if F != self.cfg.feat_dim:
            raise ValueError(f"feat must be (B,{self.cfg.feat_dim}), got (B,{F})")

        if not self.cfg.enabled:
            return self._const_logvar.expand(B, -1, -1).to(feat.device)

        out = self.net(feat).view(B, self.cfg.pred_len, self.cfg.y_dim)
        logvar = torch.clamp(out, min=float(self.cfg.logvar_min), max=float(self.cfg.logvar_max))
        return logvar
