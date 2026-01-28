# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import torch
import torch.nn as nn

from uwnav_dynamics.models.blocks import (
    ThrusterLag,
    ThrusterLagConfig,
    HydroSSMCell,
    HydroSSMConfig,
    DampingHead,
    DampingHeadConfig,
    UncertaintyHead,
    UncertaintyHeadConfig,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class S1BlocksConfig:
    """
    S1 组合模型的 blocks 开关与参数集合。

    业务逻辑：
      - 通过 YAML/代码配置，实现不同模块组合的消融实验；
      - enabled=False 时各模块应退化为“无贡献”，保证 baseline 不受影响。
    """
    thruster_lag: ThrusterLagConfig = field(default_factory=lambda: ThrusterLagConfig(enabled=False))
    hydro_ssm: HydroSSMConfig = field(default_factory=lambda: HydroSSMConfig(enabled=False))
    damping: DampingHeadConfig = field(default_factory=lambda: DampingHeadConfig(enabled=False))
    uncertainty: UncertaintyHeadConfig = field(default_factory=lambda: UncertaintyHeadConfig(enabled=False))


@dataclass
class S1PredictorConfig:
    """
    Baseline + blocks 的统一配置。

    din/dout 的含义保持不变：
      - din: 输入特征维度（X 的最后一维）
      - dout: 输出目标维度（Y 的最后一维，等于 9）

    新增：
      - u_in_idx/y_in_idx: 指定在 x 的 feature 维中，u/y 的列索引（默认按 S1 layout）
      - use_thruster_as_replacement: 用 u_eff 替换输入中的 u（不额外增加输入维）
      - use_hydro_feat: 是否将 hydro 的 h_last 拼进 head 特征
      - blocks: 四个模块的配置集合
    """
    din: int = 25
    dout: int = 9
    pred_len: int = 10  # H

    # backbone
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.0

    # ---- x 内部的字段切片约定（强烈建议在 YAML 写死，避免 silent bug）----
    # layout: [PWM 8] + [IMU 6] + [Vel_state 3] + [Power 8] = 25
    # y_seq 指的是目标状态 9 维：[Acc 3] + [Gyro 3] + [Vel_state 3]
    u_in_idx: Sequence[int] = tuple(range(0, 8))
    y_in_idx: Sequence[int] = tuple(range(8, 17))

    # ---- 融合策略 ----
    use_thruster_as_replacement: bool = True
    use_hydro_feat: bool = True

    # ---- blocks ----
    blocks: S1BlocksConfig = field(default_factory=S1BlocksConfig)


# =============================================================================
# Model
# =============================================================================

class S1Predictor(nn.Module):
    """
    Baseline + 模块化先验（blocks）

    Baseline（不启用任何 block）：
      Encoder LSTM over X (B,L,Din) -> last hidden -> MLP head
      Output:
        dY:     (B,H,Dout)  predicted increments
        logvar: (B,H,Dout)  diagonal log-variance

    可选启用：
      1) ThrusterLag：u_seq -> u_eff_seq（deadzone/sat/lag）
      2) HydroSSM： (u_eff_seq, y_seq) -> h_last（流体记忆/隐状态）
      3) DampingHead：由 y_last 构造 dY_damp（显式速度相关耗散）
      4) UncertaintyHead：可替换 baseline 的 logvar 输出（U1）
    """

    def __init__(self, cfg: S1PredictorConfig):
        super().__init__()
        self.cfg = cfg

        # ---------------------------
        # Index buffers (avoid per-forward tensor creation)
        # ---------------------------
        u_idx = torch.as_tensor(list(cfg.u_in_idx), dtype=torch.long)
        y_idx = torch.as_tensor(list(cfg.y_in_idx), dtype=torch.long)
        self.register_buffer("_u_idx", u_idx, persistent=False)
        self.register_buffer("_y_idx", y_idx, persistent=False)

        self.u_in_dim = int(self._u_idx.numel())
        self.y_in_dim = int(self._y_idx.numel())

        # ---------------------------
        # 1) Backbone encoder (LSTM)
        # ---------------------------
        self.enc = nn.LSTM(
            input_size=cfg.din,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ---------------------------
        # 2) Blocks
        # ---------------------------
        # ThrusterLag：enabled=False 时应等价于 identity（u_eff=u）
        self.thruster = ThrusterLag(cfg.blocks.thruster_lag, n_thrusters=self.u_in_dim)

        # HydroSSM：避免“就地修改 cfg”，用 replace 构造局部 cfg
        hydro_cfg = replace(cfg.blocks.hydro_ssm, u_dim=self.u_in_dim, y_dim=self.y_in_dim)
        self.hydro = HydroSSMCell(hydro_cfg)

        # DampingHead：同样用 replace 补齐 pred_len / y_dim
        damp_cfg = replace(cfg.blocks.damping, pred_len=cfg.pred_len, y_dim=cfg.dout)
        self.damping = DampingHead(damp_cfg)

        # UncertaintyHead：同样补齐 pred_len / y_dim
        unc_cfg = replace(cfg.blocks.uncertainty, pred_len=cfg.pred_len, y_dim=cfg.dout)
        self.uncertainty = UncertaintyHead(unc_cfg)

        # ---------------------------
        # 3) Head (baseline dY + logvar)
        # ---------------------------
        hydro_hidden = int(hydro_cfg.hidden_dim)  # HydroSSMConfig 必须提供 hidden_dim
        head_in = cfg.rnn_hidden + (hydro_hidden if cfg.use_hydro_feat else 0)

        out_dim = cfg.pred_len * cfg.dout * 2  # dY + logvar
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rnn_hidden, out_dim),
        )

        # ---- uncertainty feature projection ----
        # 业务逻辑：用可解释特征 (h_last, y_last, u_last) 生成 logvar
        feat_in = hydro_hidden + self.y_in_dim + self.u_in_dim
        self.unc_feat = nn.Sequential(
            nn.Linear(feat_in, int(unc_cfg.feat_dim)),
            nn.ReLU(inplace=True),
        )

    # ---------------------------
    # helpers
    # ---------------------------

    def _slice_u(self, x: torch.Tensor) -> torch.Tensor:
        """从 x 中抽取 u_seq: (B,L,u_dim)"""
        return x.index_select(dim=-1, index=self._u_idx)

    def _slice_y(self, x: torch.Tensor) -> torch.Tensor:
        """从 x 中抽取 y_seq: (B,L,y_dim)"""
        return x.index_select(dim=-1, index=self._y_idx)

    def _replace_u_in_x(self, x: torch.Tensor, u_new: torch.Tensor) -> torch.Tensor:
        """
        用 u_new 替换 x 中对应 u_in_idx 的列，保持 x 的最后维 din 不变。

        工程注意：
          - clone 避免对输入做 in-place 写，防止 autograd / DataLoader pinned-memory / view 共享问题。
          - 这里用 index 写入是允许的，因为我们写入到 clone 后的张量。
        """
        if u_new.shape[:2] != x.shape[:2] or u_new.shape[-1] != self.u_in_dim:
            raise ValueError(f"u_new must be (B,L,{self.u_in_dim}), got {tuple(u_new.shape)}")

        x2 = x.clone()
        x2.index_copy_(-1, self._u_idx, u_new)  # 比 x2[:, :, idx] = u_new 更稳定
        return x2

    # ---------------------------
    # forward
    # ---------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (B,L,Din)

        Returns:
          dY:     (B,H,Dout)  预测增量
          logvar: (B,H,Dout)  对角 log-variance
        """
        if x.ndim != 3 or x.shape[-1] != self.cfg.din:
            raise ValueError(f"x must be (B,L,{self.cfg.din}), got {tuple(x.shape)}")

        # 1) 抽取 u/y（供 blocks 使用）
        u_seq = self._slice_u(x)  # (B,L,u_dim)
        y_seq = self._slice_y(x)  # (B,L,y_dim)

        # 2) ThrusterLag：得到 u_eff（enabled=False 时等于 u）
        u_eff = self.thruster(u_seq)

        # 3) 将 u_eff 替换回 x（让 backbone 看到“等效输入”）
        if self.cfg.blocks.thruster_lag.enabled and self.cfg.use_thruster_as_replacement:
            x_enc = self._replace_u_in_x(x, u_eff)
        else:
            x_enc = x

        # 4) HydroSSM：得到 h_last（enabled=False 时应输出 0）
        # 约定：HydroSSMCell.forward(u_seq, y_seq) -> (h_seq, h_last)
        _, h_last = self.hydro(u_eff, y_seq)  # (B, hidden_dim)

        # 5) Backbone encoder
        _enc_out, (h_n, _c_n) = self.enc(x_enc)
        h_rnn = h_n[-1]  # (B, rnn_hidden)

        # 6) 拼接 head 特征
        if self.cfg.use_hydro_feat:
            h_feat = torch.cat([h_rnn, h_last], dim=-1)
        else:
            h_feat = h_rnn

        # 7) baseline head：输出 dY + logvar
        out = self.head(h_feat)  # (B, H*Dout*2)
        B = out.shape[0]
        H = self.cfg.pred_len
        D = self.cfg.dout
        out = out.view(B, H, D * 2)
        dY = out[:, :, :D]
        logvar_base = out[:, :, D:]

        # 8) DampingHead：显式耗散项（只对速度维更合理）
        y_last = y_seq[:, -1, :]               # (B, y_dim=9)
        dY_damp = self.damping(y_last)         # (B,H,Dout)，enabled=False 时应为 0
        dY = dY + dY_damp

        # 9) UncertaintyHead：可选替换 logvar
        if self.cfg.blocks.uncertainty.enabled:
            u_last = u_eff[:, -1, :]  # (B,u_dim)
            feat_raw = torch.cat([h_last, y_last, u_last], dim=-1)
            feat = self.unc_feat(feat_raw)
            logvar = self.uncertainty(feat)  # 期望输出 (B,H,Dout)
        else:
            logvar = logvar_base

        return dY, logvar
