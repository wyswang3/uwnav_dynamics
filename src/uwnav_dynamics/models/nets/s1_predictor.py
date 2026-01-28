# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

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
      - y_in_dim: 在输入 x 中，表示“状态 y_seq”的维度（用于 hydro/damping）
      - u_in_dim: 在输入 x 中，表示“控制 u_seq”的维度（用于 thruster lag）
      - u_in_idx/y_in_idx: 指定在 x 的 feature 维中，u/y 的列索引（默认按你 S1 的 layout）
      - use_hydro_feat: 是否将 hydro 的 h_last 拼进预测头特征
      - blocks: 四个模块的配置集合
    """
    din: int = 25
    dout: int = 9
    pred_len: int = 10  # H

    # backbone
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.0

    # ---- x 内部的“字段切片约定”（非常重要：让 blocks 能拿到 u_seq/y_seq）----
    # 你的 S1 输入 cols 顺序是：[PWM 8] + [IMU 6] + [Vel_state 3] + [Power 8]
    # 也就是：u = [0:8], y = [8:17]?? 这里 y_seq 指的是 9 维目标状态 (a/ω/v_state)，
    # 对应于输入中的：[Acc 3] + [Gyro 3] + [Vel_state 3]，索引为 [8:17]。
    u_in_idx: Sequence[int] = tuple(range(0, 8))
    y_in_idx: Sequence[int] = tuple(range(8, 17))

    u_in_dim: int = 8
    y_in_dim: int = 9

    # ---- 模块融合策略 ----
    use_thruster_as_replacement: bool = True   # 用 u_eff 替换 u（不额外增加输入维）
    use_hydro_feat: bool = True                # 把 h_last concat 到 head 特征（提高表达力）

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

    扩展后（可选启用）：
      1) ThrusterLag：从 u_seq 得到 u_eff_seq（deadzone/sat/lag）
         - 可选择用 u_eff 替换输入中的 u（不改变 din）
      2) HydroSSM：从 (u_eff_seq, y_seq) 得到隐状态 h_last
         - 可选择把 h_last concat 到 head 的特征（head 输入维扩展）
      3) DampingHead：由 y_last 构造 dY_damp（只对速度维度耗散）
         - 直接加到 dY 上，提升 rollout 稳定性
      4) UncertaintyHead：可替换 baseline 的 logvar 输出（U1）
         - enabled=False 时仍由 baseline head 输出 logvar

    注意：
      - 这里保持 enc 输入维度仍为 cfg.din，方便复用你已生成的数据集；
      - ThrusterLag/HydroSSM/Damping/Uncertainty 的“业务逻辑”在 blocks 内部；
      - enabled=False 必须退化为无贡献，以支持消融实验。
    """

    def __init__(self, cfg: S1PredictorConfig):
        super().__init__()
        self.cfg = cfg

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
        # ThrusterLag：默认不启用，不影响 baseline
        self.thruster = ThrusterLag(cfg.blocks.thruster_lag, n_thrusters=cfg.u_in_dim)

        # HydroSSM：默认不启用，输出全零 h
        # 注：u_dim/y_dim 需要与切片维度一致
        hydro_cfg = cfg.blocks.hydro_ssm
        hydro_cfg.u_dim = cfg.u_in_dim
        hydro_cfg.y_dim = cfg.y_in_dim
        self.hydro = HydroSSMCell(hydro_cfg)

        # DampingHead：默认不启用，返回全零 dY_damp
        damp_cfg = cfg.blocks.damping
        damp_cfg.pred_len = cfg.pred_len
        damp_cfg.y_dim = cfg.dout
        self.damping = DampingHead(damp_cfg)

        # UncertaintyHead：默认不启用（则由 baseline head 输出 logvar）
        unc_cfg = cfg.blocks.uncertainty
        unc_cfg.pred_len = cfg.pred_len
        unc_cfg.y_dim = cfg.dout
        # feat_dim 由我们决定：默认用 [h_last, y_last, u_last] -> 投影到 unc_cfg.feat_dim
        self.uncertainty = UncertaintyHead(unc_cfg)

        # ---------------------------
        # 3) Head
        # ---------------------------
        # baseline head 输出 dY + logvar
        # 如果启用 hydro_feat，则 head 输入特征维度增大：rnn_hidden + hydro_hidden_dim
        head_in = cfg.rnn_hidden + (cfg.blocks.hydro_ssm.hidden_dim if cfg.use_hydro_feat else 0)

        out_dim = cfg.pred_len * cfg.dout * 2  # dY + logvar (baseline)
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rnn_hidden, out_dim),
        )

        # 如果启用 uncertainty head，我们还需要一个“uncertainty 特征投影”
        # 业务逻辑：用可解释的特征（h_last, y_last, u_last）生成 logvar
        # 这样 logvar 不会被 dY head 的回归目标牵着跑，便于校准。
        feat_in = (cfg.blocks.hydro_ssm.hidden_dim) + cfg.y_in_dim + cfg.u_in_dim
        self.unc_feat = nn.Sequential(
            nn.Linear(feat_in, cfg.blocks.uncertainty.feat_dim),
            nn.ReLU(inplace=True),
        )

    # ---------------------------
    # helpers
    # ---------------------------

    def _slice_u(self, x: torch.Tensor) -> torch.Tensor:
        """从 x 中抽取 u_seq: (B,L,8)"""
        return x.index_select(dim=-1, index=torch.as_tensor(self.cfg.u_in_idx, device=x.device))

    def _slice_y(self, x: torch.Tensor) -> torch.Tensor:
        """从 x 中抽取 y_seq: (B,L,9)"""
        return x.index_select(dim=-1, index=torch.as_tensor(self.cfg.y_in_idx, device=x.device))

    def _replace_u_in_x(self, x: torch.Tensor, u_new: torch.Tensor) -> torch.Tensor:
        """
        用 u_new 替换 x 中对应 u_in_idx 的列，保持 x 的最后维 din 不变。
        业务逻辑：ThrusterLag 启用后，让 backbone 看到“更真实的等效输入”。
        """
        if u_new.shape != (x.shape[0], x.shape[1], self.cfg.u_in_dim):
            raise ValueError(f"u_new must be (B,L,{self.cfg.u_in_dim})")

        # 克隆一份，避免对原 x 做 in-place 修改导致 autograd/共享内存问题
        x2 = x.clone()
        idx = torch.as_tensor(self.cfg.u_in_idx, device=x.device)
        x2[:, :, idx] = u_new
        return x2

    # ---------------------------
    # forward
    # ---------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,L,Din)

        returns:
          dY:     (B,H,Dout)  预测增量
          logvar: (B,H,Dout)  对角 log-variance
        """
        if x.ndim != 3 or x.shape[-1] != self.cfg.din:
            raise ValueError(f"x must be (B,L,{self.cfg.din}), got {tuple(x.shape)}")

        # 1) 抽取 u/y（供 blocks 使用）
        u_seq = self._slice_u(x)  # (B,L,8)
        y_seq = self._slice_y(x)  # (B,L,9)

        # 2) ThrusterLag：得到 u_eff
        u_eff = self.thruster(u_seq)  # enabled=False 时等于 u_seq

        # 3) 将 u_eff 替换回 x（不改变 din），让 backbone 使用等效输入
        if self.cfg.blocks.thruster_lag.enabled and self.cfg.use_thruster_as_replacement:
            x_enc = self._replace_u_in_x(x, u_eff)
        else:
            x_enc = x

        # 4) HydroSSM：得到 h_last（用于 head 特征）
        _, h_last = self.hydro(u_eff, y_seq)  # enabled=False 时为 0

        # 5) Backbone encoder
        enc_out, (h_n, c_n) = self.enc(x_enc)
        h_rnn = h_n[-1]  # (B, rnn_hidden)

        # 6) 拼接 head 特征
        if self.cfg.use_hydro_feat:
            h_feat = torch.cat([h_rnn, h_last], dim=-1)
        else:
            h_feat = h_rnn

        # 7) baseline head：输出 dY + logvar（未加 damping）
        out = self.head(h_feat)  # (B, H*Dout*2)
        B = out.shape[0]
        H = self.cfg.pred_len
        D = self.cfg.dout
        out = out.view(B, H, D * 2)
        dY = out[:, :, :D]
        logvar_base = out[:, :, D:]

        # 8) DampingHead：显式耗散项（加到 dY 上）
        # 业务逻辑：用 y_last 构造阻尼增量，提升 rollout 稳定性
        y_last = y_seq[:, -1, :]  # (B,9)
        dY_damp = self.damping(y_last)  # enabled=False 时为 0
        dY = dY + dY_damp

        # 9) UncertaintyHead：可选替换 logvar
        if self.cfg.blocks.uncertainty.enabled:
            # 用 [h_last, y_last, u_last] -> feat -> logvar
            u_last = u_eff[:, -1, :]  # (B,8)
            feat_raw = torch.cat([h_last, y_last, u_last], dim=-1)
            feat = self.unc_feat(feat_raw)
            logvar = self.uncertainty(feat)
        else:
            logvar = logvar_base

        return dY, logvar
