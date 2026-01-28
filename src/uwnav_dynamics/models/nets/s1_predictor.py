# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

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

    enabled=False 时各模块应退化为“无贡献”，保证 baseline 不受影响。
    """
    thruster_lag: ThrusterLagConfig = field(default_factory=lambda: ThrusterLagConfig(enabled=False))
    hydro_ssm: HydroSSMConfig = field(default_factory=lambda: HydroSSMConfig(enabled=False))
    damping: DampingHeadConfig = field(default_factory=lambda: DampingHeadConfig(enabled=False))
    uncertainty: UncertaintyHeadConfig = field(default_factory=lambda: UncertaintyHeadConfig(enabled=False))


@dataclass
class S1PredictorConfig:
    """
    Baseline + blocks 的统一配置。

    din/dout 的含义：
      - din: 输入特征维度（X 的最后一维）
      - dout: 输出目标维度（Y 的最后一维）
      - pred_len: 预测步长 H

    适配“异步监督/事件流（B1）”的新增约定：
      - forward(x, m=mask) 其中 m 与 x 同形状 (B,L,Din)
      - 当 use_input_mask=True：
          encoder 实际输入为 concat([x_enc, m])，因此 LSTM input_size = din*2
      - 当 use_ffill_for_blocks=True：
          block 内部使用 “按 mask 前向保持（ffill）” 的 u/y 序列，避免事件流中缺失造成 block 状态被 0 污染
        （注意：这不会改变 encoder 的原始 x/m，仅保证 blocks 的输入连续性）

    重要：u_in_idx / y_in_idx 始终以“原始 x 的 feature 维”为基准，不因拼 mask 改变。
    """
    din: int = 25
    dout: int = 9
    pred_len: int = 10  # H

    # backbone
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.0

    # ---- x 内部字段切片（建议在 YAML 写死）----
    # layout: [PWM 8] + [IMU 6] + [Vel_state 3] + [Power 8] = 25
    # y_seq 指的是输入中的“状态/观测片段”（默认 IMU6 + Vel_state3 = 9）
    u_in_idx: Sequence[int] = tuple(range(0, 8))
    y_in_idx: Sequence[int] = tuple(range(8, 17))

    # ---- 融合策略 ----
    use_thruster_as_replacement: bool = True
    use_hydro_feat: bool = True

    # ---- 事件流/异步输入支持 ----
    use_input_mask: bool = False          # True: encoder 输入拼接 mask -> LSTM input_size = din*2
    use_ffill_for_blocks: bool = True     # True: 对 blocks 的 u/y 使用 mask ffill（仅当 m 提供时生效）

    # ---- blocks ----
    blocks: S1BlocksConfig = field(default_factory=S1BlocksConfig)


# =============================================================================
# Model
# =============================================================================


class S1Predictor(nn.Module):
    """
    Baseline + 模块化先验（blocks）

    Baseline：
      Encoder LSTM over (x or [x,m]) -> last hidden -> MLP head
      Output:
        dY:     (B,H,Dout)
        logvar: (B,H,Dout)

    异步/事件流输入（B1）：
      - forward(x, m=mask)
      - mask: 1 表示该时刻该维度有效，0 表示缺失（x 中对应值通常为 0）
      - use_input_mask=True 时 encoder 输入为 concat([x_enc, m])

    blocks 在异步输入下的稳定性：
      - use_ffill_for_blocks=True 时，u/y 供 blocks 使用会做按 mask 的前向保持，
        防止 “缺失=0” 误触发 ThrusterLag/HydroSSM 的内部动力学状态。
    """

    def __init__(self, cfg: S1PredictorConfig):
        super().__init__()
        self.cfg = cfg

        # ---------------------------
        # Index buffers
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
        enc_in_dim = int(cfg.din) * (2 if cfg.use_input_mask else 1)
        self.enc = nn.LSTM(
            input_size=enc_in_dim,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ---------------------------
        # 2) Blocks
        # ---------------------------
        self.thruster = ThrusterLag(cfg.blocks.thruster_lag, n_thrusters=self.u_in_dim)

        hydro_cfg = replace(cfg.blocks.hydro_ssm, u_dim=self.u_in_dim, y_dim=self.y_in_dim)
        self.hydro = HydroSSMCell(hydro_cfg)

        damp_cfg = replace(cfg.blocks.damping, pred_len=cfg.pred_len, y_dim=cfg.dout)
        self.damping = DampingHead(damp_cfg)

        unc_cfg = replace(cfg.blocks.uncertainty, pred_len=cfg.pred_len, y_dim=cfg.dout)
        self.uncertainty = UncertaintyHead(unc_cfg)

        # ---------------------------
        # 3) Head (baseline dY + logvar)
        # ---------------------------
        hydro_hidden = int(hydro_cfg.hidden_dim)
        head_in = cfg.rnn_hidden + (hydro_hidden if cfg.use_hydro_feat else 0)

        out_dim = cfg.pred_len * cfg.dout * 2  # dY + logvar
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rnn_hidden, out_dim),
        )

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
        """用 u_new 替换 x 中对应 u_in_idx 的列，保持 x 的最后维 din 不变。"""
        if u_new.shape[:2] != x.shape[:2] or u_new.shape[-1] != self.u_in_dim:
            raise ValueError(f"u_new must be (B,L,{self.u_in_dim}), got {tuple(u_new.shape)}")
        x2 = x.clone()
        x2.index_copy_(-1, self._u_idx, u_new)
        return x2

    @staticmethod
    def _ffill_by_mask(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        对 seq 做按 mask 的前向保持（ffill）。

        Args:
          seq : (B,L,D)
          mask: (B,L,D)  1=valid, 0=missing

        Returns:
          out : (B,L,D)  missing 时刻用最近一次 valid 值替代；序列开头若缺失则保持原值（通常为 0）
        """
        if seq.shape != mask.shape:
            raise ValueError(f"ffill requires same shape, got seq={tuple(seq.shape)} mask={tuple(mask.shape)}")

        B, L, D = seq.shape
        out = seq.clone()

        last = out[:, 0, :]  # (B,D)
        last = torch.where(mask[:, 0, :].to(dtype=torch.bool), last, last)  # no-op but keeps semantics clear

        for t in range(1, L):
            cur = out[:, t, :]
            cur_m = mask[:, t, :].to(dtype=torch.bool)
            last = torch.where(cur_m, cur, last)  # 更新 last（仅当当前有效）
            out[:, t, :] = last
        return out

    def _maybe_concat_mask(self, x_enc: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        """
        encoder 输入拼接 mask：concat([x_enc, m])，要求 m 与 x_enc 同形状。
        """
        if not self.cfg.use_input_mask:
            return x_enc
        if m is None:
            raise ValueError("cfg.use_input_mask=True but mask m is None. Provide m with shape (B,L,Din).")
        if m.shape != x_enc.shape:
            raise ValueError(f"mask m must match x shape. got x={tuple(x_enc.shape)} m={tuple(m.shape)}")
        return torch.cat([x_enc, m.to(dtype=x_enc.dtype)], dim=-1)

    # ---------------------------
    # forward
    # ---------------------------

    def forward(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (B,L,Din)
          m: optional (B,L,Din) mask, 1=valid, 0=missing (for async/event-stream datasets)

        Returns:
          dY:     (B,H,Dout)
          logvar: (B,H,Dout)
        """
        if x.ndim != 3 or x.shape[-1] != self.cfg.din:
            raise ValueError(f"x must be (B,L,{self.cfg.din}), got {tuple(x.shape)}")
        if m is not None and (m.ndim != 3 or m.shape != x.shape):
            raise ValueError(f"m must be None or (B,L,{self.cfg.din}), got {tuple(m.shape)}")

        # 1) 抽取 u/y（供 blocks 使用）
        u_seq = self._slice_u(x)  # (B,L,u_dim)
        y_seq = self._slice_y(x)  # (B,L,y_dim)

        # 1.5) 异步输入：按 mask ffill，避免 blocks 被 “缺失=0” 污染
        if self.cfg.use_ffill_for_blocks and m is not None:
            u_m = self._slice_u(m).to(dtype=u_seq.dtype)
            y_m = self._slice_y(m).to(dtype=y_seq.dtype)
            u_seq_blk = self._ffill_by_mask(u_seq, u_m)
            y_seq_blk = self._ffill_by_mask(y_seq, y_m)
        else:
            u_seq_blk = u_seq
            y_seq_blk = y_seq

        # 2) ThrusterLag：得到 u_eff（enabled=False 时等于 u）
        u_eff = self.thruster(u_seq_blk)

        # 3) 将 u_eff 替换回 x（让 backbone 看到“等效输入”）
        if self.cfg.blocks.thruster_lag.enabled and self.cfg.use_thruster_as_replacement:
            x_enc_base = self._replace_u_in_x(x, u_eff)
        else:
            x_enc_base = x

        # 4) HydroSSM：得到 h_last（enabled=False 时应输出 0）
        _, h_last = self.hydro(u_eff, y_seq_blk)  # (B, hidden_dim)

        # 5) Backbone encoder（可选拼接 mask）
        x_enc = self._maybe_concat_mask(x_enc_base, m)
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

        # 8) DampingHead：显式耗散项
        # 注意：y_last 来自“block 用的 y_seq_blk”，更符合异步/缺失下的语义（缺失会被 ffill）
        y_last = y_seq_blk[:, -1, :]          # (B, y_dim)
        dY_damp = self.damping(y_last)        # (B,H,Dout)，enabled=False 时应为 0
        dY = dY + dY_damp

        # 9) UncertaintyHead：可选替换 logvar
        if self.cfg.blocks.uncertainty.enabled:
            u_last = u_eff[:, -1, :]  # (B,u_dim)
            feat_raw = torch.cat([h_last, y_last, u_last], dim=-1)
            feat = self.unc_feat(feat_raw)
            logvar = self.uncertainty(feat)  # (B,H,Dout)
        else:
            logvar = logvar_base

        return dY, logvar


# =============================================================================
# Backward-compatible aliases (for older config loaders)
# =============================================================================
BlocksConfig = S1BlocksConfig

# 保留旧名导出（下游若 from ... import ThrusterLagConfig 等不会炸）
ThrusterLagConfig = ThrusterLagConfig
HydroSSMConfig = HydroSSMConfig

# 旧名兼容：DampingConfig / UncertaintyConfig
DampingConfig = DampingHeadConfig
UncertaintyConfig = UncertaintyHeadConfig
