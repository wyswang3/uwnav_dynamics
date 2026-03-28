"""
模块名称：S1 预测器

模块职责：
定义当前主模型 `S1Predictor`，
将 LSTM backbone、可选物理先验 blocks 与 P0 阶段辅助观测头
组合为统一的多步增量预测器。

主要功能：
1. 根据 canonical `S1PredictorConfig` 构建 encoder、head 与可选 blocks。
2. 使用 `u_in_idx / y_in_idx` 从输入特征中切出控制量与状态量。
3. 在模型构造阶段校验 execution layout contract，并检查 damping 配置与输出语义的一致性。
4. 支持 joint / grouped 两种 state head，以最小改动增强 `acc / gyro / vel` 三组解码能力。
5. 在不破坏主 `forward()` 契约的前提下，为训练阶段提供 `forward_with_aux()`。

数据流：
TrainYamlConfig.model
    ↓
S1PredictorConfig
    ↓
slice u / y from X
    ↓
encoder + optional blocks + shared head trunk
    ↓
dY / logvar / optional aux heads

依赖模块：
- torch
- uwnav_dynamics.models.blocks.*
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- `cfg_model.y_in_idx` 只负责执行层索引解释。
- 输出 9 维的物理语义分组由 semantic output layout contract 单独管理。
- P0.1 的 `dvl_obs` 仅作为训练辅助头，不改变主 state head、eval 或 artifact 协议。
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

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
from uwnav_dynamics.models.utils.execution_layout import validate_execution_layout, validate_feature_indices
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
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


@dataclass(frozen=True)
class AuxHeadConfig:
    """
    单个辅助观测头的最小配置。

    P0.1 只启用 `dvl_obs`，
    通过 enabled/hidden 控制是否构造辅助解码器及其宽度。
    """
    enabled: bool = False
    hidden: int = 128


@dataclass(frozen=True)
class AuxHeadsConfig:
    """
    P0 阶段的辅助观测头配置集合。

    当前只保留 `dvl_obs`，
    后续 P0.2 若新增 IMU 辅助头，可在此 dataclass 上做兼容扩展。
    """
    dvl_obs: AuxHeadConfig = field(default_factory=AuxHeadConfig)


@dataclass(frozen=True)
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

    # ---- 输出头策略 ----
    head_mode: str = "joint"          # "joint" | "grouped"
    group_head_hidden: int = 128

    # ---- P0 辅助观测头 ----
    aux_heads: AuxHeadsConfig = field(default_factory=AuxHeadsConfig)

    # ---- blocks ----
    blocks: S1BlocksConfig = field(default_factory=S1BlocksConfig)


# Backward-compatible aliases for existing config loaders.
BlocksConfig = S1BlocksConfig
DampingConfig = DampingHeadConfig
UncertaintyConfig = UncertaintyHeadConfig


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

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = self._resolve_cfg(cfg)
        cfg = self.cfg
        validate_feature_indices(cfg.u_in_idx, upper_bound=cfg.din, name="model.u_in_idx")
        validate_execution_layout(cfg.y_in_idx, din=cfg.din, dout=cfg.dout)
        semantic_layout = canonical_semantic_output_layout(cfg.dout)
        if str(cfg.head_mode) not in {"joint", "grouped"}:
            raise ValueError(f"Unsupported model.head_mode={cfg.head_mode!r}; expect 'joint' or 'grouped'")
        if int(cfg.group_head_hidden) <= 0:
            raise ValueError(f"model.group_head_hidden must be > 0, got {cfg.group_head_hidden}")

        # 这些索引在每个 forward 都会用到，注册成 buffer 可以避免反复创建 tensor。
        u_idx = torch.as_tensor(list(cfg.u_in_idx), dtype=torch.long)
        y_idx = torch.as_tensor(list(cfg.y_in_idx), dtype=torch.long)
        self.register_buffer("_u_idx", u_idx, persistent=False)
        self.register_buffer("_y_idx", y_idx, persistent=False)
        self.register_buffer(
            "_dvl_semantic_idx",
            torch.as_tensor(list(semantic_layout.group_indices["vel"]), dtype=torch.long),
            persistent=False,
        )
        self._group_order = ("acc", "gyro", "vel")
        self._group_dims = {group: len(semantic_layout.group_indices[group]) for group in self._group_order}

        self.u_in_dim = int(self._u_idx.numel())
        self.y_in_dim = int(self._y_idx.numel())
        self.dvl_obs_dim = int(self._dvl_semantic_idx.numel())

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

        # blocks 的构造顺序与 forward 中的数据流保持一致：输入侧先验 -> 状态侧先验 -> 输出侧先验。
        self.thruster = ThrusterLag(cfg.blocks.thruster_lag, n_thrusters=self.u_in_dim)

        # `replace()` 只用于把运行时维度补回 block config，
        # 避免直接改动 frozen dataclass 引起 train / eval 配置漂移。
        hydro_cfg = replace(cfg.blocks.hydro_ssm, u_dim=self.u_in_dim, y_dim=self.y_in_dim)
        self.hydro = HydroSSMCell(hydro_cfg)

        # DampingHead：同样用 replace 补齐 pred_len / y_dim
        damp_cfg = replace(cfg.blocks.damping, pred_len=cfg.pred_len, y_dim=cfg.dout)
        vel_indices = semantic_layout.group_indices["vel"]
        if damp_cfg.enabled and (
            damp_cfg.v_dim != len(vel_indices) or tuple(range(damp_cfg.v_start, damp_cfg.v_start + damp_cfg.v_dim)) != vel_indices
        ):
            raise ValueError(
                "DampingHeadConfig must match semantic velocity group indices: "
                f"expect start={vel_indices[0]}, dim={len(vel_indices)}, "
                f"got start={damp_cfg.v_start}, dim={damp_cfg.v_dim}"
            )
        self.damping = DampingHead(damp_cfg)

        # UncertaintyHead：同样补齐 pred_len / y_dim
        unc_cfg = replace(cfg.blocks.uncertainty, pred_len=cfg.pred_len, y_dim=cfg.dout)
        self.uncertainty = UncertaintyHead(unc_cfg)

        # 主 head 永远存在；额外 blocks 只是在它的输入或输出上叠加结构先验。
        hydro_hidden = int(hydro_cfg.hidden_dim)  # HydroSSMConfig 必须提供 hidden_dim
        head_in = cfg.rnn_hidden + (hydro_hidden if cfg.use_hydro_feat else 0)
        self.head_trunk = nn.Sequential(
            nn.Linear(head_in, cfg.rnn_hidden),
            nn.ReLU(inplace=True),
        )
        out_dim = cfg.pred_len * cfg.dout * 2  # dY + logvar
        if str(cfg.head_mode) == "joint":
            self.head_joint_out: nn.Module | None = nn.Linear(cfg.rnn_hidden, out_dim)
            self.group_heads: nn.ModuleDict | None = None
        else:
            self.head_joint_out = None
            self.group_heads = nn.ModuleDict(
                {
                    group: nn.Sequential(
                        nn.Linear(cfg.rnn_hidden, int(cfg.group_head_hidden)),
                        nn.ReLU(inplace=True),
                        nn.Linear(
                            int(cfg.group_head_hidden),
                            cfg.pred_len * len(semantic_layout.group_indices[group]) * 2,
                        ),
                    )
                    for group in self._group_order
                }
            )

        if cfg.aux_heads.dvl_obs.enabled:
            self.dvl_obs_head = nn.Sequential(
                nn.Linear(cfg.rnn_hidden, int(cfg.aux_heads.dvl_obs.hidden)),
                nn.ReLU(inplace=True),
                nn.Linear(int(cfg.aux_heads.dvl_obs.hidden), cfg.pred_len * self.dvl_obs_dim),
            )
        else:
            self.dvl_obs_head = None

        # uncertainty head 不直接吃整段序列，而是吃最后时刻的可解释摘要特征。
        feat_in = hydro_hidden + self.y_in_dim + self.u_in_dim
        self.unc_feat = nn.Sequential(
            nn.Linear(feat_in, int(unc_cfg.feat_dim)),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resolve_cfg(cfg: S1PredictorConfig | Any) -> S1PredictorConfig:
        """
        兼容两类构造入口：
          1. 直接传入 `S1PredictorConfig`
          2. 传入完整 `TrainYamlConfig`（含 parser 已解析的 `model_aux_heads`）

        这样 Patch B 不需要改 train/config.py 或 run_train.py，
        也能把 Patch A 已解析的辅助头配置真正接到模型构造。
        """
        if isinstance(cfg, S1PredictorConfig):
            return cfg

        if hasattr(cfg, "model") and hasattr(cfg, "model_aux_heads"):
            model_cfg = getattr(cfg, "model")
            model_aux_heads = getattr(cfg, "model_aux_heads")
            if not isinstance(model_cfg, S1PredictorConfig):
                raise TypeError("cfg.model must be S1PredictorConfig when constructing S1Predictor from TrainYamlConfig")

            dvl_obs_cfg = getattr(model_aux_heads, "dvl_obs", None)
            if dvl_obs_cfg is None:
                return model_cfg

            return replace(
                model_cfg,
                aux_heads=AuxHeadsConfig(
                    dvl_obs=AuxHeadConfig(
                        enabled=bool(getattr(dvl_obs_cfg, "enabled", False)),
                        hidden=int(getattr(dvl_obs_cfg, "hidden", 128)),
                    )
                ),
            )

        raise TypeError(
            "S1Predictor expects S1PredictorConfig or a config object exposing "
            "`model` + `model_aux_heads`"
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

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        """
        Args:
          x: (B,L,Din)

        Returns:
          dY:     (B,H,Dout)  预测增量
          logvar: (B,H,Dout)  对角 log-variance
          aux:    {"dvl_obs": Optional[(B,H,vel_dim)]}
        """
        if x.ndim != 3 or x.shape[-1] != self.cfg.din:
            raise ValueError(f"x must be (B,L,{self.cfg.din}), got {tuple(x.shape)}")

        # 先把输入拆成“控制量”和“状态量”两个语义子序列，供先验模块分别消费。
        u_seq = self._slice_u(x)  # (B,L,u_dim)
        y_seq = self._slice_y(x)  # (B,L,y_dim)

        # 输入侧先验：把原始指令修正成更接近真实推进器响应的等效输入。
        u_eff = self.thruster(u_seq)

        # 如果启用 replacement，就让 backbone 直接看到修正后的输入；否则保留原始 x。
        if self.cfg.blocks.thruster_lag.enabled and self.cfg.use_thruster_as_replacement:
            x_enc = self._replace_u_in_x(x, u_eff)
        else:
            x_enc = x

        # 状态侧先验：用独立隐状态吸收流体记忆，再决定是否并入 head 特征。
        _, h_last = self.hydro(u_eff, y_seq)  # (B, hidden_dim)

        # backbone 仍然是主时序建模器，blocks 是围绕它添加结构化偏置。
        _enc_out, (h_n, _c_n) = self.enc(x_enc)
        h_rnn = h_n[-1]  # (B, rnn_hidden)

        # 是否拼接 hydro 特征由配置决定，方便做“纯 LSTM vs 加先验”消融。
        if self.cfg.use_hydro_feat:
            h_feat = torch.cat([h_rnn, h_last], dim=-1)
        else:
            h_feat = h_rnn

        # baseline head 先给出最基础的 `dY + logvar`，再由后续 blocks 做可解释修正。
        head_feat = self.head_trunk(h_feat)
        B = head_feat.shape[0]
        H = self.cfg.pred_len
        D = self.cfg.dout
        if self.head_joint_out is not None:
            out = self.head_joint_out(head_feat)  # (B, H*Dout*2)
        else:
            if self.group_heads is None:
                raise RuntimeError("grouped head mode requires self.group_heads")
            dy_parts = []
            logvar_parts = []
            for group in self._group_order:
                group_dim = int(self._group_dims[group])
                part = self.group_heads[group](head_feat).view(B, H, group_dim * 2)
                dy_parts.append(part[:, :, :group_dim])
                logvar_parts.append(part[:, :, group_dim:])
            out = torch.cat([torch.cat(dy_parts, dim=-1), torch.cat(logvar_parts, dim=-1)], dim=-1)
        if out.ndim == 2:
            out = out.view(B, H, D * 2)
        dY = out[:, :, :D]
        logvar_base = out[:, :, D:]

        # 输出侧先验：阻尼项只加在物理上更合理的 velocity 语义组。
        y_last = y_seq[:, -1, :]               # (B, y_dim=9)
        dY_damp = self.damping(y_last)         # (B,H,Dout)，enabled=False 时应为 0
        dY = dY + dY_damp

        # 不确定度头打开时，完全接管 baseline logvar；关闭时继续沿用 baseline 输出。
        if self.cfg.blocks.uncertainty.enabled:
            u_last = u_eff[:, -1, :]  # (B,u_dim)
            feat_raw = torch.cat([h_last, y_last, u_last], dim=-1)
            feat = self.unc_feat(feat_raw)
            logvar = self.uncertainty(feat)  # 期望输出 (B,H,Dout)
        else:
            logvar = logvar_base

        aux: dict[str, torch.Tensor | None] = {"dvl_obs": None}
        if self.dvl_obs_head is not None:
            # 辅助头与主 head 共用同一份高层特征，避免再维护第二条编码器路径。
            dvl_out = self.dvl_obs_head(head_feat).view(B, H, self.dvl_obs_dim)
            aux["dvl_obs"] = dvl_out

        return dY, logvar, aux

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对外保持旧路径兼容：仍然只返回主 state head 的 `dY / logvar`。
        辅助观测头输出通过 `forward_with_aux()` 获取。
        """
        dY, logvar, _aux = self.forward_with_aux(x)
        return dY, logvar
