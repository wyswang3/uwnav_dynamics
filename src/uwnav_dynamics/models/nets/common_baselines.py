"""
模块名称：常见网络对照模型

模块职责：
提供论文对照与消融实验所需的常见时序网络 baseline，
并保持现有状态转移训练、评估与 replay 的模型输出契约。

主要功能：
1. 提供 MLP pooling、GRU、TCN、Transformer 四类 baseline。
2. 统一输出 `dY / logvar`，兼容 `y_hat = y0 + cumsum(dY)`。
3. 提供 `forward_with_aux()`，使训练入口无需为 baseline 单独分支。

数据流：
X(B,L,Din)
    ↓
baseline encoder
    ↓
context feature
    ↓
linear state head
    ↓
dY(B,H,Dout), logvar(B,H,Dout), aux

系统级数据流：
IMU / DVL / PWM / Power
    ↓
alignment / KF-ESKF proxy state
    ↓
quality_step_v1 dataset
    ↓
common baseline model
    ↓
eval / transition replay
    ↓
architecture comparison report

依赖模块：
- torch
- uwnav_dynamics.models.nets.s1_predictor.S1PredictorConfig
- uwnav_dynamics.models.utils.execution_layout

备注：
- 本模块只服务常见网络对照实验，不改变当前默认 StepBase 主线。
- 第一版 baseline 追求公平、稳定和可复现，不追求每类网络的极限调参。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig
from uwnav_dynamics.models.utils.execution_layout import validate_execution_layout, validate_feature_indices


_SUPPORTED_BASELINE_NAMES = {
    "baseline_mlp",
    "baseline_gru",
    "baseline_tcn",
    "baseline_transformer",
}


class _StateHeadMixin:
    """常见 baseline 共享的输出头逻辑。"""

    cfg: S1PredictorConfig
    state_head: nn.Linear

    def _format_head_output(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.state_head(context)
        raw = raw.view(raw.shape[0], int(self.cfg.pred_len), 2, int(self.cfg.dout))
        dY = raw[:, :, 0, :]
        logvar = raw[:, :, 1, :]
        return dY, logvar

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        dY, logvar = self.forward(x)
        return dY, logvar, {"dvl_obs": None}


def _resolve_cfg(cfg: S1PredictorConfig | Any) -> S1PredictorConfig:
    if isinstance(cfg, S1PredictorConfig):
        return cfg
    if hasattr(cfg, "model") and isinstance(getattr(cfg, "model"), S1PredictorConfig):
        return getattr(cfg, "model")
    raise TypeError("common baseline models expect S1PredictorConfig or TrainYamlConfig-like object")


def _validate_common_cfg(cfg: S1PredictorConfig) -> None:
    validate_feature_indices(cfg.u_in_idx, upper_bound=cfg.din, name="model.u_in_idx")
    validate_execution_layout(cfg.y_in_idx, din=cfg.din, dout=cfg.dout)
    if int(cfg.din) <= 0 or int(cfg.dout) <= 0 or int(cfg.pred_len) <= 0:
        raise ValueError(
            "model.din/dout/pred_len must all be > 0, "
            f"got {cfg.din}/{cfg.dout}/{cfg.pred_len}"
        )
    if int(cfg.rnn_hidden) <= 0 or int(cfg.rnn_layers) <= 0:
        raise ValueError(
            "model.rnn_hidden/rnn_layers must be > 0, "
            f"got {cfg.rnn_hidden}/{cfg.rnn_layers}"
        )


def _check_input(x: torch.Tensor, cfg: S1PredictorConfig) -> None:
    if x.ndim != 3 or x.shape[-1] != int(cfg.din):
        raise ValueError(f"x must be (B,L,{cfg.din}), got {tuple(x.shape)}")


class BaselineMlpPredictor(nn.Module, _StateHeadMixin):
    """历史窗 pooling 后接 MLP 的非循环 baseline。"""

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = _resolve_cfg(cfg)
        _validate_common_cfg(self.cfg)
        hidden = int(self.cfg.rnn_hidden)
        pooled_dim = int(self.cfg.din) * 3
        self.encoder = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.state_head = nn.Linear(hidden, int(self.cfg.pred_len) * int(self.cfg.dout) * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_input(x, self.cfg)
        pooled = torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
        context = self.encoder(pooled)
        return self._format_head_output(context)


class BaselineGruPredictor(nn.Module, _StateHeadMixin):
    """GRU encoder baseline，用于和 LSTM backbone 对照。"""

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = _resolve_cfg(cfg)
        _validate_common_cfg(self.cfg)
        self.enc = nn.GRU(
            input_size=int(self.cfg.din),
            hidden_size=int(self.cfg.rnn_hidden),
            num_layers=int(self.cfg.rnn_layers),
            batch_first=True,
            dropout=float(self.cfg.dropout) if int(self.cfg.rnn_layers) > 1 else 0.0,
            bidirectional=False,
        )
        self.state_head = nn.Linear(int(self.cfg.rnn_hidden), int(self.cfg.pred_len) * int(self.cfg.dout) * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_input(x, self.cfg)
        _out, h_n = self.enc(x)
        return self._format_head_output(h_n[-1])


class BaselineTcnPredictor(nn.Module, _StateHeadMixin):
    """轻量 TCN / 1D CNN baseline。"""

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = _resolve_cfg(cfg)
        _validate_common_cfg(self.cfg)
        kernel_size = int(self.cfg.tcn_kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"model.tcn_kernel_size must be positive odd, got {kernel_size}")
        layers: list[nn.Module] = []
        in_ch = int(self.cfg.din)
        hidden = int(self.cfg.rnn_hidden)
        for _ in range(int(self.cfg.rnn_layers)):
            layers.extend(
                [
                    nn.Conv1d(in_ch, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(self.cfg.dropout)),
                ]
            )
            in_ch = hidden
        self.encoder = nn.Sequential(*layers)
        self.state_head = nn.Linear(hidden, int(self.cfg.pred_len) * int(self.cfg.dout) * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_input(x, self.cfg)
        feat = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self._format_head_output(feat[:, -1, :])


class BaselineTransformerPredictor(nn.Module, _StateHeadMixin):
    """小型 Transformer encoder baseline。"""

    def __init__(self, cfg: S1PredictorConfig | Any):
        super().__init__()
        self.cfg = _resolve_cfg(cfg)
        _validate_common_cfg(self.cfg)
        hidden = int(self.cfg.rnn_hidden)
        nhead = int(self.cfg.transformer_heads)
        if nhead <= 0:
            raise ValueError(f"model.transformer_heads must be > 0, got {nhead}")
        if hidden % nhead != 0:
            raise ValueError(
                "model.rnn_hidden must be divisible by model.transformer_heads, "
                f"got {hidden}/{nhead}"
            )
        self.input_proj = nn.Linear(int(self.cfg.din), hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=max(hidden * 2, int(self.cfg.group_head_hidden)),
            dropout=float(self.cfg.dropout),
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(self.cfg.rnn_layers))
        self.state_head = nn.Linear(hidden, int(self.cfg.pred_len) * int(self.cfg.dout) * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_input(x, self.cfg)
        feat = self.encoder(self.input_proj(x))
        return self._format_head_output(feat[:, -1, :])


def build_common_baseline(name: str, cfg: S1PredictorConfig | Any) -> nn.Module:
    """按 `model.name` 构造常见网络 baseline。"""
    if name == "baseline_mlp":
        return BaselineMlpPredictor(cfg)
    if name == "baseline_gru":
        return BaselineGruPredictor(cfg)
    if name == "baseline_tcn":
        return BaselineTcnPredictor(cfg)
    if name == "baseline_transformer":
        return BaselineTransformerPredictor(cfg)
    raise ValueError(f"Unsupported common baseline model.name={name!r}")
