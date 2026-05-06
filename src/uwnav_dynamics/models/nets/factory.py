"""
模块名称：状态预测模型工厂

模块职责：
集中管理训练、评估与 replay 阶段的模型构造，
避免各入口直接绑定单一 `S1Predictor` 类。

主要功能：
1. 根据 `model.name` 构造当前支持的状态转移模型。
2. 保持旧 `s1_predictor` 路径兼容。
3. 接入新的 `stable_transition_core`，支持物理稳定内核实验。
4. 接入常见网络 baseline，支持论文对照实验。

数据流：
TrainYamlConfig 或 model config
    ↓
model.name
    ↓
S1Predictor / StableTransitionCore / common baselines
    ↓
train / eval / replay

依赖模块：
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.models.nets.stable_transition_core
- uwnav_dynamics.models.nets.common_baselines

备注：
- 旧模型仍可直接实例化；本工厂用于主运行链路。
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.models.nets.stable_transition_core import StableTransitionCore
from uwnav_dynamics.models.nets.common_baselines import _SUPPORTED_BASELINE_NAMES, build_common_baseline


def resolve_model_name(cfg: S1PredictorConfig | Any) -> str:
    """从完整训练配置或模型配置中解析模型名。"""
    model_cfg = getattr(cfg, "model", cfg)
    return str(getattr(model_cfg, "name", "s1_predictor"))


def build_state_predictor(cfg: S1PredictorConfig | Any) -> nn.Module:
    """按 `model.name` 构造状态转移预测器。"""
    name = resolve_model_name(cfg)
    if name == "s1_predictor":
        return S1Predictor(cfg)
    if name == "stable_transition_core":
        return StableTransitionCore(cfg)
    if name in _SUPPORTED_BASELINE_NAMES:
        return build_common_baseline(name, cfg)
    raise ValueError(f"Unsupported model.name={name!r}")
