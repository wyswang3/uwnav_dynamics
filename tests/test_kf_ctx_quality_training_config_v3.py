"""
模块名称：KF 质量上下文训练配置测试

模块职责：
验证把 DVL 新鲜度与速度方差上下文接入输入后的 v3 配置
能够被当前主链路正确解析，并保持 rollout 执行索引不变。

主要功能：
1. 检查 quality v3 dataset yaml 的输入维度与关键质量字段。
2. 检查 quality v3 train yaml 的 `din / dout / y_in_idx / loss` 设置。
3. 确认质量字段只进入输入上下文，不改变当前 9 维主监督输出。

数据流：
dataset yaml / train yaml
    ↓
load_dataset_config() / load_train_config()
    ↓
配置对象断言

依赖模块：
- uwnav_dynamics.preprocess.build_dataset
- uwnav_dynamics.train.config
"""

from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.preprocess.build_dataset import load_dataset_config
from uwnav_dynamics.train.config import load_train_config


def test_kf_ctx_quality_dataset_yaml_parses_with_34d_input_and_9d_target() -> None:
    cfg = load_dataset_config(Path("configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml"))

    assert cfg.name == "2026-01-10_pooltest02_s1_kf_ctx_quality_v3"
    assert cfg.sliding_cfg.hist_len == 100
    assert cfg.sliding_cfg.pred_len == 10
    assert len(cfg.sliding_cfg.input_cols) == 34
    assert len(cfg.sliding_cfg.target_cols) == 9
    assert cfg.sliding_cfg.label_dvl_mask_mode == "all_true"
    assert "HasDvlUpdate" in cfg.sliding_cfg.input_cols
    assert "DtSinceDvl_s" in cfg.sliding_cfg.input_cols
    assert "VelKfVarZ_body_mps2" in cfg.sliding_cfg.input_cols
    assert "VelKfZ_body_mps" in cfg.sliding_cfg.target_cols


def test_kf_ctx_quality_train_yaml_parses_with_quality_context_and_same_y_layout() -> None:
    cfg = load_train_config(Path("configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml"))

    assert cfg.model.din == 34
    assert cfg.model.dout == 9
    assert tuple(cfg.model.y_in_idx) == tuple(range(8, 17))
    assert cfg.model.head_mode == "grouped"
    assert cfg.loss.type == "transition_balance"
    assert cfg.loss.state_mse_weight == 0.25
    assert cfg.loss.tail_weight_power == 1.2
    assert cfg.train.metric == "val_transition_score"
