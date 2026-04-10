"""
模块名称：KF 一步状态转移配置测试

模块职责：
验证 quality-context + single-step 的 Phase 2 配置
能够被当前训练主链正确解析，作为后续一步状态转移实验入口。

主要功能：
1. 检查一步数据集 yaml 的 `pred_len=1` 与输入输出维度。
2. 检查一步训练 yaml 的 `pred_len=1`、`din/dout` 与 loss 设置。
3. 确认 rollout 执行索引与当前 9 维状态布局保持一致。

数据流：
dataset yaml / train yaml
    ↓
load_dataset_config() / load_train_config()
    ↓
配置对象断言
"""

from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.preprocess.build_dataset import load_dataset_config
from uwnav_dynamics.train.config import load_train_config


def test_kf_ctx_quality_step_dataset_yaml_parses_with_single_step_horizon() -> None:
    cfg = load_dataset_config(Path("configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml"))

    assert cfg.name == "2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1"
    assert cfg.sliding_cfg.hist_len == 100
    assert cfg.sliding_cfg.pred_len == 1
    assert len(cfg.sliding_cfg.input_cols) == 34
    assert len(cfg.sliding_cfg.target_cols) == 9
    assert cfg.sliding_cfg.label_dvl_mask_mode == "all_true"
    assert "DtSinceDvl_s" in cfg.sliding_cfg.input_cols
    assert "VelKfVarX_body_mps2" in cfg.sliding_cfg.input_cols


def test_kf_ctx_quality_step_train_yaml_parses_as_single_step_transition_run() -> None:
    cfg = load_train_config(Path("configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml"))

    assert cfg.model.din == 34
    assert cfg.model.dout == 9
    assert cfg.model.pred_len == 1
    assert tuple(cfg.model.y_in_idx) == tuple(range(8, 17))
    assert cfg.loss.type == "transition_balance"
    assert cfg.loss.state_mse_weight == 0.25
    assert cfg.loss.delta_huber_weight == 1.0
    assert cfg.loss.tail_weight_power == 0.0
    assert cfg.train.metric == "val_transition_score"
