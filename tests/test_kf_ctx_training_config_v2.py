"""
模块名称：KF 姿态上下文训练配置测试

模块职责：
验证新引入的 KF 融合状态代理量训练配置能够被现有主链路正确解析，
避免服务器迁移时出现 YAML 契约与代码 parser 脱节。

主要功能：
1. 检查 dataset yaml 的输入/目标维度与关键列约定。
2. 检查 train yaml 的 `din / dout / y_in_idx / loss` 设置。
3. 检查 8 卡 matrix yaml 的并发规模与 run 数量。

数据流：
dataset yaml / train yaml / matrix yaml
    ↓
load_dataset_config() / load_train_config() / load_matrix_launcher_config()
    ↓
配置对象断言

依赖模块：
- uwnav_dynamics.preprocess.build_dataset
- uwnav_dynamics.train.config
- uwnav_dynamics.cli.train_matrix

备注：
- 本测试只覆盖“配置可解析”与“关键契约字段正确”，
  不代表 KF 基础表生成器已经落地实现。
"""

from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.cli.train_matrix import load_matrix_launcher_config
from uwnav_dynamics.preprocess.build_dataset import load_dataset_config
from uwnav_dynamics.train.config import load_train_config


def test_kf_ctx_dataset_yaml_parses_with_29d_input_and_9d_target() -> None:
    cfg = load_dataset_config(Path("configs/dataset/pooltest02_s1_kf_ctx_v2.yaml"))

    assert cfg.name == "2026-01-10_pooltest02_s1_kf_ctx_v2"
    assert cfg.sliding_cfg.hist_len == 100
    assert cfg.sliding_cfg.pred_len == 10
    assert len(cfg.sliding_cfg.input_cols) == 29
    assert len(cfg.sliding_cfg.target_cols) == 9
    assert cfg.sliding_cfg.label_dvl_mask_mode == "all_true"
    assert "RollKf_rad" in cfg.sliding_cfg.input_cols
    assert "CosYawKf" in cfg.sliding_cfg.input_cols
    assert "VelKfZ_body_mps" in cfg.sliding_cfg.target_cols


def test_kf_ctx_train_yaml_parses_with_grouped_head_and_transition_balance() -> None:
    cfg = load_train_config(Path("configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml"))

    assert cfg.model.din == 29
    assert cfg.model.dout == 9
    assert tuple(cfg.model.y_in_idx) == tuple(range(8, 17))
    assert cfg.model.head_mode == "grouped"
    assert cfg.loss.type == "transition_balance"
    assert cfg.loss.tail_weight_power == 1.2
    assert cfg.train.metric == "val_transition_score"


def test_kf_ctx_matrix_yaml_parses_as_8_gpu_8_run_plan() -> None:
    cfg = load_matrix_launcher_config("configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml")

    assert str(cfg.work_dir) == "out/train_matrix/pooltest02_s1_kf_ctx_8gpu_v1"
    assert cfg.max_parallel == 8
    assert cfg.gpus == ("0", "1", "2", "3", "4", "5", "6", "7")
    assert len(cfg.runs) == 8
