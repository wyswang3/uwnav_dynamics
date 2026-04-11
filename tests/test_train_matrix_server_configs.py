"""
模块名称：服务器 7 卡矩阵配置测试

模块职责：
验证新的 7 卡服务器重训矩阵配置能够被当前 launcher 正确解析，
并确保多步主线与单步分支被拆成两个独立批次，避免 compare 链混入不同 horizon。

主要功能：
1. 检查 quality 7gpu matrix 的基础 train yaml、并发规模和 run 数量。
2. 检查 step 7gpu matrix 的基础 train yaml、并发规模和 run 数量。
3. 确认单步矩阵使用更大的 eval/train batch 以利用单步任务显存优势。
"""

from __future__ import annotations

from uwnav_dynamics.cli.train_matrix import load_matrix_launcher_config
from uwnav_dynamics.cli.server_pipeline import load_server_pipeline_config


def test_quality_7gpu_matrix_yaml_parses_as_8_run_server_batch() -> None:
    cfg = load_matrix_launcher_config("configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml")

    assert str(cfg.base_train_yaml) == "configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml"
    assert str(cfg.work_dir) == "out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2"
    assert cfg.max_parallel == 7
    assert cfg.gpus == ("0", "1", "2", "3", "4", "5", "6")
    assert cfg.run_eval is True
    assert cfg.compare is True
    assert len(cfg.runs) == 8


def test_quality_step_7gpu_matrix_yaml_parses_as_8_run_server_batch() -> None:
    cfg = load_matrix_launcher_config("configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml")

    assert str(cfg.base_train_yaml) == "configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml"
    assert str(cfg.work_dir) == "out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2"
    assert cfg.max_parallel == 7
    assert cfg.gpus == ("0", "1", "2", "3", "4", "5", "6")
    assert cfg.run_eval is True
    assert cfg.compare is True
    assert cfg.eval_batch_size == 2048
    assert len(cfg.runs) == 8


def test_server_pipeline_7gpu_v2_points_to_new_matrix_batches() -> None:
    cfg = load_server_pipeline_config("configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml")

    assert str(cfg.work_dir) == "out/server_pipeline/pooltest02_kf_full_7gpu_v2"
    assert len(cfg.train_matrix_configs) == 2
    assert str(cfg.train_matrix_configs[0]) == "configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml"
    assert str(cfg.train_matrix_configs[1]) == "configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml"
    assert len(cfg.replay_jobs) == 2
