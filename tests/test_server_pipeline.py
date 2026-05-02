"""
模块名称：服务器全流程编排测试

模块职责：
验证“预处理 -> smoke -> train_matrix -> replay_matrix”总控入口的最小编排行为，
并确认它会自动从 train_matrix 的 `summary.csv` 生成 replay matrix 配置。

主要功能：
1. 构造一个最小 server pipeline yaml 与假的 train_matrix summary.csv。
2. monkeypatch 子进程执行，避免真实训练与评估。
3. 调用 `cli.server_pipeline` 并检查 phase 状态表、生成的 replay config 与命令顺序。

数据流：
server pipeline yaml + fake train_matrix summary.csv
    ↓
server_pipeline.main()
    ↓
manifest.yaml / phase_status.csv / generated replay matrix yaml
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest
import yaml

from uwnav_dynamics.cli import server_pipeline


def test_server_pipeline_orchestrates_phases_and_generates_replay_config(tmp_path, monkeypatch):
    work_dir = tmp_path / "server_out"
    matrix_dir = tmp_path / "train_matrix_out"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    train_yaml = tmp_path / "candidate.yaml"
    train_yaml.write_text("run:\n  out_dir: out\n  variant: demo\n", encoding="utf-8")

    summary_csv = matrix_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "label", "role", "status", "yaml_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "candidate_a",
                "label": "Candidate A",
                "role": "primary",
                "status": "ok",
                "yaml_path": str(train_yaml.relative_to(matrix_dir.parent)),
            }
        )
        writer.writerow(
            {
                "name": "candidate_b",
                "label": "Candidate B",
                "role": "ablation",
                "status": "failed",
                "yaml_path": str(train_yaml.relative_to(matrix_dir.parent)),
            }
        )

    cfg_path = tmp_path / "server_pipeline.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": str(work_dir),
                    "fail_fast": False,
                },
                "preprocess": {
                    "fusion_yamls": ["configs/fusion/demo.yaml"],
                    "dataset_yamls": ["configs/dataset/demo.yaml"],
                },
                "smoke": [
                    {
                        "name": "smoke_a",
                        "yaml": "configs/train/demo.yaml",
                        "device": "cpu",
                        "epochs": 1,
                    }
                ],
                "train_matrix": ["configs/launch/demo_matrix.yaml"],
                "replay": [
                    {
                        "name": "matrix_replay",
                        "source_summary_csv": str(summary_csv),
                        "work_dir": str(tmp_path / "replay_out"),
                        "split": "test",
                        "device": "cpu",
                        "min_seconds": 3,
                        "max_seconds_per_segment": 8,
                        "dt_s": 1.0,
                        "save_samples": 2,
                        "include_statuses": ["ok"],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, text=None):
        calls.append(list(cmd))

        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr(server_pipeline.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.server_pipeline",
            "-c",
            str(cfg_path),
        ],
    )

    assert server_pipeline.main() == 0

    phase_status = work_dir / "phase_status.csv"
    manifest = work_dir / "manifest.yaml"
    generated_cfg = work_dir / "generated_replay_matrix" / "matrix_replay.yaml"
    assert phase_status.exists()
    assert manifest.exists()
    assert generated_cfg.exists()

    with phase_status.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["phase"] for row in rows] == ["fusion", "dataset", "smoke", "train_matrix", "replay"]
    assert all(row["status"] == "ok" for row in rows)

    gen = yaml.safe_load(generated_cfg.read_text(encoding="utf-8"))
    assert gen["launcher"]["split"] == "test"
    assert gen["launcher"]["min_seconds"] == 3.0
    assert gen["launcher"]["dt_s"] == 1.0
    assert gen["launcher"]["max_seconds_per_segment"] == 8.0
    assert len(gen["runs"]) == 1
    assert gen["runs"][0]["name"] == "candidate_a"

    assert calls[0][2] == "uwnav_dynamics.preprocess.fusion.cli_fuse_train_base"
    assert calls[1][2] == "uwnav_dynamics.preprocess.build_dataset"
    assert calls[2][2] == "uwnav_dynamics.cli.train"
    assert calls[3][2] == "uwnav_dynamics.cli.train_matrix"
    assert calls[4][2] == "uwnav_dynamics.cli.transition_replay_matrix"


def test_server_pipeline_resolves_relative_workdirs_from_config_dir(tmp_path, monkeypatch):
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    artifacts_root = tmp_path / "artifacts"
    matrix_dir = artifacts_root / "train_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    train_yaml = artifacts_root / "candidate.yaml"
    train_yaml.write_text("run:\n  out_dir: out\n  variant: demo\n", encoding="utf-8")

    summary_csv = matrix_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "label", "role", "status", "yaml_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "candidate_a",
                "label": "Candidate A",
                "role": "primary",
                "status": "ok",
                "yaml_path": "../candidate.yaml",
            }
        )

    cfg_path = config_root / "server_pipeline_rel.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": "../artifacts/server_out",
                    "fail_fast": False,
                },
                "preprocess": {
                    "fusion_yamls": ["configs/fusion/demo.yaml"],
                    "dataset_yamls": ["configs/dataset/demo.yaml"],
                },
                "smoke": [
                    {
                        "name": "smoke_a",
                        "yaml": "configs/train/demo.yaml",
                        "device": "cpu",
                        "epochs": 1,
                    }
                ],
                "train_matrix": ["configs/launch/demo_matrix.yaml"],
                "replay": [
                    {
                        "name": "matrix_replay",
                        "source_summary_csv": "../artifacts/train_matrix/summary.csv",
                        "work_dir": "../artifacts/replay_out",
                        "split": "test",
                        "device": "cpu",
                        "min_steps": 3,
                        "save_samples": 2,
                        "include_statuses": ["ok"],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    def _fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, text=None):
        class _Proc:
            returncode = 0
        return _Proc()

    monkeypatch.setattr(server_pipeline.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.server_pipeline",
            "-c",
            str(cfg_path),
        ],
    )

    assert server_pipeline.main() == 0

    work_dir = artifacts_root / "server_out"
    generated_cfg = work_dir / "generated_replay_matrix" / "matrix_replay.yaml"
    assert (work_dir / "phase_status.csv").exists()
    assert generated_cfg.exists()

    gen = yaml.safe_load(generated_cfg.read_text(encoding="utf-8"))
    assert gen["launcher"]["work_dir"] == "../../replay_out"
    assert gen["runs"][0]["train_yaml"] == "../../candidate.yaml"


def test_server_pipeline_rejects_workdir_outside_inferred_repo_root(tmp_path, monkeypatch):
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text("name,label,role,status,yaml_path\n", encoding="utf-8")

    cfg_path = config_root / "server_pipeline_outside.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "launcher": {
                    "work_dir": "../../outside/server_out",
                    "fail_fast": False,
                },
                "preprocess": {},
                "smoke": [],
                "train_matrix": [],
                "replay": [
                    {
                        "name": "matrix_replay",
                        "source_summary_csv": str(summary_csv),
                        "work_dir": "../../outside/replay_out",
                        "split": "test",
                        "device": "cpu",
                        "min_steps": 3,
                        "save_samples": 2,
                        "include_statuses": ["ok"],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.server_pipeline",
            "-c",
            str(cfg_path),
        ],
    )

    with pytest.raises(ValueError, match="outside inferred repo root"):
        server_pipeline.main()
