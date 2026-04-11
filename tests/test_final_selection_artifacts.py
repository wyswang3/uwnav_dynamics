"""
模块名称：最终选模报表测试

模块职责：
验证训练矩阵汇总表与 replay 排名表可以正确合并成 `final_selection.csv`，
并生成论文图表复用所需的产物清单。

主要功能：
1. 检查 `build_final_selection_rows()` 的字段映射与 winner 标记。
2. 检查 `paper_artifact_manifest.yaml` 是否写出关键表格与 winner 路径。
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from uwnav_dynamics.experiment.final_selection import (
    build_final_selection_rows,
    write_final_selection_csv,
    write_paper_artifact_manifest,
)


def _write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_final_selection_rows_merge_train_and_replay_metrics(tmp_path: Path) -> None:
    train_summary = tmp_path / "train_matrix" / "summary.csv"
    replay_ranking = tmp_path / "replay_matrix" / "ranking.csv"

    _write_csv(
        train_summary,
        fieldnames=[
            "name",
            "label",
            "role",
            "status",
            "yaml_path",
            "run_dir",
            "eval_dir",
            "log_path",
            "monitor_name",
            "best_monitor",
            "selected_val_loss",
            "rmse_global",
            "rmse_global_masked",
            "final_step_rmse_global_masked",
            "rmse_growth_masked",
            "tail_abs_p95_masked",
            "worst_abs_bias_masked",
        ],
        rows=[
            {
                "name": "cand_a",
                "label": "A",
                "role": "primary",
                "status": "ok",
                "yaml_path": "cfgs/cand_a.yaml",
                "run_dir": "runs/cand_a",
                "eval_dir": "runs/cand_a/eval_test",
                "log_path": "logs/cand_a.log",
                "monitor_name": "val_transition_score",
                "best_monitor": 0.12,
                "selected_val_loss": 0.34,
                "rmse_global": 0.10,
                "rmse_global_masked": 0.08,
                "final_step_rmse_global_masked": 0.09,
                "rmse_growth_masked": 1.20,
                "tail_abs_p95_masked": 0.20,
                "worst_abs_bias_masked": 0.03,
            },
        ],
    )
    _write_csv(
        replay_ranking,
        fieldnames=[
            "overall_rank",
            "overall_rank_score",
            "selection_pass",
            "name",
            "label",
            "role",
            "out_dir",
            "metrics_path",
            "rmse_global",
            "mae_global",
            "final_step_rmse_global_mean",
            "rmse_growth_p95",
            "tail_abs_p95_global",
            "tail_abs_p99_global",
            "worst_abs_bias",
            "nonfinite_trigger_count",
        ],
        rows=[
            {
                "overall_rank": 1,
                "overall_rank_score": 1.25,
                "selection_pass": True,
                "name": "cand_a",
                "label": "A",
                "role": "primary",
                "out_dir": "runs/cand_a",
                "metrics_path": "runs/cand_a/metrics.yaml",
                "rmse_global": 0.05,
                "mae_global": 0.04,
                "final_step_rmse_global_mean": 0.06,
                "rmse_growth_p95": 1.10,
                "tail_abs_p95_global": 0.07,
                "tail_abs_p99_global": 0.09,
                "worst_abs_bias": 0.02,
                "nonfinite_trigger_count": 0,
            },
        ],
    )

    rows = build_final_selection_rows(
        selection_scope="quality_step_v1_replay",
        train_summary_csv=train_summary,
        replay_ranking_csv=replay_ranking,
        base_dir=tmp_path,
    )

    assert len(rows) == 1
    assert rows[0]["selection_scope"] == "quality_step_v1_replay"
    assert rows[0]["is_scope_winner"] is True
    assert rows[0]["train_monitor_name"] == "val_transition_score"
    assert rows[0]["replay_rmse_global"] == "0.05"


def test_paper_artifact_manifest_writes_winner_paths(tmp_path: Path) -> None:
    final_selection_csv = tmp_path / "final_selection.csv"
    rows = [
        {
            "selection_scope": "scope_a",
            "is_scope_winner": True,
            "selection_pass": True,
            "overall_rank": 1,
            "overall_rank_score": 1.0,
            "name": "cand_a",
            "label": "Candidate A",
            "role": "primary",
            "train_yaml": "cfgs/cand_a.yaml",
            "train_run_dir": "runs/cand_a",
            "train_eval_dir": "runs/cand_a/eval_test",
            "train_log_path": "logs/cand_a.log",
            "train_status": "ok",
            "train_monitor_name": "val_transition_score",
            "train_best_monitor": 0.1,
            "train_selected_val_loss": 0.2,
            "train_rmse_global": 0.3,
            "train_mae_global": 0.2,
            "train_rmse_global_masked": 0.1,
            "train_mae_global_masked": 0.1,
            "train_final_step_rmse_global_masked": 0.1,
            "train_rmse_growth_masked": 1.0,
            "train_tail_abs_p95_masked": 0.2,
            "train_worst_abs_bias_masked": 0.03,
            "replay_out_dir": "replay/runs/cand_a",
            "replay_metrics_path": "replay/runs/cand_a/metrics.yaml",
            "replay_rmse_global": 0.05,
            "replay_mae_global": 0.04,
            "replay_final_step_rmse_global_mean": 0.06,
            "replay_rmse_growth_p95": 1.1,
            "replay_tail_abs_p95_global": 0.07,
            "replay_tail_abs_p99_global": 0.09,
            "replay_worst_abs_bias": 0.02,
            "replay_nonfinite_trigger_count": 0,
        }
    ]
    write_final_selection_csv(rows=rows, path=final_selection_csv)

    phase_status_csv = tmp_path / "phase_status.csv"
    phase_status_csv.write_text("phase,status\ntrain_matrix,ok\n", encoding="utf-8")
    train_summary_csv = tmp_path / "train_matrix" / "summary.csv"
    train_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    train_summary_csv.write_text("name\ncand_a\n", encoding="utf-8")
    replay_ranking_csv = tmp_path / "replay_matrix" / "ranking.csv"
    replay_ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    replay_ranking_csv.write_text("name\ncand_a\n", encoding="utf-8")

    manifest_path = tmp_path / "paper_artifact_manifest.yaml"
    write_paper_artifact_manifest(
        path=manifest_path,
        work_dir=tmp_path,
        phase_status_csv=phase_status_csv,
        final_selection_csv=final_selection_csv,
        train_matrix_summary_paths=[train_summary_csv],
        replay_ranking_paths=[replay_ranking_csv],
        final_selection_rows=rows,
    )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert payload["tables"]["final_selection_csv"] == "final_selection.csv"
    assert payload["scope_winners"][0]["name"] == "cand_a"
