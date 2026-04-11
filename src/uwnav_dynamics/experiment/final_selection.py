"""
模块名称：最终选模报表与论文产物清单

模块职责：
把训练矩阵 `summary.csv` 与 replay `ranking.csv` 合并成统一的最终选模表，
并生成可直接服务论文与技术汇报的产物清单。

主要功能：
1. 合并训练期与 replay 期的核心指标，输出 `final_selection.csv`。
2. 标记每个 selection scope 下的 winner，避免不同实验族被错误地跨表混排。
3. 生成 `paper_artifact_manifest.yaml`，收口表格、图目录与 winner 产物路径。

数据流：
train_matrix/summary.csv + replay_matrix/ranking.csv
    ↓
merge / scope-aware winner tagging
    ↓
final_selection.csv
    ↓
paper_artifact_manifest.yaml

依赖模块：
- csv
- yaml
- uwnav_dynamics.experiment.paths

备注：
- 本模块只做离线审计与清单汇总，不负责训练、评估与 replay 本身。
- 不同 selection scope 之间的排行不可直接解释为统一全局优劣。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uwnav_dynamics.experiment.paths import relative_path_str


FINAL_SELECTION_FIELDS = [
    "selection_scope",
    "is_scope_winner",
    "selection_pass",
    "overall_rank",
    "overall_rank_score",
    "name",
    "label",
    "role",
    "train_yaml",
    "train_run_dir",
    "train_eval_dir",
    "train_log_path",
    "train_status",
    "train_monitor_name",
    "train_best_monitor",
    "train_selected_val_loss",
    "train_rmse_global",
    "train_mae_global",
    "train_rmse_global_masked",
    "train_mae_global_masked",
    "train_final_step_rmse_global_masked",
    "train_rmse_growth_masked",
    "train_tail_abs_p95_masked",
    "train_worst_abs_bias_masked",
    "replay_out_dir",
    "replay_metrics_path",
    "replay_rmse_global",
    "replay_mae_global",
    "replay_final_step_rmse_global_mean",
    "replay_rmse_growth_p95",
    "replay_tail_abs_p95_global",
    "replay_tail_abs_p99_global",
    "replay_worst_abs_bias",
    "replay_nonfinite_trigger_count",
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def build_final_selection_rows(
    *,
    selection_scope: str,
    train_summary_csv: Path,
    replay_ranking_csv: Path,
    base_dir: Path,
) -> list[dict[str, Any]]:
    """合并某个 scope 的训练矩阵汇总与 replay 排名。"""
    train_rows = _read_csv_rows(train_summary_csv)
    replay_rows = _read_csv_rows(replay_ranking_csv)
    train_by_name = {str(row.get("name", "")): row for row in train_rows}

    merged: list[dict[str, Any]] = []
    for replay_row in replay_rows:
        name = str(replay_row.get("name", ""))
        train_row = train_by_name.get(name, {})
        merged.append(
            {
                "selection_scope": str(selection_scope),
                "is_scope_winner": bool(int(replay_row.get("overall_rank", 0) or 0) == 1),
                "selection_pass": replay_row.get("selection_pass", ""),
                "overall_rank": replay_row.get("overall_rank", ""),
                "overall_rank_score": replay_row.get("overall_rank_score", ""),
                "name": name,
                "label": replay_row.get("label", train_row.get("label", "")),
                "role": replay_row.get("role", train_row.get("role", "")),
                "train_yaml": train_row.get("yaml_path", ""),
                "train_run_dir": train_row.get("run_dir", ""),
                "train_eval_dir": train_row.get("eval_dir", ""),
                "train_log_path": train_row.get("log_path", ""),
                "train_status": train_row.get("status", ""),
                "train_monitor_name": train_row.get("monitor_name", ""),
                "train_best_monitor": train_row.get("best_monitor", ""),
                "train_selected_val_loss": train_row.get("selected_val_loss", ""),
                "train_rmse_global": train_row.get("rmse_global", ""),
                "train_mae_global": train_row.get("mae_global", ""),
                "train_rmse_global_masked": train_row.get("rmse_global_masked", ""),
                "train_mae_global_masked": train_row.get("mae_global_masked", ""),
                "train_final_step_rmse_global_masked": train_row.get("final_step_rmse_global_masked", ""),
                "train_rmse_growth_masked": train_row.get("rmse_growth_masked", ""),
                "train_tail_abs_p95_masked": train_row.get("tail_abs_p95_masked", ""),
                "train_worst_abs_bias_masked": train_row.get("worst_abs_bias_masked", ""),
                "replay_out_dir": replay_row.get("out_dir", ""),
                "replay_metrics_path": replay_row.get("metrics_path", ""),
                "replay_rmse_global": replay_row.get("rmse_global", ""),
                "replay_mae_global": replay_row.get("mae_global", ""),
                "replay_final_step_rmse_global_mean": replay_row.get("final_step_rmse_global_mean", ""),
                "replay_rmse_growth_p95": replay_row.get("rmse_growth_p95", ""),
                "replay_tail_abs_p95_global": replay_row.get("tail_abs_p95_global", ""),
                "replay_tail_abs_p99_global": replay_row.get("tail_abs_p99_global", ""),
                "replay_worst_abs_bias": replay_row.get("worst_abs_bias", ""),
                "replay_nonfinite_trigger_count": replay_row.get("nonfinite_trigger_count", ""),
            }
        )

    # 统一把相对路径收口到 server_pipeline work_dir，便于交接时直接点击。
    for row in merged:
        for key in ("train_yaml", "train_run_dir", "train_eval_dir", "train_log_path"):
            val = str(row.get(key, "")).strip()
            if val == "":
                continue
            row[key] = relative_path_str((train_summary_csv.parent / val).resolve(), base_dir=base_dir)
        for key in ("replay_out_dir", "replay_metrics_path"):
            val = str(row.get(key, "")).strip()
            if val == "":
                continue
            row[key] = relative_path_str((replay_ranking_csv.parent / val).resolve(), base_dir=base_dir)
    return merged


def write_final_selection_csv(*, rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """写出统一最终选模表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_SELECTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FINAL_SELECTION_FIELDS})


def write_paper_artifact_manifest(
    *,
    path: Path,
    work_dir: Path,
    phase_status_csv: Path,
    final_selection_csv: Path,
    train_matrix_summary_paths: Sequence[Path],
    replay_ranking_paths: Sequence[Path],
    final_selection_rows: Sequence[Mapping[str, Any]],
) -> None:
    """写出论文与技术汇报优先复用的产物清单。"""
    winners = [row for row in final_selection_rows if bool(row.get("is_scope_winner", False))]
    payload = {
        "schema_version": "paper_artifact_manifest_v1",
        "work_dir": relative_path_str(work_dir, base_dir=path.parent),
        "tables": {
            "phase_status_csv": relative_path_str(phase_status_csv, base_dir=path.parent),
            "final_selection_csv": relative_path_str(final_selection_csv, base_dir=path.parent),
            "train_matrix_summaries": [
                relative_path_str(item, base_dir=path.parent) for item in train_matrix_summary_paths
            ],
            "replay_rankings": [
                relative_path_str(item, base_dir=path.parent) for item in replay_ranking_paths
            ],
        },
        "figures": {
            "train_compare_dirs": [
                relative_path_str(item.parent / "compare_test", base_dir=path.parent)
                for item in train_matrix_summary_paths
                if (item.parent / "compare_test").exists()
            ],
            "replay_compare_dirs": [
                relative_path_str(item.parent / "compare_test", base_dir=path.parent)
                for item in replay_ranking_paths
                if (item.parent / "compare_test").exists()
            ],
        },
        "scope_winners": [
            {
                "selection_scope": row.get("selection_scope", ""),
                "name": row.get("name", ""),
                "label": row.get("label", ""),
                "role": row.get("role", ""),
                "train_eval_dir": row.get("train_eval_dir", ""),
                "replay_out_dir": row.get("replay_out_dir", ""),
                "replay_metrics_path": row.get("replay_metrics_path", ""),
            }
            for row in winners
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
