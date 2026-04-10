"""
模块名称：状态求解器 replay 批量评估入口

模块职责：
面向“多个候选网络模型需要按同一长序列 replay 协议统一比较”的场景，
提供标准化的批量评估、汇总与排行命令行入口。

主要功能：
1. 从 matrix yaml 读取多个候选模型的 train yaml / ckpt / split / replay 参数。
2. 顺序执行每个候选的状态求解器 replay 验证，并把结果落到独立目录。
3. 汇总所有候选的核心指标到 `summary.csv`。
4. 按统一排行协议生成 `ranking.csv`，便于判断当前最优方案。

数据流：
replay matrix yaml
    ↓
load_trained_transition_solver()
    ↓
run_transition_replay()
    ↓
per-run replay artifact
    ↓
summary.csv / ranking.csv / manifest.yaml

依赖模块：
- csv
- yaml
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.experiment.paths
- uwnav_dynamics.solver.replay
- uwnav_dynamics.solver.reporting
- uwnav_dynamics.solver.transition_solver

备注：
- 当前入口服务“离线 replay 统一筛选”，不是训练调度器。
- 排行协议只用于离线候选比较，不等价于闭环最终结论。
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uwnav_dynamics.experiment.layout import load_yaml_dict
from uwnav_dynamics.experiment.paths import relative_path_str, to_snapshot_value
from uwnav_dynamics.solver.replay import load_replay_dataset, run_transition_replay, write_replay_outputs
from uwnav_dynamics.solver.reporting import (
    REPLAY_RANK_METRICS,
    REPLAY_SUMMARY_FIELDS,
    flatten_replay_metrics,
    replay_ranking_protocol,
)
from uwnav_dynamics.solver.transition_solver import load_trained_transition_solver


_ROLE_CHOICES = {"primary", "baseline", "ablation"}


@dataclass(frozen=True)
class ReplayMatrixRunSpec:
    """单个候选模型的 replay 批量评估配置。"""
    name: str
    label: str
    train_yaml: Path
    role: str | None = None
    ckpt: Path | None = None
    split: str | None = None
    device: str | None = None
    min_steps: int | None = None
    max_segments: int | None = None
    max_steps_per_segment: int | None = None
    save_samples: int | None = None


@dataclass(frozen=True)
class ReplayMatrixConfig:
    """replay 批量评估的全局运行配置。"""
    work_dir: Path
    split: str
    device: str | None
    min_steps: int
    max_segments: int | None
    max_steps_per_segment: int | None
    save_samples: int
    fail_fast: bool
    runs: tuple[ReplayMatrixRunSpec, ...] = ()


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a dict, got {type(value)}")
    return value


def _parse_optional_int(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_optional_path(value: Any, *, where: str) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def load_replay_matrix_config(path: str | Path) -> ReplayMatrixConfig:
    """从 yaml 加载 replay 批量评估配置。"""
    raw = load_yaml_dict(path)
    launcher = _require_mapping(raw.get("launcher", {}), where="launcher")
    runs_raw = raw.get("runs", [])
    if not isinstance(runs_raw, list) or len(runs_raw) <= 0:
        raise ValueError("runs must be a non-empty list")

    work_dir = Path(str(launcher.get("work_dir", "")))
    if str(work_dir) == "." or str(work_dir) == "":
        raise ValueError("launcher.work_dir must be provided")

    split = str(launcher.get("split", "test"))
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported launcher.split={split!r}")

    parsed_runs: list[ReplayMatrixRunSpec] = []
    for idx, item in enumerate(runs_raw):
        entry = _require_mapping(item, where=f"runs[{idx}]")
        name = str(entry.get("name", "")).strip()
        label = str(entry.get("label", name)).strip()
        train_yaml = str(entry.get("train_yaml", entry.get("yaml", ""))).strip()
        if not name:
            raise ValueError(f"runs[{idx}].name must be non-empty")
        if not label:
            raise ValueError(f"runs[{idx}].label must be non-empty")
        if not train_yaml:
            raise ValueError(f"runs[{idx}].train_yaml must be provided")
        role_raw = entry.get("role")
        role = None if role_raw in (None, "") else str(role_raw)
        if role is not None and role not in _ROLE_CHOICES:
            raise ValueError(f"runs[{idx}].role={role!r} not in {sorted(_ROLE_CHOICES)}")

        parsed_runs.append(
            ReplayMatrixRunSpec(
                name=name,
                label=label,
                role=role,
                train_yaml=Path(train_yaml),
                ckpt=_parse_optional_path(entry.get("ckpt"), where=f"runs[{idx}].ckpt"),
                split=None if entry.get("split") in (None, "") else str(entry.get("split")),
                device=None if entry.get("device") in (None, "") else str(entry.get("device")),
                min_steps=_parse_optional_int(entry.get("min_steps"), where=f"runs[{idx}].min_steps"),
                max_segments=_parse_optional_int(entry.get("max_segments"), where=f"runs[{idx}].max_segments"),
                max_steps_per_segment=_parse_optional_int(
                    entry.get("max_steps_per_segment"),
                    where=f"runs[{idx}].max_steps_per_segment",
                ),
                save_samples=_parse_optional_int(entry.get("save_samples"), where=f"runs[{idx}].save_samples"),
            )
        )

    return ReplayMatrixConfig(
        work_dir=work_dir,
        split=split,
        device=None if launcher.get("device") in (None, "") else str(launcher.get("device")),
        min_steps=int(launcher.get("min_steps", 50)),
        max_segments=_parse_optional_int(launcher.get("max_segments"), where="launcher.max_segments"),
        max_steps_per_segment=_parse_optional_int(
            launcher.get("max_steps_per_segment"),
            where="launcher.max_steps_per_segment",
        ),
        save_samples=int(launcher.get("save_samples", 8)),
        fail_fast=bool(launcher.get("fail_fast", False)),
        runs=tuple(parsed_runs),
    )


def _resolve_repo_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (repo_root / value).resolve()


def _safe_error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ").strip()


def _float_or_inf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if math.isfinite(out) else float("inf")


def _compute_ranking_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [
        dict(row)
        for row in rows
        if str(row.get("status", "")) == "ok"
        and int(row.get("segment_count", 0) or 0) > 0
    ]
    if not successful:
        return []

    name_order = {str(row["name"]): idx for idx, row in enumerate(successful)}
    for metric_name, _weight in REPLAY_RANK_METRICS:
        ordered = sorted(
            successful,
            key=lambda row: (
                int(_float_or_inf(row.get("nonfinite_trigger_count", 0)) > 0.0),
                _float_or_inf(row.get(metric_name)),
                name_order[str(row["name"])],
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row[f"rank_{metric_name}"] = int(rank)

    total_weight = float(sum(weight for _name, weight in REPLAY_RANK_METRICS))
    for row in successful:
        weighted = sum(float(row[f"rank_{name}"]) * float(weight) for name, weight in REPLAY_RANK_METRICS)
        row["overall_rank_score"] = float(weighted / max(total_weight, 1.0))
        row["selection_pass"] = bool(int(row.get("nonfinite_trigger_count", 0) or 0) == 0)

    ordered_overall = sorted(
        successful,
        key=lambda row: (
            int(not bool(row["selection_pass"])),
            _float_or_inf(row["overall_rank_score"]),
            _float_or_inf(row.get("rmse_global")),
            _float_or_inf(row.get("final_step_rmse_global_mean")),
            _float_or_inf(row.get("rmse_growth_p95")),
            str(row["name"]),
        ),
    )
    for rank, row in enumerate(ordered_overall, start=1):
        row["overall_rank"] = int(rank)
    return ordered_overall


def _write_manifest(*, cfg: ReplayMatrixConfig, path: Path, repo_root: Path) -> None:
    payload = {
        "schema_version": "transition_replay_matrix_v1",
        "launcher": to_snapshot_value(
            {
                "work_dir": _resolve_repo_path(repo_root, cfg.work_dir),
                "split": cfg.split,
                "device": cfg.device,
                "min_steps": cfg.min_steps,
                "max_segments": cfg.max_segments,
                "max_steps_per_segment": cfg.max_steps_per_segment,
                "save_samples": cfg.save_samples,
                "fail_fast": cfg.fail_fast,
            },
            base_dir=path.parent,
        ),
        "ranking_protocol": replay_ranking_protocol(),
        "runs": [
            to_snapshot_value(
                {
                    "name": run.name,
                    "label": run.label,
                    "role": run.role,
                    "train_yaml": _resolve_repo_path(repo_root, run.train_yaml),
                    "ckpt": None if run.ckpt is None else _resolve_repo_path(repo_root, run.ckpt),
                    "split": run.split,
                    "device": run.device,
                    "min_steps": run.min_steps,
                    "max_segments": run.max_segments,
                    "max_steps_per_segment": run.max_steps_per_segment,
                    "save_samples": run.save_samples,
                },
                base_dir=path.parent,
            )
            for run in cfg.runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_summary(*, rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "name",
        "label",
        "role",
        "status",
        "error",
        "train_yaml",
        "ckpt",
        "out_dir",
        "metrics_path",
    ] + REPLAY_SUMMARY_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(payload)


def _write_ranking(*, rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "overall_rank",
        "overall_rank_score",
        "selection_pass",
        "name",
        "label",
        "role",
        "status",
        "out_dir",
        "metrics_path",
    ] + [f"rank_{name}" for name, _weight in REPLAY_RANK_METRICS] + REPLAY_SUMMARY_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(payload)


def run_transition_replay_matrix(cfg: ReplayMatrixConfig, *, repo_root: Path) -> tuple[Path, Path]:
    """按统一协议执行多个候选模型的长序列 replay，并写出 summary/ranking。"""
    work_dir = _resolve_repo_path(repo_root, cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(cfg=cfg, path=work_dir / "manifest.yaml", repo_root=repo_root)

    summary_rows: list[dict[str, Any]] = []
    for run in cfg.runs:
        split = str(run.split or cfg.split)
        out_dir = work_dir / "runs" / run.name
        metrics_path = out_dir / "metrics.yaml"
        row = {
            "name": run.name,
            "label": run.label,
            "role": run.role or "",
            "status": "ok",
            "error": "",
            "train_yaml": relative_path_str(_resolve_repo_path(repo_root, run.train_yaml), base_dir=work_dir),
            "ckpt": "" if run.ckpt is None else relative_path_str(_resolve_repo_path(repo_root, run.ckpt), base_dir=work_dir),
            "out_dir": relative_path_str(out_dir, base_dir=work_dir),
            "metrics_path": relative_path_str(metrics_path, base_dir=work_dir),
        }
        row.update({field: "" for field in REPLAY_SUMMARY_FIELDS})
        try:
            loaded = load_trained_transition_solver(
                train_yaml=_resolve_repo_path(repo_root, run.train_yaml),
                ckpt=None if run.ckpt is None else _resolve_repo_path(repo_root, run.ckpt),
                device=(run.device or cfg.device),
            )
            replay_dataset = load_replay_dataset(loaded.cfg_train.data.data_dir)
            replay_result = run_transition_replay(
                replay_dataset=replay_dataset,
                split_indices_path=loaded.run_layout.split_indices_path,
                split_name=split,
                solver=loaded.solver,
                min_steps=int(run.min_steps if run.min_steps is not None else cfg.min_steps),
                max_segments=run.max_segments if run.max_segments is not None else cfg.max_segments,
                max_steps_per_segment=(
                    run.max_steps_per_segment
                    if run.max_steps_per_segment is not None
                    else cfg.max_steps_per_segment
                ),
                save_samples=int(run.save_samples if run.save_samples is not None else cfg.save_samples),
            )
            cfg_snapshot = {
                "train_yaml": _resolve_repo_path(repo_root, run.train_yaml),
                "ckpt": None if run.ckpt is None else _resolve_repo_path(repo_root, run.ckpt),
                "split": split,
                "device": run.device or cfg.device or str(loaded.cfg_train.run.device),
                "data_dir": loaded.cfg_train.data.data_dir,
                "split_indices_path": loaded.run_layout.split_indices_path,
                "min_steps": int(run.min_steps if run.min_steps is not None else cfg.min_steps),
                "max_segments": run.max_segments if run.max_segments is not None else cfg.max_segments,
                "max_steps_per_segment": (
                    run.max_steps_per_segment
                    if run.max_steps_per_segment is not None
                    else cfg.max_steps_per_segment
                ),
                "save_samples": int(run.save_samples if run.save_samples is not None else cfg.save_samples),
            }
            write_replay_outputs(
                out_dir=out_dir,
                replay_result=replay_result,
                cfg_snapshot=cfg_snapshot,
                path_root=work_dir,
            )
            row.update(flatten_replay_metrics(replay_result.metrics))
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = _safe_error_text(exc)
            if cfg.fail_fast:
                summary_rows.append(row)
                break
        summary_rows.append(row)

    summary_path = work_dir / "summary.csv"
    _write_summary(rows=summary_rows, path=summary_path)

    ranking_rows = _compute_ranking_rows(summary_rows)
    ranking_path = work_dir / "ranking.csv"
    _write_ranking(rows=ranking_rows, path=ranking_path)
    return summary_path, ranking_path


def main() -> int:
    """状态求解器 replay 批量评估命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.transition_replay_matrix")
    ap.add_argument("-c", "--config", type=str, required=True, help="replay matrix yaml")
    args = ap.parse_args()

    repo_root = Path.cwd()
    cfg = load_replay_matrix_config(Path(args.config))
    summary_path, ranking_path = run_transition_replay_matrix(cfg, repo_root=repo_root)
    print(f"[REPLAY_MATRIX] summary written to: {summary_path}")
    print(f"[REPLAY_MATRIX] ranking written to: {ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
