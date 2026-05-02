"""
模块名称：服务器全流程编排入口

模块职责：
面向多 GPU 服务器上的正式实验，
把“融合预处理 -> 数据集构建 -> 单卡 smoke -> 多卡矩阵训练 -> replay 方案评估”
收口为一次配置驱动的全流程编排。

主要功能：
1. 解析 server pipeline yaml，明确每个阶段要执行的配置与输出目录。
2. 顺序调度 fusion、dataset、smoke、train_matrix 与 replay_matrix 五类阶段。
3. 从 train_matrix 的 `summary.csv` 自动生成 replay matrix 配置，避免人工手改候选列表。
4. 统一写出 `manifest.yaml`、`phase_status.csv` 与生成的 replay matrix yaml，便于复现与交接。
5. 在 replay 结束后合并 `summary.csv + ranking.csv`，输出 `final_selection.csv`
   与 `paper_artifact_manifest.yaml` 作为最终选模与论文图表入口。
6. 支持 replay 阶段用秒数配置长时验证，避免把 `50 steps` 误当成 `50s`。

数据流：
server pipeline yaml
    ↓
phase commands / generated replay configs
    ↓
fusion -> dataset -> smoke -> train_matrix -> replay_matrix
    ↓
server work_dir/manifest.yaml + phase_status.csv + final result dirs

依赖模块：
- csv
- subprocess
- yaml
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.experiment.paths

备注：
- 本模块只做 orchestration，不改写 train/eval/replay 的业务逻辑。
- replay 阶段默认从 train_matrix 的 `summary.csv` 中选取 `status=ok` 的候选。
- 当前常见用法是 7 GPU 或 8 GPU 的单卡并发矩阵；具体卡数由下游 matrix yaml 决定。
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uwnav_dynamics.experiment.final_selection import (
    build_final_selection_rows,
    write_final_selection_csv,
    write_paper_artifact_manifest,
)
from uwnav_dynamics.experiment.layout import load_yaml_dict
from uwnav_dynamics.experiment.paths import (
    infer_repo_root,
    relative_path_str,
    resolve_config_path,
    resolve_repo_output_path,
    to_snapshot_value,
)


@dataclass(frozen=True)
class SmokeTrainSpec:
    """单卡 smoke 阶段的最小训练配置。"""
    name: str
    yaml_path: Path
    device: str | None = None
    epochs: int | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class ReplayFromMatrixSpec:
    """基于 train_matrix 汇总表自动生成 replay matrix 的配置。"""
    name: str
    source_summary_csv: Path
    work_dir: Path
    split: str = "test"
    device: str | None = None
    min_steps: int = 50
    min_seconds: float | None = None
    dt_s: float = 0.01
    max_segments: int | None = None
    max_steps_per_segment: int | None = None
    max_seconds_per_segment: float | None = None
    save_samples: int = 8
    include_statuses: tuple[str, ...] = ("ok",)
    top_k: int | None = None


@dataclass(frozen=True)
class ServerPipelineConfig:
    """服务器全流程编排配置。"""
    config_path: Path
    work_dir: Path
    fail_fast: bool
    fusion_yamls: tuple[Path, ...] = ()
    dataset_yamls: tuple[Path, ...] = ()
    smoke_runs: tuple[SmokeTrainSpec, ...] = ()
    train_matrix_configs: tuple[Path, ...] = ()
    replay_jobs: tuple[ReplayFromMatrixSpec, ...] = ()


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a dict, got {type(value)}")
    return value


def _parse_path_list(values: Any, *, where: str) -> tuple[Path, ...]:
    if values in (None, []):
        return ()
    if not isinstance(values, list):
        raise TypeError(f"{where} must be a list")
    return tuple(Path(str(v)) for v in values)


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_server_pipeline_config(path: str | Path) -> ServerPipelineConfig:
    """从 yaml 加载服务器全流程编排配置。"""
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_dict(config_path)
    launcher = _require_mapping(raw.get("launcher", {}), where="launcher")
    preprocess = _require_mapping(raw.get("preprocess", {}), where="preprocess")
    smoke = raw.get("smoke", [])
    train_matrix = raw.get("train_matrix", [])
    replay = raw.get("replay", [])

    if not isinstance(smoke, list):
        raise TypeError("smoke must be a list")
    if not isinstance(train_matrix, list):
        raise TypeError("train_matrix must be a list")
    if not isinstance(replay, list):
        raise TypeError("replay must be a list")

    smoke_runs: list[SmokeTrainSpec] = []
    for idx, item in enumerate(smoke):
        entry = _require_mapping(item, where=f"smoke[{idx}]")
        name = str(entry.get("name", "")).strip()
        yaml_path = str(entry.get("yaml", "")).strip()
        if not name or not yaml_path:
            raise ValueError(f"smoke[{idx}] requires non-empty name and yaml")
        smoke_runs.append(
            SmokeTrainSpec(
                name=name,
                yaml_path=Path(yaml_path),
                device=None if entry.get("device") in (None, "") else str(entry.get("device")),
                epochs=_parse_optional_int(entry.get("epochs")),
                batch_size=_parse_optional_int(entry.get("batch_size")),
            )
        )

    replay_jobs: list[ReplayFromMatrixSpec] = []
    for idx, item in enumerate(replay):
        entry = _require_mapping(item, where=f"replay[{idx}]")
        name = str(entry.get("name", "")).strip()
        source_summary_csv = str(entry.get("source_summary_csv", "")).strip()
        work_dir = str(entry.get("work_dir", "")).strip()
        if not name or not source_summary_csv or not work_dir:
            raise ValueError(
                f"replay[{idx}] requires non-empty name/source_summary_csv/work_dir"
            )
        include_statuses_raw = entry.get("include_statuses", ["ok"])
        if not isinstance(include_statuses_raw, list) or len(include_statuses_raw) <= 0:
            raise ValueError(f"replay[{idx}].include_statuses must be a non-empty list")
        replay_jobs.append(
            ReplayFromMatrixSpec(
                name=name,
                source_summary_csv=Path(source_summary_csv),
                work_dir=Path(work_dir),
                split=str(entry.get("split", "test")),
                device=None if entry.get("device") in (None, "") else str(entry.get("device")),
                min_steps=int(entry.get("min_steps", 50)),
                min_seconds=_parse_optional_float(entry.get("min_seconds")),
                dt_s=float(entry.get("dt_s", entry.get("dt", 0.01))),
                max_segments=_parse_optional_int(entry.get("max_segments")),
                max_steps_per_segment=_parse_optional_int(entry.get("max_steps_per_segment")),
                max_seconds_per_segment=_parse_optional_float(entry.get("max_seconds_per_segment")),
                save_samples=int(entry.get("save_samples", 8)),
                include_statuses=tuple(str(v) for v in include_statuses_raw),
                top_k=_parse_optional_int(entry.get("top_k")),
            )
        )

    return ServerPipelineConfig(
        config_path=config_path,
        work_dir=Path(str(launcher.get("work_dir", "out/server_pipeline/default"))),
        fail_fast=bool(launcher.get("fail_fast", False)),
        fusion_yamls=_parse_path_list(preprocess.get("fusion_yamls", []), where="preprocess.fusion_yamls"),
        dataset_yamls=_parse_path_list(preprocess.get("dataset_yamls", []), where="preprocess.dataset_yamls"),
        smoke_runs=tuple(smoke_runs),
        train_matrix_configs=tuple(Path(str(v)) for v in train_matrix),
        replay_jobs=tuple(replay_jobs),
    )


def _resolve_repo_path(repo_root: Path, value: Path, *, config_dir: Path | None = None) -> Path:
    if config_dir is None:
        return value if value.is_absolute() else (repo_root / value).resolve()
    return resolve_config_path(value, repo_root=repo_root, config_dir=config_dir)


def _build_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    prefix = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = prefix if py_path == "" else (prefix + os.pathsep + py_path)
    return env


def _write_manifest(*, cfg: ServerPipelineConfig, repo_root: Path, path: Path) -> None:
    config_dir = cfg.config_path.parent
    payload = {
        "schema_version": "server_pipeline_v1",
        "launcher": to_snapshot_value(
            {
                "work_dir": _resolve_repo_path(repo_root, cfg.work_dir, config_dir=config_dir),
                "fail_fast": cfg.fail_fast,
            },
            base_dir=path.parent,
        ),
        "preprocess": {
            "fusion_yamls": to_snapshot_value(
                [_resolve_repo_path(repo_root, p, config_dir=config_dir) for p in cfg.fusion_yamls],
                base_dir=path.parent,
            ),
            "dataset_yamls": to_snapshot_value(
                [_resolve_repo_path(repo_root, p, config_dir=config_dir) for p in cfg.dataset_yamls],
                base_dir=path.parent,
            ),
        },
        "smoke": [
            to_snapshot_value(
                {
                    "name": item.name,
                    "yaml": _resolve_repo_path(repo_root, item.yaml_path, config_dir=config_dir),
                    "device": item.device,
                    "epochs": item.epochs,
                    "batch_size": item.batch_size,
                },
                base_dir=path.parent,
            )
            for item in cfg.smoke_runs
        ],
        "train_matrix": to_snapshot_value(
            [_resolve_repo_path(repo_root, p, config_dir=config_dir) for p in cfg.train_matrix_configs],
            base_dir=path.parent,
        ),
        "replay": [
            to_snapshot_value(
                {
                    "name": item.name,
                    "source_summary_csv": _resolve_repo_path(repo_root, item.source_summary_csv, config_dir=config_dir),
                    "work_dir": resolve_repo_output_path(
                        item.work_dir,
                        repo_root=repo_root,
                        config_dir=config_dir,
                        field_name=f"replay[{item.name}].work_dir",
                    ),
                    "split": item.split,
                    "device": item.device,
                    "min_steps": item.min_steps,
                    "min_seconds": item.min_seconds,
                    "dt_s": item.dt_s,
                    "max_segments": item.max_segments,
                    "max_steps_per_segment": item.max_steps_per_segment,
                    "max_seconds_per_segment": item.max_seconds_per_segment,
                    "save_samples": item.save_samples,
                    "include_statuses": list(item.include_statuses),
                    "top_k": item.top_k,
                },
                base_dir=path.parent,
            )
            for item in cfg.replay_jobs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_phase_status(*, rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "phase",
        "name",
        "status",
        "returncode",
        "config_path",
        "generated_config_path",
        "log_path",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _run_logged(*, cmd: list[str], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env(repo_root)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[SERVER_PIPELINE] cmd: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(proc.returncode)


def _smoke_cmd(spec: SmokeTrainSpec) -> list[str]:
    cmd = [sys.executable, "-m", "uwnav_dynamics.cli.train", "-y", str(spec.yaml_path)]
    if spec.device is not None:
        cmd += ["--device", spec.device]
    if spec.epochs is not None:
        cmd += ["--epochs", str(spec.epochs)]
    if spec.batch_size is not None:
        cmd += ["--batch_size", str(spec.batch_size)]
    return cmd


def _build_generated_replay_config(
    *,
    spec: ReplayFromMatrixSpec,
    repo_root: Path,
    generated_dir: Path,
    config_dir: Path,
) -> Path:
    summary_csv = _resolve_repo_path(repo_root, spec.source_summary_csv, config_dir=config_dir)
    if not summary_csv.exists():
        raise FileNotFoundError(f"replay source summary.csv not found: {summary_csv}")

    runs: list[dict[str, Any]] = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() not in set(spec.include_statuses):
                continue
            yaml_rel = str(row.get("yaml_path", "")).strip()
            if yaml_rel == "":
                continue
            runs.append(
                {
                    "name": str(row.get("name", "")).strip(),
                    "label": str(row.get("label", row.get("name", ""))).strip(),
                    "role": str(row.get("role", "")).strip() or None,
                    "train_yaml": relative_path_str(
                        (summary_csv.parent / yaml_rel).resolve(),
                        base_dir=generated_dir,
                    ),
                }
            )

    if spec.top_k is not None:
        runs = runs[: int(spec.top_k)]
    if len(runs) <= 0:
        raise RuntimeError(f"no replay candidates selected from summary.csv: {summary_csv}")

    payload = {
        "launcher": {
            "work_dir": relative_path_str(
                resolve_repo_output_path(
                    spec.work_dir,
                    repo_root=repo_root,
                    config_dir=config_dir,
                    field_name=f"replay[{spec.name}].work_dir",
                ),
                base_dir=generated_dir,
            ),
            "split": spec.split,
            "device": spec.device,
            "min_steps": spec.min_steps,
            "min_seconds": spec.min_seconds,
            "dt_s": spec.dt_s,
            "max_segments": spec.max_segments,
            "max_steps_per_segment": spec.max_steps_per_segment,
            "max_seconds_per_segment": spec.max_seconds_per_segment,
            "save_samples": spec.save_samples,
            "fail_fast": False,
        },
        "runs": runs,
        "source": {
            "summary_csv": relative_path_str(summary_csv, base_dir=generated_dir),
            "include_statuses": list(spec.include_statuses),
            "top_k": spec.top_k,
        },
    }
    generated_dir.mkdir(parents=True, exist_ok=True)
    out_path = generated_dir / f"{spec.name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path


def run_server_pipeline(cfg: ServerPipelineConfig, *, repo_root: Path) -> Path:
    """执行服务器全流程编排，并返回 phase_status.csv 路径。"""
    config_dir = cfg.config_path.parent
    work_dir = resolve_repo_output_path(
        cfg.work_dir,
        repo_root=repo_root,
        config_dir=config_dir,
        field_name="launcher.work_dir",
    )
    logs_dir = work_dir / "logs"
    generated_dir = work_dir / "generated_replay_matrix"
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(cfg=cfg, repo_root=repo_root, path=work_dir / "manifest.yaml")

    phase_rows: list[dict[str, Any]] = []

    def _record(
        *,
        phase: str,
        name: str,
        status: str,
        returncode: int,
        config_path: Path | None,
        generated_config_path: Path | None,
        log_path: Path | None,
        note: str = "",
    ) -> None:
        phase_rows.append(
            {
                "phase": phase,
                "name": name,
                "status": status,
                "returncode": int(returncode),
                "config_path": "" if config_path is None else relative_path_str(config_path, base_dir=work_dir),
                "generated_config_path": (
                    "" if generated_config_path is None else relative_path_str(generated_config_path, base_dir=work_dir)
                ),
                "log_path": "" if log_path is None else relative_path_str(log_path, base_dir=work_dir),
                "note": note,
            }
        )

    def _run_phase(phase: str, name: str, cmd: list[str], config_path: Path) -> bool:
        log_path = logs_dir / f"{phase}_{name}.log"
        rc = _run_logged(cmd=cmd, repo_root=repo_root, log_path=log_path)
        status = "ok" if rc == 0 else "failed"
        _record(
            phase=phase,
            name=name,
            status=status,
            returncode=rc,
            config_path=_resolve_repo_path(repo_root, config_path, config_dir=config_dir),
            generated_config_path=None,
            log_path=log_path,
        )
        if rc != 0 and cfg.fail_fast:
            _write_phase_status(rows=phase_rows, path=work_dir / "phase_status.csv")
            return False
        return True

    for item in cfg.fusion_yamls:
        if not _run_phase(
            "fusion",
            Path(item).stem,
            [sys.executable, "-m", "uwnav_dynamics.preprocess.fusion.cli_fuse_train_base", "-y", str(item)],
            item,
        ):
            return work_dir / "phase_status.csv"

    for item in cfg.dataset_yamls:
        if not _run_phase(
            "dataset",
            Path(item).stem,
            [sys.executable, "-m", "uwnav_dynamics.preprocess.build_dataset", "-y", str(item)],
            item,
        ):
            return work_dir / "phase_status.csv"

    for item in cfg.smoke_runs:
        if not _run_phase("smoke", item.name, _smoke_cmd(item), item.yaml_path):
            return work_dir / "phase_status.csv"

    for item in cfg.train_matrix_configs:
        if not _run_phase(
            "train_matrix",
            Path(item).stem,
            [sys.executable, "-m", "uwnav_dynamics.cli.train_matrix", "-c", str(item)],
            item,
        ):
            return work_dir / "phase_status.csv"

    for item in cfg.replay_jobs:
        generated_cfg = _build_generated_replay_config(
            spec=item,
            repo_root=repo_root,
            generated_dir=generated_dir,
            config_dir=config_dir,
        )
        log_path = logs_dir / f"replay_{item.name}.log"
        rc = _run_logged(
            cmd=[sys.executable, "-m", "uwnav_dynamics.cli.transition_replay_matrix", "-c", str(generated_cfg)],
            repo_root=repo_root,
            log_path=log_path,
        )
        status = "ok" if rc == 0 else "failed"
        _record(
            phase="replay",
            name=item.name,
            status=status,
            returncode=rc,
            config_path=_resolve_repo_path(repo_root, item.source_summary_csv, config_dir=config_dir),
            generated_config_path=generated_cfg,
            log_path=log_path,
        )
        if rc != 0 and cfg.fail_fast:
            _write_phase_status(rows=phase_rows, path=work_dir / "phase_status.csv")
            return work_dir / "phase_status.csv"

    final_selection_rows: list[dict[str, Any]] = []
    train_summary_paths: list[Path] = []
    replay_ranking_paths: list[Path] = []
    for item in cfg.replay_jobs:
        train_summary_csv = _resolve_repo_path(repo_root, item.source_summary_csv, config_dir=config_dir)
        replay_ranking_csv = resolve_repo_output_path(
            item.work_dir,
            repo_root=repo_root,
            config_dir=config_dir,
            field_name=f"replay[{item.name}].work_dir",
        ) / "ranking.csv"
        if not train_summary_csv.exists() or not replay_ranking_csv.exists():
            continue
        train_summary_paths.append(train_summary_csv)
        replay_ranking_paths.append(replay_ranking_csv)
        final_selection_rows.extend(
            build_final_selection_rows(
                selection_scope=item.name,
                train_summary_csv=train_summary_csv,
                replay_ranking_csv=replay_ranking_csv,
                base_dir=work_dir,
            )
        )

    _write_phase_status(rows=phase_rows, path=work_dir / "phase_status.csv")

    if final_selection_rows:
        final_selection_csv = work_dir / "final_selection.csv"
        unique_train_summaries = list(dict.fromkeys(train_summary_paths))
        unique_replay_rankings = list(dict.fromkeys(replay_ranking_paths))
        write_final_selection_csv(rows=final_selection_rows, path=final_selection_csv)
        write_paper_artifact_manifest(
            path=work_dir / "paper_artifact_manifest.yaml",
            work_dir=work_dir,
            phase_status_csv=work_dir / "phase_status.csv",
            final_selection_csv=final_selection_csv,
            train_matrix_summary_paths=unique_train_summaries,
            replay_ranking_paths=unique_replay_rankings,
            final_selection_rows=final_selection_rows,
        )
    return work_dir / "phase_status.csv"


def main() -> int:
    """服务器全流程编排命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.server_pipeline")
    ap.add_argument("-c", "--config", type=str, required=True, help="server pipeline yaml")
    args = ap.parse_args()

    repo_root = infer_repo_root(Path(args.config))
    cfg = load_server_pipeline_config(Path(args.config))
    status_path = run_server_pipeline(cfg, repo_root=repo_root)
    print(f"[SERVER_PIPELINE] phase status written to: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
