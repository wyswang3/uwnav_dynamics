"""
模块名称：多 GPU 实验矩阵调度入口

模块职责：
面向“同一份基础数据与主模型配置，想并行比较多个建模思路”的场景，
负责把一个基础 train yaml 扩展成多个变体 yaml，
并按 GPU 列表调度 `train -> eval -> compare` 全流程。

主要功能：
1. 从 matrix yaml 读取基础 train yaml、公共 override 与每个变体的局部 override。
2. 为每个变体物化独立 train yaml，确保 `run.out_dir / run.variant` 与评估路径可追溯。
3. 按 GPU 令牌并发执行单卡训练与可选评估，避免当前小模型硬上 DDP。
4. 汇总每个变体的 `metrics.yaml` 为一张 CSV，并生成多模型 compare 图。

数据流：
matrix yaml + base train yaml
    ↓
deep merge / generated train yamls
    ↓
GPU scheduler（每个变体独占一张卡）
    ↓
train.run_train
    ↓
cli.eval
    ↓
summary.csv + compare plots

依赖模块：
- yaml
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.viz.eval.plot_model_compare

备注：
- 当前实现是“8 卡并发实验矩阵”，不是单模型 DDP。
- 这样做的原因是当前 baseline 只有约 1M 参数，更适合并发比较多种思路。
- 生成的 train yaml 会作为本次实验的显式审计产物保留在 `configs/train/generated/<work_dir_name>/`，
  避免在 `out/` 等运行目录中混入新的训练配置文件。
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uwnav_dynamics.experiment.layout import RunLayout, load_yaml_dict


_ROLE_CHOICES = {"primary", "baseline", "ablation"}
_COMPARE_METRIC_CHOICES = {"rmse", "mae"}


@dataclass(frozen=True)
class MatrixRunSpec:
    """实验矩阵中单个变体的标识、标签和局部 override。"""
    name: str
    label: str
    role: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixLauncherConfig:
    """多 GPU 实验矩阵调度的完整运行配置。"""
    base_train_yaml: Path
    work_dir: Path
    gpus: tuple[str, ...]
    max_parallel: int
    run_eval: bool
    eval_split: str
    eval_device: str | None
    eval_batch_size: int | None
    plots: bool
    plot_fmt: str
    dt: float
    x_axis: str
    n_plot_samples: int
    compare: bool
    compare_metrics: tuple[str, ...]
    fail_fast: bool
    common_overrides: dict[str, Any] = field(default_factory=dict)
    runs: tuple[MatrixRunSpec, ...] = ()


@dataclass(frozen=True)
class PreparedMatrixRun:
    """物化后的单个实验变体及其路径约定。"""
    spec: MatrixRunSpec
    yaml_path: Path
    run_dir: Path
    eval_dir: Path
    log_path: Path


@dataclass
class ActiveStage:
    """当前正在某张 GPU 上执行的训练或评估阶段。"""
    run: PreparedMatrixRun
    gpu: str
    stage: str
    cmd: list[str]
    proc: subprocess.Popen[str]
    log_handle: Any


@dataclass(frozen=True)
class RunOutcome:
    """单个实验变体执行完成后的结果摘要。"""
    name: str
    label: str
    role: str | None
    gpu: str
    status: str
    failed_stage: str | None
    returncode: int
    yaml_path: Path
    run_dir: Path
    eval_dir: Path
    log_path: Path


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a dict, got {type(value)}")
    return value


def _deep_merge_dict(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """
    递归合并两个 mapping，不原地修改输入对象。

    规则：
    - dict 对 dict：继续递归合并
    - 其他类型：用 update 覆盖 base
    """
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_gpu_list(values: Sequence[Any]) -> tuple[str, ...]:
    if len(values) == 0:
        raise ValueError("launcher.gpus must contain at least one GPU id")
    return tuple(str(v) for v in values)


def _normalize_compare_metrics(values: Sequence[Any]) -> tuple[str, ...]:
    metrics = tuple(str(v) for v in values)
    if len(metrics) == 0:
        raise ValueError("launcher.compare_metrics must not be empty when compare=true")
    extra = [m for m in metrics if m not in _COMPARE_METRIC_CHOICES]
    if extra:
        raise ValueError(
            f"unsupported compare metric(s): {extra}; allowed={sorted(_COMPARE_METRIC_CHOICES)}"
        )
    return metrics


def load_matrix_launcher_config(path: str | Path) -> MatrixLauncherConfig:
    """从矩阵 launcher YAML 解析强类型调度配置。"""
    raw = load_yaml_dict(path)

    base_train_yaml = Path(str(raw.get("base_train_yaml", "")))
    if not str(base_train_yaml):
        raise KeyError("Missing required key: base_train_yaml")

    launcher_d = _require_mapping(raw.get("launcher", {}) or {}, where="launcher")
    common_overrides = _require_mapping(raw.get("common_overrides", {}) or {}, where="common_overrides")
    raw_runs = raw.get("runs", [])
    if not isinstance(raw_runs, list) or len(raw_runs) == 0:
        raise ValueError("runs must be a non-empty list")

    gpus = _normalize_gpu_list(launcher_d.get("gpus", []))
    max_parallel = int(launcher_d.get("max_parallel", len(gpus)))
    if max_parallel <= 0:
        raise ValueError("launcher.max_parallel must be > 0")

    run_eval = bool(launcher_d.get("run_eval", True))
    compare = bool(launcher_d.get("compare", True))
    if compare and not run_eval:
        raise ValueError("launcher.compare=true requires launcher.run_eval=true")

    runs: list[MatrixRunSpec] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(raw_runs):
        run_d = _require_mapping(item, where=f"runs[{idx}]")
        name = str(run_d.get("name", "")).strip()
        if not name:
            raise ValueError(f"runs[{idx}].name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"duplicate runs[{idx}].name={name!r}")
        seen_names.add(name)

        label = str(run_d.get("label", name))
        role_raw = run_d.get("role", None)
        role = None if role_raw is None else str(role_raw)
        if role is not None and role not in _ROLE_CHOICES:
            raise ValueError(
                f"runs[{idx}].role={role!r} is invalid; allowed={sorted(_ROLE_CHOICES)}"
            )
        overrides = _require_mapping(run_d.get("overrides", {}) or {}, where=f"runs[{idx}].overrides")
        runs.append(MatrixRunSpec(name=name, label=label, role=role, overrides=overrides))

    compare_metrics = _normalize_compare_metrics(
        launcher_d.get("compare_metrics", ["rmse", "mae"])
    )

    return MatrixLauncherConfig(
        base_train_yaml=base_train_yaml,
        work_dir=Path(str(launcher_d.get("work_dir", "out/train_matrix/default"))),
        gpus=gpus,
        max_parallel=min(max_parallel, len(gpus)),
        run_eval=run_eval,
        eval_split=str(launcher_d.get("eval_split", "test")),
        eval_device=(str(launcher_d["eval_device"]) if "eval_device" in launcher_d and launcher_d["eval_device"] is not None else None),
        eval_batch_size=(
            int(launcher_d["eval_batch_size"])
            if "eval_batch_size" in launcher_d and launcher_d["eval_batch_size"] is not None
            else None
        ),
        plots=bool(launcher_d.get("plots", False)),
        plot_fmt=str(launcher_d.get("plot_fmt", "png")),
        dt=float(launcher_d.get("dt", 0.01)),
        x_axis=str(launcher_d.get("x_axis", "sec")),
        n_plot_samples=int(launcher_d.get("n_plot_samples", 8)),
        compare=compare,
        compare_metrics=compare_metrics,
        fail_fast=bool(launcher_d.get("fail_fast", False)),
        common_overrides=common_overrides,
        runs=tuple(runs),
    )


def _default_run_config(
    *,
    merged: dict[str, Any],
    spec: MatrixRunSpec,
    launcher_cfg: MatrixLauncherConfig,
) -> dict[str, Any]:
    run_d = _require_mapping(merged.setdefault("run", {}), where="merged.run")
    common_run = launcher_cfg.common_overrides.get("run", {})
    spec_run = spec.overrides.get("run", {})
    if not isinstance(common_run, dict):
        common_run = {}
    if not isinstance(spec_run, dict):
        spec_run = {}

    if "name" not in common_run and "name" not in spec_run:
        run_d["name"] = spec.name
    if "variant" not in common_run and "variant" not in spec_run:
        run_d["variant"] = spec.name
    if "out_dir" not in common_run and "out_dir" not in spec_run:
        run_d["out_dir"] = str(launcher_cfg.work_dir / "runs")
    return merged


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _resolve_generated_train_dir(cfg: MatrixLauncherConfig, *, repo_root: Path) -> Path:
    """把矩阵物化后的训练 YAML 统一放到 `configs/train/generated/<work_dir_name>/`。"""
    work_dir_name = cfg.work_dir.name or "default"
    generated_dir = repo_root / "configs" / "train" / "generated" / work_dir_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


def prepare_matrix_runs(cfg: MatrixLauncherConfig, *, repo_root: Path) -> list[PreparedMatrixRun]:
    """根据基础 YAML 与各变体 override 生成独立运行配置文件。"""
    base_yaml_path = cfg.base_train_yaml
    if not base_yaml_path.is_absolute():
        base_yaml_path = repo_root / base_yaml_path
    base_yaml = load_yaml_dict(base_yaml_path)

    resolved_work_dir = _resolve_repo_path(repo_root, cfg.work_dir)
    generated_dir = _resolve_generated_train_dir(cfg, repo_root=repo_root)
    logs_dir = resolved_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedMatrixRun] = []
    seen_run_dirs: set[Path] = set()
    for spec in cfg.runs:
        # merge 顺序固定为：base -> common_overrides -> run-specific overrides，
        # 保证每个实验变体都能从生成后的 yaml 直接复现。
        merged = _deep_merge_dict(base_yaml, cfg.common_overrides)
        merged = _deep_merge_dict(merged, spec.overrides)
        merged = _default_run_config(merged=merged, spec=spec, launcher_cfg=cfg)

        run_d = _require_mapping(merged.get("run", {}), where="merged.run")
        layout = RunLayout(
            out_dir=Path(str(run_d["out_dir"])),
            variant=str(run_d["variant"]),
        )
        if layout.run_dir in seen_run_dirs:
            raise ValueError(f"duplicate run_dir detected in matrix config: {layout.run_dir}")
        seen_run_dirs.add(layout.run_dir)

        # 生成态 yaml 本身就是实验审计产物，后续 summary / manifest 都引用它。
        yaml_path = generated_dir / f"{spec.name}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)

        prepared.append(
            PreparedMatrixRun(
                spec=spec,
                yaml_path=yaml_path,
                run_dir=layout.run_dir,
                eval_dir=layout.eval_dir(cfg.eval_split),
                log_path=logs_dir / f"{spec.name}.log",
            )
        )

    return prepared


def _build_env(*, repo_root: Path, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    src_path = str(repo_root / "src")
    py_path = env.get("PYTHONPATH", "")
    parts = [p for p in py_path.split(os.pathsep) if p]
    if src_path not in parts:
        parts.insert(0, src_path)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def _open_stage_log(path: Path, *, stage: str, gpu: str, cmd: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    handle.write(f"\n===== stage={stage} gpu={gpu} =====\n")
    handle.write(f"{_format_cmd(cmd)}\n")
    handle.flush()
    return handle


def _launch_stage(
    *,
    run: PreparedMatrixRun,
    gpu: str,
    stage: str,
    cmd: list[str],
    repo_root: Path,
) -> ActiveStage:
    log_handle = _open_stage_log(run.log_path, stage=stage, gpu=gpu, cmd=cmd)
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=_build_env(repo_root=repo_root, gpu=gpu),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[MATRIX] launch stage={stage} gpu={gpu} run={run.spec.name}")
    print(f"[MATRIX] cmd: {_format_cmd(cmd)}")
    return ActiveStage(run=run, gpu=gpu, stage=stage, cmd=cmd, proc=proc, log_handle=log_handle)


def _train_cmd(run: PreparedMatrixRun) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uwnav_dynamics.train.run_train",
        "-y",
        str(run.yaml_path),
    ]


def _eval_cmd(run: PreparedMatrixRun, cfg: MatrixLauncherConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "uwnav_dynamics.cli.eval",
        "-y",
        str(run.yaml_path),
        "--split",
        cfg.eval_split,
    ]
    if cfg.eval_device is not None:
        cmd += ["--device", cfg.eval_device]
    if cfg.eval_batch_size is not None:
        cmd += ["--batch_size", str(cfg.eval_batch_size)]
    if cfg.plots:
        cmd += [
            "--plots",
            "--plot_fmt",
            cfg.plot_fmt,
            "--dt",
            str(cfg.dt),
            "--x_axis",
            cfg.x_axis,
            "--n_plot_samples",
            str(cfg.n_plot_samples),
        ]
    return cmd


def _write_summary(
    *,
    cfg: MatrixLauncherConfig,
    outcomes: Sequence[RunOutcome],
    summary_csv: Path,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "label",
        "role",
        "gpu",
        "status",
        "failed_stage",
        "returncode",
        "yaml_path",
        "run_dir",
        "eval_dir",
        "log_path",
        "rmse_global",
        "mae_global",
        "rmse_global_masked",
        "mae_global_masked",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in outcomes:
            row = {
                "name": outcome.name,
                "label": outcome.label,
                "role": outcome.role or "",
                "gpu": outcome.gpu,
                "status": outcome.status,
                "failed_stage": outcome.failed_stage or "",
                "returncode": outcome.returncode,
                "yaml_path": str(outcome.yaml_path),
                "run_dir": str(outcome.run_dir),
                "eval_dir": str(outcome.eval_dir),
                "log_path": str(outcome.log_path),
                "rmse_global": "",
                "mae_global": "",
                "rmse_global_masked": "",
                "mae_global_masked": "",
            }
            metrics_path = outcome.eval_dir / "metrics.yaml"
            if cfg.run_eval and outcome.status == "ok" and metrics_path.exists():
                metrics = load_yaml_dict(metrics_path)
                row["rmse_global"] = metrics.get("rmse_global", "")
                row["mae_global"] = metrics.get("mae_global", "")
                row["rmse_global_masked"] = metrics.get("rmse_global_masked", "")
                row["mae_global_masked"] = metrics.get("mae_global_masked", "")
            writer.writerow(row)


def _write_manifest(
    *,
    cfg: MatrixLauncherConfig,
    prepared_runs: Sequence[PreparedMatrixRun],
    path: Path,
) -> None:
    payload = {
        "base_train_yaml": str(cfg.base_train_yaml),
        "work_dir": str(cfg.work_dir),
        "gpus": list(cfg.gpus),
        "max_parallel": cfg.max_parallel,
        "run_eval": cfg.run_eval,
        "eval_split": cfg.eval_split,
        "compare": cfg.compare,
        "compare_metrics": list(cfg.compare_metrics),
        "runs": [
            {
                "name": run.spec.name,
                "label": run.spec.label,
                "role": run.spec.role,
                "yaml_path": str(run.yaml_path),
                "run_dir": str(run.run_dir),
                "eval_dir": str(run.eval_dir),
                "log_path": str(run.log_path),
            }
            for run in prepared_runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _generate_compare_outputs(
    *,
    cfg: MatrixLauncherConfig,
    outcomes: Sequence[RunOutcome],
    compare_root: Path,
) -> None:
    if not cfg.compare:
        return
    successful = [o for o in outcomes if o.status == "ok" and (o.eval_dir / "metrics.yaml").exists()]
    if len(successful) < 2:
        print("[MATRIX] skip compare: fewer than 2 successful evaluated runs")
        return

    from uwnav_dynamics.viz.eval.plot_model_compare import ModelCompareCfg, plot_horizon_compare

    compare_dir = compare_root / f"compare_{cfg.eval_split}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    eval_dirs = [o.eval_dir for o in successful]
    labels = [o.label for o in successful]
    roles = [o.role for o in successful]
    for metric in cfg.compare_metrics:
        plot_horizon_compare(
            eval_dirs=eval_dirs,
            labels=labels,
            roles=roles,
            out_dir=compare_dir,
            cfg=ModelCompareCfg(
                dt_s=cfg.dt,
                use_seconds=(cfg.x_axis == "sec"),
                metric=metric,
                fmt=cfg.plot_fmt,
            ),
        )
        print(f"[MATRIX] wrote compare plots for metric={metric} under: {compare_dir}")


def run_matrix_launcher(cfg: MatrixLauncherConfig, *, repo_root: Path) -> int:
    """按 GPU 池调度整个实验矩阵，并汇总 summary/compare 产物。"""
    resolved_work_dir = _resolve_repo_path(repo_root, cfg.work_dir)
    prepared_runs = prepare_matrix_runs(cfg, repo_root=repo_root)
    _write_manifest(cfg=cfg, prepared_runs=prepared_runs, path=resolved_work_dir / "manifest.yaml")

    pending = list(prepared_runs)
    gpu_pool = list(cfg.gpus[: cfg.max_parallel])
    active: dict[int, ActiveStage] = {}
    outcomes: list[RunOutcome] = []
    stop_launch = False

    while pending or active:
        # 先尽可能把空闲 GPU 填满 train 任务；eval 任务只会在 train 成功后接力同一张卡。
        while pending and gpu_pool and not stop_launch:
            run = pending.pop(0)
            gpu = gpu_pool.pop(0)
            stage = _launch_stage(
                run=run,
                gpu=gpu,
                stage="train",
                cmd=_train_cmd(run),
                repo_root=repo_root,
            )
            active[stage.proc.pid] = stage

        finished_pids: list[int] = []
        launched_after_poll: list[ActiveStage] = []
        for pid, stage in active.items():
            # poll-only 调度避免阻塞在单个长任务上；主循环按 1s 粒度轮询所有 GPU。
            ret = stage.proc.poll()
            if ret is None:
                continue
            stage.log_handle.close()
            finished_pids.append(pid)
            if ret != 0:
                # 失败任务立即释放 GPU，并把失败阶段写入 summary，便于事后定位。
                outcomes.append(
                    RunOutcome(
                        name=stage.run.spec.name,
                        label=stage.run.spec.label,
                        role=stage.run.spec.role,
                        gpu=stage.gpu,
                        status="failed",
                        failed_stage=stage.stage,
                        returncode=int(ret),
                        yaml_path=stage.run.yaml_path,
                        run_dir=stage.run.run_dir,
                        eval_dir=stage.run.eval_dir,
                        log_path=stage.run.log_path,
                    )
                )
                gpu_pool.append(stage.gpu)
                print(
                    f"[MATRIX] stage failed run={stage.run.spec.name} stage={stage.stage} "
                    f"gpu={stage.gpu} rc={ret}"
                )
                if cfg.fail_fast:
                    stop_launch = True
                continue

            if stage.stage == "train" and cfg.run_eval:
                # train 成功后复用同一张 GPU 紧接着做 eval，减少调度和路径解析复杂度。
                launched_after_poll.append(_launch_stage(
                    run=stage.run,
                    gpu=stage.gpu,
                    stage="eval",
                    cmd=_eval_cmd(stage.run, cfg),
                    repo_root=repo_root,
                ))
            else:
                outcomes.append(
                    RunOutcome(
                        name=stage.run.spec.name,
                        label=stage.run.spec.label,
                        role=stage.run.spec.role,
                        gpu=stage.gpu,
                        status="ok",
                        failed_stage=None,
                        returncode=0,
                        yaml_path=stage.run.yaml_path,
                        run_dir=stage.run.run_dir,
                        eval_dir=stage.run.eval_dir,
                        log_path=stage.run.log_path,
                    )
                )
                gpu_pool.append(stage.gpu)
                print(f"[MATRIX] completed run={stage.run.spec.name} gpu={stage.gpu}")

        for pid in finished_pids:
            del active[pid]
        for stage in launched_after_poll:
            active[stage.proc.pid] = stage

        if active:
            time.sleep(1.0)

        if stop_launch and not active:
            break

    if stop_launch and pending:
        # fail-fast 时，尚未启动的任务统一记为 skipped，而不是静默丢弃。
        for run in pending:
            outcomes.append(
                RunOutcome(
                    name=run.spec.name,
                    label=run.spec.label,
                    role=run.spec.role,
                    gpu="",
                    status="skipped",
                    failed_stage=None,
                    returncode=-1,
                    yaml_path=run.yaml_path,
                    run_dir=run.run_dir,
                    eval_dir=run.eval_dir,
                    log_path=run.log_path,
                )
            )

    # summary / compare 放在最后统一生成，确保成功和失败实验都进入总表。
    outcomes.sort(key=lambda item: item.name)
    _write_summary(cfg=cfg, outcomes=outcomes, summary_csv=resolved_work_dir / "summary.csv")
    _generate_compare_outputs(cfg=cfg, outcomes=outcomes, compare_root=resolved_work_dir)

    failed = [o for o in outcomes if o.status == "failed"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    print(f"[MATRIX] summary written to: {resolved_work_dir / 'summary.csv'}")
    if failed:
        print(f"[MATRIX] failed runs: {[o.name for o in failed]}")
    if skipped:
        print(f"[MATRIX] skipped runs: {[o.name for o in skipped]}")
    return 0 if not failed else 1


def main() -> int:
    """多 GPU 实验矩阵命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.train_matrix")
    ap.add_argument("-c", "--config", type=str, required=True, help="matrix launcher yaml")
    args = ap.parse_args()

    repo_root = Path.cwd()
    cfg = load_matrix_launcher_config(Path(args.config))
    return run_matrix_launcher(cfg, repo_root=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
