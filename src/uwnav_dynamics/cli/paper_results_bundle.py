"""
模块名称：论文结果一键打包入口

模块职责：
面向服务器上的论文结果生产流程，
把“原始传感器预处理与观测图 -> 对齐 -> 训练/评估主线 -> 模块对比/路线对比图”
收口成一个配置驱动的一键命令。

主要功能：
1. 从 dataset yaml 触发 IMU / DVL / PWM / Power 的预处理与关键观测图生成。
2. 可选执行 train-base 对齐，再复用既有 `server_pipeline` 跑 fusion / dataset / smoke / train_matrix / replay。
3. 自动收集训练示例图、matrix compare 图与 replay compare 图，并复制到统一 bundle 目录。
4. 支持从 `summary.csv / final_selection.csv` 直接生成论文用的模块对比和路线对比汇总图。
5. 自动汇总单点/稀疏绘图 sidecar warning，便于后续批量审查。
6. 统一写出 bundle manifest，便于服务器侧复现、归档与论文写作引用。

数据流：
paper bundle yaml
    ↓
sensor preprocess + observation figures + align
    ↓
server_pipeline
    ↓
summary / final_selection / compare dirs
    ↓
paper_results_bundle/*

依赖模块：
- yaml
- shutil
- subprocess
- uwnav_dynamics.cli.server_pipeline
- uwnav_dynamics.viz.eval.plot_paper_ablation_summary
- uwnav_dynamics.viz.plots.*

备注：
- 本模块不替代训练、评估与 replay 的业务逻辑，只负责编排与论文导向的产物收口。
- 论文结果图默认追求“简洁、美观、比例得体”，统一复用项目当前主流科研绘图风格。
- 对于单点或稀疏时序导致跳过折线绘制的情况，会统一汇总到 bundle 根目录的 warning summary。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from uwnav_dynamics.analysis.imu_stats import analyze_imu, save_imu_stats_txt
from uwnav_dynamics.cli.server_pipeline import load_server_pipeline_config, run_server_pipeline
from uwnav_dynamics.cli.utils import resolve_run_out_dir
from uwnav_dynamics.experiment.layout import load_yaml_dict
from uwnav_dynamics.experiment.paths import relative_path_str, resolve_config_path
from uwnav_dynamics.io.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.dvl_reader import read_dvl_csv
from uwnav_dynamics.io.readers.imu_reader import read_imu_csv
from uwnav_dynamics.io.readers.power_reader import read_power_csv
from uwnav_dynamics.preprocess.dvl.pipeline import DvlPreprocessConfig, run_dvl_preprocess_csv
from uwnav_dynamics.preprocess.imu.pipeline import ImuPreprocessConfig, run_imu_preprocess_csv
from uwnav_dynamics.preprocess.power.pipeline import PowerPreprocessConfig, build_aux_power_from_dataset
from uwnav_dynamics.viz.eval.plot_paper_ablation_summary import plot_paper_ablation_summary
from uwnav_dynamics.viz.plots.dvl_plots import save_dvl_bi_be_vel_2rows, save_dvl_proc_figures
from uwnav_dynamics.viz.plots.imu_plot import save_imu_proc_3rows_from_csv, save_imu_raw_figures
from uwnav_dynamics.viz.plots.power_plots import (
    resolve_power_time_window,
    save_power_currents_8motors,
    save_power_sync_overview_8motors,
)


@dataclass(frozen=True)
class SensorStageConfig:
    """原始观测预处理与关键观测图阶段配置。"""

    dataset_yaml: Path
    align_yaml: Path | None = None
    use_rel_time: bool = True
    run_imu: bool = True
    run_dvl: bool = True
    run_pwm: bool = True
    run_power: bool = True
    power_window_mode: str = "peak_total_power"
    power_window_s: float = 60.0
    keep_power_current_qa: bool = False


@dataclass(frozen=True)
class SummaryFigureJob:
    """论文汇总图导出任务。"""

    name: str
    csv: str
    mode: str
    labels: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    winner_only: bool = False


@dataclass(frozen=True)
class PaperResultsBundleConfig:
    """论文结果一键打包总配置。"""

    config_path: Path
    work_dir: Path
    plot_fmt: str
    run_server_pipeline: bool
    copy_compare_dirs: bool
    server_pipeline_config: Path | None
    sensor: SensorStageConfig | None
    summary_jobs: tuple[SummaryFigureJob, ...] = ()


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a dict, got {type(value)}")
    return value


def _resolve_repo_path(repo_root: Path, value: Path, *, config_dir: Path | None = None) -> Path:
    if config_dir is None:
        return value if value.is_absolute() else (repo_root / value).resolve()
    return resolve_config_path(value, repo_root=repo_root, config_dir=config_dir)


def _build_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    prefix = str((repo_root / "src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = prefix if prev == "" else (prefix + os.pathsep + prev)
    return env


def _copy_file(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_tree(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _plot_warning_sidecar(path: Path) -> Path:
    return path.with_suffix(".plot_warnings.txt")


def _copy_plot_warning_sidecar(src_plot: Path, dst_plot: Path) -> Path | None:
    return _copy_file(_plot_warning_sidecar(src_plot), _plot_warning_sidecar(dst_plot))


def _run_logged_cmd(*, cmd: list[str], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env(repo_root)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[PAPER_BUNDLE] cmd: " + " ".join(cmd) + "\n")
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


def _parse_plot_warning_file(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        item: dict[str, Any] = {}
        for chunk in line.split("|"):
            token = chunk.strip()
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"finite_points", "total_points"}:
                try:
                    item[key] = int(value)
                except ValueError:
                    item[key] = value
            else:
                item[key] = value
        if item:
            entries.append(item)
    return entries


def _write_plot_warning_summary(bundle_dir: Path) -> tuple[Path, Path, list[str]]:
    warning_files = sorted(bundle_dir.rglob("*.plot_warnings.txt"))
    summary_rows: list[dict[str, Any]] = []
    total_entries = 0
    for path in warning_files:
        rel_path = relative_path_str(path, base_dir=bundle_dir)
        entries = _parse_plot_warning_file(path)
        total_entries += len(entries)
        summary_rows.append(
            {
                "path": rel_path,
                "entry_count": len(entries),
                "entries": entries,
            }
        )

    summary_yaml = bundle_dir / "plot_warning_summary.yaml"
    summary_txt = bundle_dir / "plot_warning_summary.txt"
    payload = {
        "schema_version": "plot_warning_summary_v1",
        "bundle_dir": relative_path_str(bundle_dir, base_dir=bundle_dir),
        "total_warning_files": len(summary_rows),
        "total_warning_entries": total_entries,
        "warning_files": summary_rows,
    }
    with open(summary_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    lines = [
        "# Plot warning summary",
        f"total_warning_files={len(summary_rows)}",
        f"total_warning_entries={total_entries}",
        "",
    ]
    for row in summary_rows:
        lines.append(f"[{row['path']}] entry_count={row['entry_count']}")
        for entry in row["entries"]:
            lines.append(
                "  - "
                + " | ".join(
                    [
                        f"panel={entry.get('panel', '')}",
                        f"series={entry.get('series', '')}",
                        f"reason={entry.get('reason', '')}",
                        f"finite_points={entry.get('finite_points', '')}",
                        f"total_points={entry.get('total_points', '')}",
                    ]
                )
            )
        lines.append("")
    summary_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_yaml, summary_txt, [relative_path_str(p, base_dir=bundle_dir) for p in warning_files]


def _resolve_align_outputs(align_yaml: Path) -> dict[str, Path]:
    raw = load_yaml_dict(align_yaml)
    align_d = _require_mapping(raw.get("align", {}), where="align")
    return {
        "imu_proc_csv": Path(str(align_d.get("imu_proc_csv", ""))).expanduser(),
        "pwm_csv": Path(str(align_d.get("pwm_csv", ""))).expanduser(),
        "dvl_proc_csv": Path(str(align_d.get("dvl_proc_csv", ""))).expanduser(),
        "power_csv": Path(str(align_d.get("power_csv", ""))).expanduser(),
        "out_csv": Path(str(align_d.get("out_csv", ""))).expanduser(),
    }


def _resolve_special_csv(job_csv: str, *, server_work_dir: Path | None) -> Path:
    lowered = job_csv.strip().lower()
    if lowered == "auto_final_selection":
        if server_work_dir is None:
            raise ValueError("auto_final_selection requires server pipeline work_dir")
        return server_work_dir / "final_selection.csv"
    raise ValueError(f"Unsupported special csv token: {job_csv!r}")


def load_paper_results_bundle_config(path: str | Path) -> PaperResultsBundleConfig:
    """从 yaml 加载论文结果一键打包配置。"""
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_dict(config_path)

    launcher = _require_mapping(raw.get("launcher", {}), where="launcher")
    preprocess = _require_mapping(raw.get("preprocess", {}), where="preprocess")
    summary_jobs_raw = raw.get("summary_jobs", [])
    if summary_jobs_raw in (None, ""):
        summary_jobs_raw = []
    if not isinstance(summary_jobs_raw, list):
        raise TypeError("summary_jobs must be a list")

    sensor_cfg = None
    dataset_yaml = preprocess.get("dataset_yaml")
    if dataset_yaml not in (None, ""):
        sensor_cfg = SensorStageConfig(
            dataset_yaml=Path(str(dataset_yaml)),
            align_yaml=None if preprocess.get("align_yaml") in (None, "") else Path(str(preprocess.get("align_yaml"))),
            use_rel_time=bool(preprocess.get("use_rel_time", True)),
            run_imu=bool(preprocess.get("run_imu", True)),
            run_dvl=bool(preprocess.get("run_dvl", True)),
            run_pwm=bool(preprocess.get("run_pwm", True)),
            run_power=bool(preprocess.get("run_power", True)),
            power_window_mode=str(preprocess.get("power_window_mode", "peak_total_power")),
            power_window_s=float(preprocess.get("power_window_s", 60.0)),
            keep_power_current_qa=bool(preprocess.get("keep_power_current_qa", False)),
        )

    jobs: list[SummaryFigureJob] = []
    for idx, item in enumerate(summary_jobs_raw):
        entry = _require_mapping(item, where=f"summary_jobs[{idx}]")
        name = str(entry.get("name", "")).strip()
        csv_path = str(entry.get("csv", "")).strip()
        mode = str(entry.get("mode", "")).strip()
        if not name or not csv_path or not mode:
            raise ValueError(f"summary_jobs[{idx}] requires non-empty name/csv/mode")
        jobs.append(
            SummaryFigureJob(
                name=name,
                csv=csv_path,
                mode=mode,
                labels=tuple(str(v) for v in entry.get("labels", []) or []),
                names=tuple(str(v) for v in entry.get("names", []) or []),
                scopes=tuple(str(v) for v in entry.get("scopes", []) or []),
                winner_only=bool(entry.get("winner_only", False)),
            )
        )

    server_cfg_path = launcher.get("server_pipeline_config")
    return PaperResultsBundleConfig(
        config_path=config_path,
        work_dir=Path(str(launcher.get("work_dir", "out/paper_results_bundle/default"))),
        plot_fmt=str(launcher.get("plot_fmt", "png")),
        run_server_pipeline=bool(launcher.get("run_server_pipeline", True)),
        copy_compare_dirs=bool(launcher.get("copy_compare_dirs", True)),
        server_pipeline_config=None if server_cfg_path in (None, "") else Path(str(server_cfg_path)),
        sensor=sensor_cfg,
        summary_jobs=tuple(jobs),
    )


def _run_sensor_stage(
    *,
    cfg: SensorStageConfig,
    repo_root: Path,
    bundle_dir: Path,
    config_dir: Path,
    logs_dir: Path,
) -> dict[str, str]:
    dataset_yaml = _resolve_repo_path(repo_root, cfg.dataset_yaml, config_dir=config_dir)
    ds = DatasetSpec.load(dataset_yaml)
    ds.ensure_paths_exist(
        require_imu=cfg.run_imu,
        require_dvl=cfg.run_dvl,
        require_pwm=cfg.run_pwm,
        require_volt=cfg.run_power,
    )

    figures_dir = bundle_dir / "figures" / "sensors"
    artifacts: dict[str, str] = {}

    align_outputs: dict[str, Path] = {}
    if cfg.align_yaml is not None:
        align_yaml = _resolve_repo_path(repo_root, cfg.align_yaml, config_dir=config_dir)
        align_outputs = {k: _resolve_repo_path(repo_root, v) for k, v in _resolve_align_outputs(align_yaml).items()}
    else:
        align_yaml = None

    if cfg.run_imu:
        imu_path = ds.sensor_path("imu")
        assert imu_path is not None
        imu = read_imu_csv(imu_path, kind=ds.selection["imu"].kind or "unknown", g0_mps2=9.78)
        stats = analyze_imu(imu, bias_window_s=20.0, dt_large_threshold_s=0.05)
        stats_txt = save_imu_stats_txt(stats, out_root=bundle_dir / "sensor_stats" / "imu")
        raw_fig, dt_fig = save_imu_raw_figures(
            imu,
            out_root=figures_dir / "imu_raw",
            use_rel_time=cfg.use_rel_time,
        )
        proc_csv = align_outputs.get("imu_proc_csv", repo_root / "out" / "imu_proc" / f"{imu_path.stem}_proc.csv")
        proc_csv.parent.mkdir(parents=True, exist_ok=True)
        run_imu_preprocess_csv(in_csv=imu_path, out_csv=proc_csv, cfg=ImuPreprocessConfig())
        proc_fig = save_imu_proc_3rows_from_csv(
            proc_csv=proc_csv,
            out_root=figures_dir / "imu_proc",
            use_rel_time=cfg.use_rel_time,
        )
        artifacts.update(
            {
                "imu_stats_txt": str(stats_txt),
                "imu_raw_png": str(raw_fig),
                "imu_raw_warnings_txt": str(_plot_warning_sidecar(raw_fig)),
                "imu_dt_png": str(dt_fig),
                "imu_dt_warnings_txt": str(_plot_warning_sidecar(dt_fig)),
                "imu_proc_csv": str(proc_csv),
                "imu_proc_png": str(proc_fig),
                "imu_proc_warnings_txt": str(_plot_warning_sidecar(proc_fig)),
            }
        )

    if cfg.run_dvl:
        dvl_path = ds.sensor_path("dvl")
        if dvl_path is not None:
            dvl = read_dvl_csv(dvl_path, kind=ds.selection["dvl"].kind or "unknown")
            raw_png = save_dvl_bi_be_vel_2rows(
                dvl,
                out_root=figures_dir / "dvl_raw",
                use_rel_time=cfg.use_rel_time,
            )
            proc_csv = align_outputs.get("dvl_proc_csv", repo_root / "out" / "dvl_proc" / f"{dvl_path.stem}_proc.csv")
            proc_csv.parent.mkdir(parents=True, exist_ok=True)
            run_dvl_preprocess_csv(
                in_csv=dvl_path,
                out_csv=proc_csv,
                cfg=DvlPreprocessConfig(
                    vel_scale=1.0,
                    min_quality=0,
                    use_status_as_valid=True,
                    bi_vel_cols=("Vx_body(m_s)", "Vy_body(m_s)", "Vz_body(m_s)"),
                    be_up_col="Vu_enu(m_s)",
                    depth_col="Depth(m)",
                    depth_scale=1.0,
                    status_col="Valid",
                ),
            )
            proc_png = save_dvl_proc_figures(
                proc_csv=proc_csv,
                out_root=figures_dir / "dvl_proc",
                use_rel_time=cfg.use_rel_time,
            )
            artifacts.update(
                {
                    "dvl_raw_png": str(raw_png),
                    "dvl_raw_warnings_txt": str(_plot_warning_sidecar(raw_png)),
                    "dvl_proc_csv": str(proc_csv),
                    "dvl_proc_png": str(proc_png),
                    "dvl_proc_warnings_txt": str(_plot_warning_sidecar(proc_png)),
                }
            )

    if cfg.run_pwm:
        pwm_log = logs_dir / "sensor_pwm.log"
        rc = _run_logged_cmd(
            cmd=[
                sys.executable,
                "apps/tools/pwm_preprocess_and_plot.py",
                "--dataset_yaml",
                str(dataset_yaml),
                "--out_dir",
                str((repo_root / "out" / "pwm").resolve()),
            ],
            repo_root=repo_root,
            log_path=pwm_log,
        )
        if rc != 0:
            raise RuntimeError(f"PWM preprocess/plot failed, see log: {pwm_log}")
        pwm_stem = Path(ds.selection["pwm"].file).stem
        pwm_dir = repo_root / "out" / "pwm" / ds.meta.dataset_id
        artifacts.update(
            {
                "pwm_log": str(pwm_log),
                "pwm_csv": str(pwm_dir / f"{pwm_stem}_cmd_aligned.csv"),
                "pwm_png": str(pwm_dir / f"{pwm_stem}_cmd_8ch.png"),
                "pwm_warnings_txt": str(pwm_dir / f"{pwm_stem}_cmd_8ch.plot_warnings.txt"),
            }
        )
        copied = _copy_file(Path(artifacts["pwm_png"]), figures_dir / "pwm" / f"{pwm_stem}_cmd_8ch.png")
        if copied is not None:
            artifacts["pwm_png_bundle"] = str(copied)
            copied_warning = _copy_plot_warning_sidecar(Path(artifacts["pwm_png"]), copied)
            if copied_warning is not None:
                artifacts["pwm_warnings_txt_bundle"] = str(copied_warning)

    if cfg.run_power:
        volt_path = ds.sensor_path("volt")
        assert volt_path is not None
        power = read_power_csv(volt_path)
        aux_csv = build_aux_power_from_dataset(
            ds,
            cfg=PowerPreprocessConfig(out_root=str((repo_root / "out").resolve())),
        )
        win_lo, win_hi = resolve_power_time_window(
            power,
            window_s=float(cfg.power_window_s),
            window_mode=str(cfg.power_window_mode),
        )
        overview_png = save_power_sync_overview_8motors(
            power,
            out_root=figures_dir / "power",
            use_rel_time=cfg.use_rel_time,
            t_start=win_lo,
            t_end=win_hi,
        )
        artifacts.update(
            {
                "power_aux_csv": str(aux_csv),
                "power_overview_png": str(overview_png),
                "power_overview_warnings_txt": str(_plot_warning_sidecar(overview_png)),
                "power_window_start": f"{win_lo:.6f}",
                "power_window_end": f"{win_hi:.6f}",
            }
        )
        if cfg.keep_power_current_qa:
            current_png = save_power_currents_8motors(
                power,
                out_root=figures_dir / "power_current_qa",
                use_rel_time=cfg.use_rel_time,
            )
            artifacts["power_current_qa_png"] = str(current_png)
            artifacts["power_current_qa_warnings_txt"] = str(_plot_warning_sidecar(current_png))

    if align_yaml is not None:
        align_log = logs_dir / "align.log"
        rc = _run_logged_cmd(
            cmd=[sys.executable, "-m", "uwnav_dynamics.preprocess.align.cli_align", "-y", str(align_yaml)],
            repo_root=repo_root,
            log_path=align_log,
        )
        if rc != 0:
            raise RuntimeError(f"align failed, see log: {align_log}")
        artifacts["align_yaml"] = str(align_yaml)
        artifacts["align_log"] = str(align_log)
        artifacts["train_base_csv"] = str(_resolve_repo_path(repo_root, _resolve_align_outputs(align_yaml)["out_csv"]))

    return artifacts


def _collect_training_examples(
    *,
    server_cfg_path: Path | None,
    repo_root: Path,
    bundle_dir: Path,
) -> tuple[list[str], list[str]]:
    if server_cfg_path is None:
        return [], []
    server_cfg = load_server_pipeline_config(server_cfg_path)
    config_dir = server_cfg.config_path.parent
    copied: list[str] = []
    warning_files: list[str] = []
    for smoke in server_cfg.smoke_runs:
        yaml_path = _resolve_repo_path(repo_root, smoke.yaml_path, config_dir=config_dir)
        out_dir, variant = resolve_run_out_dir(yaml_path)
        run_dir = (repo_root / out_dir / variant).resolve() if not out_dir.is_absolute() else (out_dir / variant).resolve()
        dash = run_dir / "train_plots" / "training_dashboard.png"
        loss = run_dir / "train_plots" / "training_loss_curve.png"
        warnings = run_dir / "train_plots" / "training_plots.plot_warnings.txt"
        dash_dst = bundle_dir / "figures" / "training_examples" / f"{smoke.name}_training_dashboard.png"
        loss_dst = bundle_dir / "figures" / "training_examples" / f"{smoke.name}_training_loss_curve.png"
        warnings_dst = bundle_dir / "figures" / "training_examples" / f"{smoke.name}_training_plots.plot_warnings.txt"
        if _copy_file(dash, dash_dst) is not None:
            copied.append(str(dash_dst))
        if _copy_file(loss, loss_dst) is not None:
            copied.append(str(loss_dst))
        if _copy_file(warnings, warnings_dst) is not None:
            warning_files.append(str(warnings_dst))
    return copied, warning_files


def _collect_compare_dirs(
    *,
    paper_manifest_path: Path,
    bundle_dir: Path,
) -> list[str]:
    if not paper_manifest_path.exists():
        return []
    with open(paper_manifest_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    figures_d = _require_mapping(payload.get("figures", {}), where="figures")
    copied: list[str] = []
    for group_name in ("train_compare_dirs", "replay_compare_dirs"):
        values = figures_d.get(group_name, []) or []
        if not isinstance(values, list):
            continue
        for idx, item in enumerate(values):
            src = (paper_manifest_path.parent / str(item)).resolve()
            dst = bundle_dir / "compare_exports" / group_name / f"{idx:02d}_{src.name}"
            copied_dir = _copy_tree(src, dst)
            if copied_dir is not None:
                copied.append(str(copied_dir))
    return copied


def run_paper_results_bundle(cfg: PaperResultsBundleConfig, *, repo_root: Path) -> Path:
    """执行论文结果一键打包流程，并返回 manifest 路径。"""
    config_dir = cfg.config_path.parent
    work_dir = _resolve_repo_path(repo_root, cfg.work_dir, config_dir=config_dir)
    logs_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sensor_artifacts: dict[str, str] = {}
    if cfg.sensor is not None:
        sensor_artifacts = _run_sensor_stage(
            cfg=cfg.sensor,
            repo_root=repo_root,
            bundle_dir=work_dir,
            config_dir=config_dir,
            logs_dir=logs_dir,
        )

    server_work_dir: Path | None = None
    server_phase_status: Path | None = None
    paper_manifest_path: Path | None = None
    if cfg.server_pipeline_config is not None:
        server_cfg_path = _resolve_repo_path(repo_root, cfg.server_pipeline_config, config_dir=config_dir)
        server_cfg = load_server_pipeline_config(server_cfg_path)
        server_work_dir = _resolve_repo_path(repo_root, server_cfg.work_dir, config_dir=server_cfg.config_path.parent)
        if cfg.run_server_pipeline:
            server_phase_status = run_server_pipeline(server_cfg, repo_root=repo_root)
        else:
            server_phase_status = server_work_dir / "phase_status.csv"
        paper_manifest_path = server_work_dir / "paper_artifact_manifest.yaml"
    else:
        server_cfg_path = None

    summary_outputs: list[str] = []
    for job in cfg.summary_jobs:
        if job.csv.strip().lower().startswith("auto_"):
            csv_path = _resolve_special_csv(job.csv, server_work_dir=server_work_dir)
        else:
            csv_path = _resolve_repo_path(repo_root, Path(job.csv), config_dir=config_dir)
        out_dir = work_dir / "figures" / "summaries" / job.name
        out_path = plot_paper_ablation_summary(
            csv_path=csv_path,
            out_dir=out_dir,
            mode=job.mode,
            include_labels=job.labels,
            include_names=job.names,
            include_scopes=job.scopes,
            winner_only=job.winner_only,
            fmt=cfg.plot_fmt,
        )
        summary_outputs.append(str(out_path))

    training_examples, training_warning_files = _collect_training_examples(
        server_cfg_path=server_cfg_path,
        repo_root=repo_root,
        bundle_dir=work_dir,
    )

    compare_exports: list[str] = []
    if cfg.copy_compare_dirs and paper_manifest_path is not None:
        compare_exports = _collect_compare_dirs(
            paper_manifest_path=paper_manifest_path,
            bundle_dir=work_dir,
        )

    if server_work_dir is not None:
        for name in ("phase_status.csv", "final_selection.csv", "paper_artifact_manifest.yaml"):
            _copy_file(server_work_dir / name, work_dir / "upstream_artifacts" / name)

    plot_warning_summary_yaml, plot_warning_summary_txt, plot_warning_files = _write_plot_warning_summary(work_dir)

    sensor_manifest: dict[str, str] = {}
    for key, value in sensor_artifacts.items():
        if key.endswith(("_csv", "_png", "_txt", "_log", "_yaml")):
            sensor_manifest[key] = relative_path_str(Path(value), base_dir=work_dir)
        else:
            sensor_manifest[key] = str(value)

    manifest = {
        "schema_version": "paper_results_bundle_v1",
        "config_path": relative_path_str(cfg.config_path, base_dir=work_dir),
        "work_dir": relative_path_str(work_dir, base_dir=work_dir),
        "sensor_artifacts": sensor_manifest,
        "server_pipeline": {
            "config": None if server_cfg_path is None else relative_path_str(server_cfg_path, base_dir=work_dir),
            "work_dir": None if server_work_dir is None else relative_path_str(server_work_dir, base_dir=work_dir),
            "phase_status": None if server_phase_status is None else relative_path_str(server_phase_status, base_dir=work_dir),
            "paper_artifact_manifest": (
                None if paper_manifest_path is None else relative_path_str(paper_manifest_path, base_dir=work_dir)
            ),
        },
        "summary_outputs": [relative_path_str(Path(p), base_dir=work_dir) for p in summary_outputs],
        "training_examples": [relative_path_str(Path(p), base_dir=work_dir) for p in training_examples],
        "training_warning_files": [relative_path_str(Path(p), base_dir=work_dir) for p in training_warning_files],
        "compare_exports": [relative_path_str(Path(p), base_dir=work_dir) for p in compare_exports],
        "plot_warning_summary": {
            "yaml": relative_path_str(plot_warning_summary_yaml, base_dir=work_dir),
            "txt": relative_path_str(plot_warning_summary_txt, base_dir=work_dir),
            "warning_files": plot_warning_files,
        },
    }
    manifest_path = work_dir / "paper_results_bundle_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return manifest_path


def main() -> int:
    """论文结果一键打包命令行入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.paper_results_bundle")
    ap.add_argument("-c", "--config", type=str, required=True, help="paper results bundle yaml")
    args = ap.parse_args()

    repo_root = Path.cwd()
    cfg = load_paper_results_bundle_config(args.config)
    manifest_path = run_paper_results_bundle(cfg, repo_root=repo_root)
    print(f"[PAPER_BUNDLE] manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
