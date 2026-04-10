"""
模块名称：replay compare 绘图测试

模块职责：
验证 replay 模型筛选图能够从稳定的 `metrics.yaml + step_metrics.csv`
读盘，并保持 primary / baseline 的排序与输出命名契约。

主要功能：
1. 构造两个最小 replay run 目录。
2. 验证 summary compare 图会把 primary 排在 baseline 之前。
3. 验证命令式导出会写出 replay compare 两张图片。

数据流：
synthetic replay artifacts
    ↓
plot_replay_compare
    ↓
replay_model_compare.png / replay_long_horizon_curves.png
"""

from __future__ import annotations

from pathlib import Path

import yaml

from uwnav_dynamics.viz.eval.plot_replay_compare import (
    ReplayComparePlotCfg,
    build_replay_long_horizon_figure,
    build_replay_summary_figure,
    plot_replay_compare,
)


def _write_replay_run(root: Path, name: str, *, scale: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rmse_global": 0.02 * scale,
        "mae_global": 0.01 * scale,
        "final_step": {"rmse_global_mean": 0.03 * scale},
        "rollout_growth": {"rmse_last_over_first_p95": 1.20 * scale},
        "tail_error": {"abs_p95_global": 0.05 * scale},
        "bias": {"worst_abs_bias": 0.004 * scale},
        "long_horizon": {
            "time_to_threshold": {
                "rmse": {"failure_rate": 0.10 * scale},
            }
        },
    }
    with open(run_dir / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)

    lines = [
        "step,active_segments,value_count,rmse_global,mae_global,abs_p50_global,abs_p95_global,"
        "rmse_survival_rate,abs_survival_rate,rmse_threshold,abs_error_threshold",
    ]
    for step in range(1, 6):
        rmse = 0.01 * scale * step
        mae = 0.008 * scale * step
        abs_p50 = 0.009 * scale * step
        abs_p95 = 0.015 * scale * step
        rmse_survival = max(0.0, 1.0 - 0.05 * scale * (step - 1))
        abs_survival = max(0.0, 1.0 - 0.04 * scale * (step - 1))
        lines.append(
            ",".join(
                [
                    str(step),
                    "4",
                    "12",
                    f"{rmse:.6f}",
                    f"{mae:.6f}",
                    f"{abs_p50:.6f}",
                    f"{abs_p95:.6f}",
                    f"{rmse_survival:.6f}",
                    f"{abs_survival:.6f}",
                    "0.050000",
                    "0.100000",
                ]
            )
        )
    (run_dir / "step_metrics.csv").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def test_replay_summary_figure_orders_primary_before_baseline(tmp_path: Path) -> None:
    primary = _write_replay_run(tmp_path, "primary_run", scale=0.8)
    baseline = _write_replay_run(tmp_path, "baseline_run", scale=1.2)

    fig, axes = build_replay_summary_figure(
        run_dirs=[baseline, primary],
        labels=["baseline_dyn", "step_dyn"],
        roles=["baseline", "primary"],
        cfg=ReplayComparePlotCfg(fmt="png"),
    )

    assert [tick.get_text() for tick in axes[1, 0].get_xticklabels()] == ["step_dyn", "baseline_dyn"]
    assert axes[0, 0].patches[0].get_height() < axes[0, 0].patches[1].get_height()
    fig.clf()


def test_replay_compare_plot_writes_summary_and_curve_pngs(tmp_path: Path) -> None:
    primary = _write_replay_run(tmp_path, "primary_run", scale=0.8)
    baseline = _write_replay_run(tmp_path, "baseline_run", scale=1.2)

    fig, axes = build_replay_long_horizon_figure(
        run_dirs=[primary, baseline],
        labels=["step_dyn", "baseline_dyn"],
        roles=["primary", "baseline"],
        cfg=ReplayComparePlotCfg(fmt="png"),
    )
    assert axes.shape == (2, 2)
    assert axes[1, 0].get_xlabel() == "Replay step (k)"
    fig.clf()

    plot_replay_compare(
        run_dirs=[primary, baseline],
        labels=["step_dyn", "baseline_dyn"],
        roles=["primary", "baseline"],
        out_dir=tmp_path / "plots",
        cfg=ReplayComparePlotCfg(fmt="png"),
    )

    assert (tmp_path / "plots" / "replay_model_compare.png").exists()
    assert (tmp_path / "plots" / "replay_long_horizon_curves.png").exists()
