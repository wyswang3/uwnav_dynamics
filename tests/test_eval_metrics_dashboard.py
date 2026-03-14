"""
模块名称：评估指标总览图测试

模块职责：
验证 `plot_metrics_dashboard.py` 能从稳定的 `metrics.yaml`
读取多类评估指标并输出一张 2x2 总览图。

主要功能：
1. 构造最小 `metrics.yaml`。
2. 验证 dashboard figure 的子图数量与标题稳定。
3. 验证命令式导出会写出 `metrics_dashboard.png`。

数据流：
synthetic metrics.yaml
    ↓
plot_metrics_dashboard
    ↓
metrics_dashboard.png

依赖模块：
- yaml
- uwnav_dynamics.viz.eval.plot_metrics_dashboard
"""

from __future__ import annotations

from pathlib import Path

import yaml

from uwnav_dynamics.viz.eval.plot_metrics_dashboard import (
    build_metrics_dashboard_figure,
    plot_metrics_dashboard,
)


def _write_metrics(eval_dir: Path) -> Path:
    payload = {
        "rmse_global": 0.3,
        "mae_global": 0.2,
        "rmse_global_masked": 0.25,
        "mae_global_masked": 0.18,
        "control_readiness": {
            "physical": {
                "dense": {
                    "final_step": {
                        "group_rmse": {"acc": 0.2, "gyro": 0.25, "vel": 0.3},
                        "group_mae": {"acc": 0.16, "gyro": 0.20, "vel": 0.24},
                    },
                    "tail_error": {
                        "abs_p95_global": 0.6,
                        "abs_p99_global": 0.8,
                        "final_step_abs_p95_global": 0.5,
                    },
                    "rollout_growth": {
                        "rmse_last_over_first": 1.3,
                    },
                    "bias": {
                        "worst_component": "vel_x",
                        "worst_abs_bias": 0.11,
                    },
                },
                "masked": {
                    "final_step": {
                        "group_rmse": {"acc": 0.18, "gyro": 0.22, "vel": 0.27},
                        "group_mae": {"acc": 0.14, "gyro": 0.18, "vel": 0.21},
                    },
                    "tail_error": {
                        "abs_p95_global": 0.5,
                        "abs_p99_global": 0.7,
                        "final_step_abs_p95_global": 0.45,
                    },
                    "rollout_growth": {
                        "rmse_last_over_first": 1.15,
                    },
                    "bias": {
                        "worst_component": "vel_y",
                        "worst_abs_bias": 0.09,
                    },
                },
            }
        },
    }
    path = eval_dir / "metrics.yaml"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path


def test_metrics_dashboard_builds_and_writes_png(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_test"
    _write_metrics(eval_dir)

    metrics = yaml.safe_load((eval_dir / "metrics.yaml").read_text(encoding="utf-8"))
    fig, axes = build_metrics_dashboard_figure(metrics)

    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes.ravel()] == [
        "Global Metrics",
        "Group Final-Step RMSE",
        "Tail And Growth",
        "Worst Bias Component",
    ]

    plot_metrics_dashboard(eval_dir, eval_dir / "plots", fmt="png")
    assert (eval_dir / "plots" / "metrics_dashboard.png").exists()
    fig.clf()
