"""
模块名称：PR3 eval-viz 解耦编排测试

模块职责：
验证正式用户入口 `cli/eval.py` 与 `cli/pipeline.py`
已经承担 eval -> viz 编排职责，并确保 `evaluate.py` 的旧绘图入口显式弃用。

主要功能：
1. 验证 `cli/eval.py` 在无绘图与有绘图两种模式下的命令编排。
2. 验证新接入的 `plot_control_readiness.py` 会进入正式 viz 编排链。
3. 验证绘图阶段失败时返回非零码且保留数值 artifact 的日志语义。
4. 验证 `cli/pipeline.py` 通过 `cli/eval.py` 进行后续编排。

数据流：
临时 train yaml + fake checkpoint
    ↓
cli.utils 路径解析
    ↓
cli/eval.py 或 cli/pipeline.py
    ↓
subprocess command sequence
    ↓
编排契约断言

依赖模块：
- uwnav_dynamics.cli.eval
- uwnav_dynamics.cli.pipeline
- uwnav_dynamics.cli.utils
- uwnav_dynamics.eval.evaluate

备注：
- 本测试不执行真实训练、评估与绘图，只验证编排顺序与失败处理策略。
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

from uwnav_dynamics.cli import eval as cli_eval
from uwnav_dynamics.cli import pipeline as cli_pipeline
from uwnav_dynamics.cli.utils import resolve_eval_out_dir, resolve_eval_plots_dir
from uwnav_dynamics.eval import evaluate


def _write_train_yaml(tmp_path: Path) -> Path:
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B9_eval_viz",
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return train_yaml


def _touch_ckpt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")
    return path


def test_eval_path_helpers_follow_run_layout_contract(tmp_path):
    train_yaml = _write_train_yaml(tmp_path)

    eval_dir = resolve_eval_out_dir(train_yaml, split="test", out_dir_override=None)
    plots_dir = resolve_eval_plots_dir(eval_dir)

    assert eval_dir == tmp_path / "out" / "B9_eval_viz" / "eval_test"
    assert plots_dir == tmp_path / "out" / "B9_eval_viz" / "eval_test" / "plots"


def test_cli_eval_runs_numeric_only_without_plots(tmp_path, monkeypatch, capsys):
    train_yaml = _write_train_yaml(tmp_path)
    ckpt = _touch_ckpt(tmp_path / "manual_best.pth")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        commands.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.eval",
            "-y",
            str(train_yaml),
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
        ],
    )

    ret = cli_eval.main()

    assert ret == 0
    assert len(commands) == 1
    assert commands[0][2] == "uwnav_dynamics.eval.evaluate"
    assert "--plots" not in commands[0]

    out = capsys.readouterr().out
    assert "[CLI][EVAL][NUMERIC]" in out
    assert "numeric artifacts saved under" in out


def test_cli_eval_runs_numeric_then_viz_with_plots(tmp_path, monkeypatch, capsys):
    train_yaml = _write_train_yaml(tmp_path)
    ckpt = _touch_ckpt(tmp_path / "manual_best.pth")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        commands.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.eval",
            "-y",
            str(train_yaml),
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
            "--plots",
            "--plot_fmt",
            "both",
            "--dt",
            "0.02",
            "--x_axis",
            "step",
            "--n_plot_samples",
            "5",
        ],
    )

    ret = cli_eval.main()

    eval_dir = resolve_eval_out_dir(train_yaml, split="test", out_dir_override=None)
    plots_dir = resolve_eval_plots_dir(eval_dir)

    assert ret == 0
    assert [cmd[2] for cmd in commands] == [
        "uwnav_dynamics.eval.evaluate",
        "uwnav_dynamics.viz.eval.plot_horizon_metrics",
        "uwnav_dynamics.viz.eval.plot_horizon_metrics",
        "uwnav_dynamics.viz.eval.plot_control_readiness",
        "uwnav_dynamics.viz.eval.plot_rollout_samples",
        "uwnav_dynamics.viz.eval.plot_pred_vs_observed",
        "uwnav_dynamics.viz.eval.plot_component_residuals",
        "uwnav_dynamics.viz.eval.plot_metrics_dashboard",
    ]
    assert commands[1][commands[1].index("--metric") + 1] == "rmse"
    assert commands[2][commands[2].index("--metric") + 1] == "mae"
    assert commands[1][commands[1].index("--eval_dir") + 1] == str(eval_dir)
    assert commands[1][commands[1].index("--out_dir") + 1] == str(plots_dir)
    assert commands[3][commands[3].index("--out_dir") + 1] == str(plots_dir)
    assert commands[1][commands[1].index("--x") + 1] == "step"
    assert commands[4][commands[4].index("--n") + 1] == "5"
    assert commands[5][commands[5].index("--mode") + 1] == "component"
    assert commands[6][commands[6].index("--out_dir") + 1] == str(plots_dir)
    assert commands[7][commands[7].index("--out_dir") + 1] == str(plots_dir)

    out = capsys.readouterr().out
    assert "start visualization stage under" in out
    assert "plots saved under" in out


def test_cli_eval_keeps_numeric_artifacts_when_plot_stage_fails(tmp_path, monkeypatch, capsys):
    train_yaml = _write_train_yaml(tmp_path)
    ckpt = _touch_ckpt(tmp_path / "manual_best.pth")
    commands: list[list[str]] = []
    return_codes = iter([0, 9])

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        commands.append(list(cmd))
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setattr(cli_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.eval",
            "-y",
            str(train_yaml),
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
            "--plots",
        ],
    )

    ret = cli_eval.main()

    eval_dir = resolve_eval_out_dir(train_yaml, split="test", out_dir_override=None)

    assert ret == 9
    assert [cmd[2] for cmd in commands] == [
        "uwnav_dynamics.eval.evaluate",
        "uwnav_dynamics.viz.eval.plot_horizon_metrics",
    ]

    out = capsys.readouterr().out
    assert "numeric artifacts saved under" in out
    assert f"numeric artifacts were kept under: {eval_dir}" in out


def test_pipeline_invokes_cli_eval_instead_of_eval_module(tmp_path, monkeypatch):
    train_yaml = _write_train_yaml(tmp_path)
    ckpt = _touch_ckpt(tmp_path / "out" / "B9_eval_viz" / "best.pth")
    commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        commands.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.cli.pipeline",
            "-y",
            str(train_yaml),
            "--device",
            "cpu",
            "--eval_split",
            "test",
            "--plots",
        ],
    )

    ret = cli_pipeline.main()

    assert ckpt.exists()
    assert ret == 0
    assert [cmd[2] for cmd in commands] == [
        "uwnav_dynamics.train.run_train",
        "uwnav_dynamics.cli.eval",
    ]
    assert "--plots" in commands[1]
    assert "uwnav_dynamics.eval.evaluate" not in commands[1]


def test_evaluate_rejects_deprecated_plots_flag(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwnav_dynamics.eval.evaluate",
            "--yaml",
            "dummy_train.yaml",
            "--ckpt",
            "dummy_best.pth",
            "--split",
            "test",
            "--plots",
        ],
    )

    ret = evaluate.main()

    assert ret == 2
    err = capsys.readouterr().err
    assert "`--plots` 已弃用" in err
    assert "cli.eval" in err
