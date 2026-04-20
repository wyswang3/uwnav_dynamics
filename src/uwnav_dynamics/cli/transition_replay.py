"""
模块名称：状态求解器 replay CLI

模块职责：
为训练后的网络模型提供正式的“经验型状态求解器 + 长序列 autoregressive replay”
命令行入口，用于验证当前实验数据下的长期递推可行性。

主要功能：
1. 解析 train yaml、checkpoint、split 与 replay 运行参数。
2. 加载训练后模型与 run-scoped split/scaler artifact，构造状态求解器。
3. 基于 `processed dataset + base_csv` 执行长序列 replay 验证。
4. 将数值结果落盘到 `replay_<split>/` 目录，包含 step-wise 长线 artifact，供后续图表与筛选复用。
5. 支持用秒数指定 replay 长度，避免把 `50 steps` 误认为 `50s`。

数据流：
train yaml + ckpt
    ↓
trained transition solver
    ↓
autoregressive replay on split segments
    ↓
metrics.yaml / segment_metrics.csv / step_metrics.csv / pred_samples.npz

依赖模块：
- uwnav_dynamics.cli.utils
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.solver.replay
- uwnav_dynamics.solver.transition_solver

备注：
- 当前阶段该 CLI 主要服务“从轨迹预测器走向经验型状态求解器”的升级验证。
- 它不是闭环控制或仿真器最终入口，但会为后续 `step(u_t, dt)` 集成提供数值证据。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uwnav_dynamics.cli.utils import pick_ckpt
from uwnav_dynamics.experiment.layout import run_layout_from_train_yaml
from uwnav_dynamics.solver.replay import (
    ReplayThresholdSpec,
    load_replay_dataset,
    resolve_replay_step_count,
    run_transition_replay,
    write_replay_outputs,
)
from uwnav_dynamics.solver.transition_solver import load_trained_transition_solver


def main() -> int:
    """长序列 autoregressive replay 验证入口。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.cli.transition_replay")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="train yaml")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path or run_dir")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--min_steps", type=int, default=50)
    ap.add_argument("--min_seconds", type=float, default=None, help="minimum replay segment length in seconds")
    ap.add_argument("--dt", type=float, default=0.01, help="sample period used when converting seconds to steps")
    ap.add_argument("--max_segments", type=int, default=None)
    ap.add_argument("--max_steps_per_segment", type=int, default=None)
    ap.add_argument("--max_seconds_per_segment", type=float, default=None)
    ap.add_argument("--save_samples", type=int, default=8)
    ap.add_argument("--rmse_threshold", type=float, default=0.05)
    ap.add_argument("--abs_error_threshold", type=float, default=0.10)
    args = ap.parse_args()

    train_yaml = Path(args.yaml)
    run_layout = run_layout_from_train_yaml(train_yaml)
    ckpt_path = pick_ckpt(Path(args.ckpt)) if args.ckpt is not None else pick_ckpt(run_layout.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_layout.run_dir / f"replay_{args.split}")

    loaded = load_trained_transition_solver(
        train_yaml=train_yaml,
        ckpt=ckpt_path,
        device=args.device,
    )
    replay_dataset = load_replay_dataset(loaded.cfg_train.data.data_dir)
    min_steps = resolve_replay_step_count(
        steps=int(args.min_steps),
        seconds=args.min_seconds,
        dt_s=float(args.dt),
        default_steps=50,
    )
    max_steps_per_segment = (
        resolve_replay_step_count(
            steps=args.max_steps_per_segment,
            seconds=args.max_seconds_per_segment,
            dt_s=float(args.dt),
            default_steps=min_steps,
        )
        if args.max_steps_per_segment is not None or args.max_seconds_per_segment is not None
        else None
    )
    replay_result = run_transition_replay(
        replay_dataset=replay_dataset,
        split_indices_path=loaded.run_layout.split_indices_path,
        split_name=str(args.split),
        solver=loaded.solver,
        min_steps=int(min_steps),
        max_segments=args.max_segments,
        max_steps_per_segment=max_steps_per_segment,
        save_samples=int(args.save_samples),
        thresholds=ReplayThresholdSpec(
            rmse_threshold=float(args.rmse_threshold),
            abs_error_threshold=float(args.abs_error_threshold),
        ),
    )
    cfg_snapshot = {
        "train_yaml": train_yaml,
        "ckpt": ckpt_path,
        "split": str(args.split),
        "device": str(args.device or loaded.cfg_train.run.device),
        "data_dir": loaded.cfg_train.data.data_dir,
        "split_indices_path": loaded.run_layout.split_indices_path,
        "dt_s": float(args.dt),
        "min_seconds": args.min_seconds,
        "min_steps": int(min_steps),
        "max_segments": args.max_segments,
        "max_seconds_per_segment": args.max_seconds_per_segment,
        "max_steps_per_segment": max_steps_per_segment,
        "save_samples": int(args.save_samples),
        "rmse_threshold": float(args.rmse_threshold),
        "abs_error_threshold": float(args.abs_error_threshold),
    }
    write_replay_outputs(
        out_dir=out_dir,
        replay_result=replay_result,
        cfg_snapshot=cfg_snapshot,
        path_root=Path.cwd(),
    )

    print(f"[REPLAY] ckpt={ckpt_path}")
    print(f"[REPLAY] split={args.split}")
    print(f"[REPLAY] out_dir={out_dir}")
    print(f"[REPLAY] solver_step_semantics={replay_result.metrics['solver_step_semantics']}")
    print(f"[REPLAY] rmse_global={replay_result.metrics['rmse_global']:.6f}")
    print(f"[REPLAY] mae_global={replay_result.metrics['mae_global']:.6f}")
    print(
        "[REPLAY] rmse_threshold_failure_rate="
        f"{replay_result.metrics['long_horizon']['time_to_threshold']['rmse']['failure_rate']:.6f}"
    )
    print(f"[REPLAY] segment_count={replay_result.metrics['segment_count']}")
    print(f"[REPLAY] total_steps={replay_result.metrics['total_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
