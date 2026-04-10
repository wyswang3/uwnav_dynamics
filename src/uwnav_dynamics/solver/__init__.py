"""
模块名称：状态求解器工具包

模块职责：
汇总经验型状态求解器与长序列 replay 验证相关的可复用能力，
作为“训练好的网络模型 -> 可递推状态求解器”之间的桥接层。

主要功能：
1. 提供训练后模型的单步状态求解接口。
2. 提供基于现有实验数据的长序列 autoregressive replay 验证入口。
3. 为后续 controller replay / simulator loop 预留统一接口边界。

数据流：
train yaml + ckpt + split/scaler artifact
    ↓
transition solver
    ↓
single-step prediction / replay validation

依赖模块：
- uwnav_dynamics.solver.transition_solver
- uwnav_dynamics.solver.replay

备注：
- 当前阶段优先服务“经验型状态求解器 + 现有数据可行性验证”。
- 尚未覆盖闭环控制器、MPC 或 RL 环境完整接口。
"""

from .transition_solver import (
    TransitionSolverLoadResult,
    TransitionSolverRolloutResult,
    TrainedTransitionSolver,
    load_trained_transition_solver,
)
from .replay import (
    ReplayDataset,
    ReplayResult,
    ReplaySegment,
    build_replay_segments,
    load_replay_dataset,
    run_transition_replay,
    write_replay_outputs,
)
from .reporting import (
    REPLAY_RANK_METRICS,
    REPLAY_SUMMARY_FIELDS,
    flatten_replay_metrics,
    replay_ranking_protocol,
)

__all__ = [
    "ReplayDataset",
    "ReplayResult",
    "ReplaySegment",
    "REPLAY_RANK_METRICS",
    "REPLAY_SUMMARY_FIELDS",
    "TransitionSolverLoadResult",
    "TransitionSolverRolloutResult",
    "TrainedTransitionSolver",
    "build_replay_segments",
    "flatten_replay_metrics",
    "load_replay_dataset",
    "load_trained_transition_solver",
    "replay_ranking_protocol",
    "run_transition_replay",
    "write_replay_outputs",
]
