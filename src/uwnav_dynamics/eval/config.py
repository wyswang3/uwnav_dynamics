"""
模块名称：评估配置装配

模块职责：
负责从 train yaml 与评估 CLI override 中组装数值评估所需的运行时配置，
并保证评估阶段继续复用训练侧 canonical parser，而不是重新解释模型结构。

主要功能：
1. 复用训练配置解析结果，得到与训练一致的模型拓扑配置。
2. 解析 run-scoped split / scaler / eval 输出目录路径。
3. 仅装配数值评估运行时字段，不承载绘图参数。

数据流：
train yaml
    ↓
train.config.load_train_config()
    ↓
RunLayout 解析 run_dir / split / scaler / eval_dir
    ↓
EvalConfig（数值评估运行时） + S1PredictorConfig
    ↓
eval.evaluate 数值评估主流程

依赖模块：
- uwnav_dynamics.train.config
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.models.nets.s1_predictor

备注：
- EvalConfig 只描述数值评估运行时，不包含绘图 orchestration 参数。
- 该模块必须保持 PR1 建立的 train/eval 单一配置真源关系。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig
from uwnav_dynamics.train.config import load_train_config


@dataclass(frozen=True)
class EvalConfig:
    """`evaluate.py` 使用的数值评估运行时配置。"""
    data_dir: Path
    ckpt: Path
    out_dir: Path

    device: str = "cpu"
    batch_size: int = 512

    split_name: str = "test"
    split_indices_path: Path = Path("split_indices.npz")
    x_scaler_path: Path = Path("scalers/x_scaler.npz")
    y_scaler_path: Path = Path("scalers/y_scaler.npz")

    y0_source: str = "x_last_state"
    mode: str = "delta_cumsum"
    long_horizon_fraction: float = 0.4

    save_samples: int = 256


def build_eval_config(
    *,
    train_yaml: str | Path,
    ckpt: str | Path,
    split: str,
    device: str | None,
    batch_size: int | None,
    out_dir: str | Path | None,
    save_samples: int,
) -> Tuple[EvalConfig, S1PredictorConfig]:
    """
    Resolve evaluation inputs from the train yaml and CLI overrides.

    The train yaml remains the single source of truth for:
      - dataset directory
      - artifact layout (split/scaler locations)
      - rollout semantics
      - model topology
    """
    cfg_train = load_train_config(Path(train_yaml))
    layout = RunLayout(out_dir=Path(cfg_train.run.out_dir), variant=str(cfg_train.run.variant))
    resolved_out_dir = Path(out_dir) if out_dir is not None else layout.eval_dir(split)
    if not layout.split_indices_path.exists():
        raise FileNotFoundError(f"Missing split indices: {layout.split_indices_path}")
    if not layout.x_scaler_path.exists() or not layout.y_scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler files under: {layout.scalers_dir} (need x_scaler.npz and y_scaler.npz)"
        )

    cfg_eval = EvalConfig(
        data_dir=Path(cfg_train.data.data_dir),
        ckpt=Path(ckpt),
        out_dir=resolved_out_dir,
        device=device if device is not None else str(cfg_train.run.device),
        batch_size=int(batch_size) if batch_size is not None else int(cfg_train.data.batch_size),
        split_name=split,
        split_indices_path=layout.split_indices_path,
        x_scaler_path=layout.x_scaler_path,
        y_scaler_path=layout.y_scaler_path,
        y0_source=str(cfg_train.rollout.y0_source),
        mode=str(cfg_train.rollout.mode),
        long_horizon_fraction=float(cfg_train.loss.late_horizon_fraction),
        save_samples=int(save_samples),
    )
    return cfg_eval, cfg_train.model
