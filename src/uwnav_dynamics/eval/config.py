from __future__ import annotations

"""
Evaluation-side config assembly.

The evaluator reads a train yaml plus a concrete checkpoint path and reconstructs
the data/model/runtime settings needed for a deterministic offline evaluation.

Design rule:
  - Eval reuses the canonical training parser so model topology is guaranteed to
    match training exactly.
  - `EvalConfig` only carries evaluation runtime parameters; it must not become
    a second place that re-explains the model schema.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig
from uwnav_dynamics.train.config import load_train_config


@dataclass(frozen=True)
class EvalConfig:
    """Resolved runtime config consumed by `evaluate.py`."""
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

    save_samples: int = 256

    make_plots: bool = False
    plot_fmt: str = "png"
    dt_s: float = 0.01
    x_axis: str = "sec"
    n_plot_samples: int = 8


def build_eval_config(
    *,
    train_yaml: str | Path,
    ckpt: str | Path,
    split: str,
    device: str | None,
    batch_size: int | None,
    out_dir: str | Path | None,
    save_samples: int,
    make_plots: bool,
    plot_fmt: str,
    dt_s: float,
    x_axis: str,
    n_plot_samples: int,
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
        save_samples=int(save_samples),
        make_plots=bool(make_plots),
        plot_fmt=str(plot_fmt),
        dt_s=float(dt_s),
        x_axis=str(x_axis),
        n_plot_samples=int(n_plot_samples),
    )
    return cfg_eval, cfg_train.model
