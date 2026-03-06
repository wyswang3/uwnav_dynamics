from __future__ import annotations

"""
Evaluation-side config assembly.

The evaluator reads a train yaml plus a concrete checkpoint path and reconstructs
the data/model/runtime settings needed for a deterministic offline evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from uwnav_dynamics.experiment.layout import load_yaml_dict, run_layout_from_mapping
from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig


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
    cfg_y = load_yaml_dict(train_yaml)

    data_y = cfg_y.get("data", {}) or {}
    if not isinstance(data_y, dict):
        raise TypeError("YAML key 'data' must be a dict")

    data_dir = Path(data_y.get("data_dir", ""))
    if not str(data_dir):
        raise KeyError("Missing data.data_dir in train yaml")

    rollout = cfg_y.get("rollout", {}) or {}
    if not isinstance(rollout, dict):
        raise TypeError("rollout must be a dict")

    run = cfg_y.get("run", {}) or {}
    if not isinstance(run, dict):
        raise TypeError("run must be a dict")

    layout = run_layout_from_mapping(run)
    resolved_out_dir = Path(out_dir) if out_dir is not None else layout.eval_dir(split)
    if not layout.split_indices_path.exists():
        raise FileNotFoundError(f"Missing split indices: {layout.split_indices_path}")
    if not layout.x_scaler_path.exists() or not layout.y_scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler files under: {layout.scalers_dir} (need x_scaler.npz and y_scaler.npz)"
        )

    cfg_eval = EvalConfig(
        data_dir=data_dir,
        ckpt=Path(ckpt),
        out_dir=resolved_out_dir,
        device=device if device is not None else str(run.get("device", "cpu")),
        batch_size=int(batch_size) if batch_size is not None else int(data_y.get("batch_size", 512)),
        split_name=split,
        split_indices_path=layout.split_indices_path,
        x_scaler_path=layout.x_scaler_path,
        y_scaler_path=layout.y_scaler_path,
        y0_source=str(rollout.get("y0_source", "x_last_state")),
        mode=str(rollout.get("mode", "delta_cumsum")),
        save_samples=int(save_samples),
        make_plots=bool(make_plots),
        plot_fmt=str(plot_fmt),
        dt_s=float(dt_s),
        x_axis=str(x_axis),
        n_plot_samples=int(n_plot_samples),
    )
    return cfg_eval, _build_model_config(cfg_y)


def _build_model_config(cfg_y: dict[str, Any]) -> S1PredictorConfig:
    """
    Rebuild the predictor config expected by the checkpoint loader.

    Keep this function aligned with `train.config.build_from_dict`; otherwise eval
    may instantiate a model whose forward-path semantics differ from training.
    """
    model_y = cfg_y.get("model", {}) or {}
    if not isinstance(model_y, dict):
        raise TypeError("model must be a dict")
    return S1PredictorConfig(
        din=int(model_y["din"]),
        dout=int(model_y["dout"]),
        pred_len=int(model_y["pred_len"]),
        rnn_hidden=int(model_y["rnn_hidden"]),
        rnn_layers=int(model_y["rnn_layers"]),
        dropout=float(model_y.get("dropout", 0.0)),
        u_in_idx=tuple(model_y.get("u_in_idx", list(range(0, 8)))),
        y_in_idx=tuple(model_y.get("y_in_idx", list(range(8, 17)))),
        use_thruster_as_replacement=bool(model_y.get("use_thruster_as_replacement", True)),
        use_hydro_feat=bool(model_y.get("use_hydro_feat", True)),
    )
