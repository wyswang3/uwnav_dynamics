from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml

from uwnav_dynamics.train.data import DataConfig
from uwnav_dynamics.train.trainer import TrainConfig
from uwnav_dynamics.models.nets.s1_predictor import S1PredictorConfig


@dataclass(frozen=True)
class LossConfig:
    type: str = "nll_diag"
    logvar_clip_min: float = -10.0
    logvar_clip_max: float = 6.0


@dataclass(frozen=True)
class RolloutConfig:
    y0_source: str = "x_last_state"   # "x_last_state" only for v0
    mode: str = "delta_cumsum"        # "delta_cumsum" only for v0


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int = 0
    device: str = "cuda"
    amp: bool = False
    out_dir: Path = Path("out/ckpts/s1_baseline")


@dataclass(frozen=True)
class TrainYamlConfig:
    run: RunConfig
    data: DataConfig
    model: S1PredictorConfig
    rollout: RolloutConfig
    loss: LossConfig
    train: TrainConfig


def _req(d: Dict[str, Any], key: str, *, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {where}")
    return d[key]


def _as_path(p: Any, *, where: str) -> Path:
    if not isinstance(p, (str, Path)):
        raise TypeError(f"{where} must be a path string, got: {type(p)}")
    return Path(p)


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise TypeError(f"Top-level YAML must be a dict: {path}")
    return d


def build_from_dict(d: Dict[str, Any]) -> TrainYamlConfig:
    # ---------------- run ----------------
    run_d = _req(d, "run", where="root")
    if not isinstance(run_d, dict):
        raise TypeError("run must be a dict")

    run = RunConfig(
        name=str(_req(run_d, "name", where="run")),
        seed=int(run_d.get("seed", 0)),
        device=str(run_d.get("device", "cuda")),
        amp=bool(run_d.get("amp", False)),
        out_dir=_as_path(run_d.get("out_dir", "out/ckpts/s1_baseline"), where="run.out_dir"),
    )

    # ---------------- data ----------------
    data_d = _req(d, "data", where="root")
    if not isinstance(data_d, dict):
        raise TypeError("data must be a dict")

    split_d = data_d.get("split", {})
    if split_d is None:
        split_d = {}
    if not isinstance(split_d, dict):
        raise TypeError("data.split must be a dict")

    data = DataConfig(
        data_dir=_as_path(_req(data_d, "data_dir", where="data"), where="data.data_dir"),
        batch_size=int(data_d.get("batch_size", 256)),
        num_workers=int(data_d.get("num_workers", 4)),
        pin_memory=bool(data_d.get("pin_memory", True)),
        train_ratio=float(split_d.get("train_ratio", 0.7)),
        val_ratio=float(split_d.get("val_ratio", 0.15)),
        seed=run.seed,
    )

    # ---------------- model ----------------
    model_d = _req(d, "model", where="root")
    if not isinstance(model_d, dict):
        raise TypeError("model must be a dict")

    model_name = str(model_d.get("name", "s1_predictor"))
    if model_name != "s1_predictor":
        raise ValueError(f"Unsupported model.name={model_name!r} (v0 only supports 's1_predictor')")

    model = S1PredictorConfig(
        din=int(model_d.get("din", 25)),
        dout=int(model_d.get("dout", 9)),
        pred_len=int(model_d.get("pred_len", 10)),
        rnn_hidden=int(model_d.get("rnn_hidden", 256)),
        rnn_layers=int(model_d.get("rnn_layers", 2)),
        dropout=float(model_d.get("dropout", 0.0)),
    )

    # ---------------- rollout ----------------
    rollout_d = d.get("rollout", {}) or {}
    if not isinstance(rollout_d, dict):
        raise TypeError("rollout must be a dict")

    rollout = RolloutConfig(
        y0_source=str(rollout_d.get("y0_source", "x_last_state")),
        mode=str(rollout_d.get("mode", "delta_cumsum")),
    )
    if rollout.y0_source != "x_last_state":
        raise ValueError(f"Unsupported rollout.y0_source={rollout.y0_source!r} (v0 only supports 'x_last_state')")
    if rollout.mode != "delta_cumsum":
        raise ValueError(f"Unsupported rollout.mode={rollout.mode!r} (v0 only supports 'delta_cumsum')")

    # ---------------- loss ----------------
    loss_d = d.get("loss", {}) or {}
    if not isinstance(loss_d, dict):
        raise TypeError("loss must be a dict")

    loss_type = str(loss_d.get("type", "nll_diag"))
    if loss_type != "nll_diag":
        raise ValueError(f"Unsupported loss.type={loss_type!r} (v0 only supports 'nll_diag')")

    clip = loss_d.get("logvar_clip", [-10.0, 6.0])
    if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
        raise TypeError("loss.logvar_clip must be a list/tuple of [min,max]")

    loss = LossConfig(
        type=loss_type,
        logvar_clip_min=float(clip[0]),
        logvar_clip_max=float(clip[1]),
    )

    # ---------------- optim/train -> TrainConfig ----------------
    optim_d = d.get("optim", {}) or {}
    if not isinstance(optim_d, dict):
        raise TypeError("optim must be a dict")

    train_d = d.get("train", {}) or {}
    if not isinstance(train_d, dict):
        raise TypeError("train must be a dict")

    optim_name = str(optim_d.get("name", "adamw")).lower()
    if optim_name != "adamw":
        raise ValueError(f"Unsupported optim.name={optim_name!r} (v0 only supports 'adamw')")

    train_cfg = TrainConfig(
        epochs=int(train_d.get("epochs", 30)),
        lr=float(optim_d.get("lr", 1e-3)),
        weight_decay=float(optim_d.get("weight_decay", 1e-4)),
        grad_clip=float(optim_d.get("grad_clip", 1.0)),
        device=run.device,
        amp=run.amp,
        out_dir=run.out_dir,
    )

    return TrainYamlConfig(
        run=run,
        data=data,
        model=model,
        rollout=rollout,
        loss=loss,
        train=train_cfg,
    )


def load_train_config(yaml_path: Path) -> TrainYamlConfig:
    d = load_yaml(yaml_path)
    return build_from_dict(d)
