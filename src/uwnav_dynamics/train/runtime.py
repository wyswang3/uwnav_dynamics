from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from uwnav_dynamics.train.config import TrainYamlConfig
from uwnav_dynamics.train.data import DataConfig


@dataclass(frozen=True)
class TrainCliOverrides:
    data_dir: str | Path | None = None
    device: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    out_dir: str | Path | None = None
    variant: str | None = None
    seed: int | None = None
    amp: bool | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_train_overrides(cfg: TrainYamlConfig, overrides: TrainCliOverrides) -> TrainYamlConfig:
    """
    Apply CLI overrides without mutating the original config object.

    This function intentionally stays pure so the repository always has a clear
    boundary between:
      - canonical config parsed from yaml
      - runtime-only overrides injected by CLI / pipeline
    """
    run_cfg = cfg.run
    data_cfg = cfg.data
    train_cfg = cfg.train

    if overrides.data_dir is not None:
        data_cfg = replace(data_cfg, data_dir=Path(overrides.data_dir))
    if overrides.device is not None:
        device = str(overrides.device)
        run_cfg = replace(run_cfg, device=device)
        train_cfg = replace(train_cfg, device=device)
    if overrides.epochs is not None:
        train_cfg = replace(train_cfg, epochs=int(overrides.epochs))
    if overrides.batch_size is not None:
        data_cfg = replace(data_cfg, batch_size=int(overrides.batch_size))
    if overrides.num_workers is not None:
        data_cfg = replace(data_cfg, num_workers=int(overrides.num_workers))
    if overrides.pin_memory is not None:
        data_cfg = replace(data_cfg, pin_memory=bool(overrides.pin_memory))
    if overrides.out_dir is not None:
        out_dir = Path(overrides.out_dir)
        run_cfg = replace(run_cfg, out_dir=out_dir)
        train_cfg = replace(train_cfg, out_dir=out_dir)
    if overrides.variant is not None:
        run_cfg = replace(run_cfg, variant=str(overrides.variant))
    if overrides.seed is not None:
        seed = int(overrides.seed)
        run_cfg = replace(run_cfg, seed=seed)
        data_cfg = replace(data_cfg, seed=seed)
    if overrides.amp is not None:
        amp = bool(overrides.amp)
        run_cfg = replace(run_cfg, amp=amp)
        train_cfg = replace(train_cfg, amp=amp)

    return replace(cfg, run=run_cfg, data=data_cfg, train=train_cfg)


def resolve_runtime_device(device_name: str) -> torch.device:
    dev_str = str(device_name).lower()
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        dev_str = "cpu"
    return torch.device(dev_str)


def reconcile_data_config_for_device(data_cfg: DataConfig, device: torch.device) -> tuple[DataConfig, str | None]:
    if device.type == "cpu" and data_cfg.pin_memory:
        return replace(data_cfg, pin_memory=False), "[INFO] CPU training: pin_memory=True is useless; auto set to False."
    return data_cfg, None


def save_resolved_train_config(
    path: str | Path,
    cfg: TrainYamlConfig,
    *,
    source_yaml: str | Path,
    cli_overrides: Any | None = None,
    requested_device: str | None = None,
    runtime_device: str | None = None,
    run_dir: str | Path | None = None,
    split_indices_path: str | Path | None = None,
    x_scaler_path: str | Path | None = None,
    y_scaler_path: str | Path | None = None,
) -> Path:
    """
    Persist the exact post-override training config snapshot.

    `resolved_train.yaml` is meant to be an audit-friendly experiment record:
      - reproduce the run later without re-guessing CLI overrides
      - explain how train-time artifacts map to split/scaler files
      - support future open-source and research review with a compact snapshot
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_serializable(cfg)
    payload["_meta"] = {
        "schema_version": "train_resolved_v1",
        "source_yaml": str(Path(source_yaml)),
    }
    if cli_overrides is not None:
        payload["_meta"]["cli_overrides"] = _to_serializable(cli_overrides)
    if requested_device is not None:
        payload["_meta"]["requested_device"] = str(requested_device)
    if runtime_device is not None:
        payload["_meta"]["runtime_device"] = str(runtime_device)
    if run_dir is not None:
        payload["_meta"]["run_dir"] = str(Path(run_dir))
    if split_indices_path is not None:
        payload["_meta"]["split_indices_path"] = str(Path(split_indices_path))
    if x_scaler_path is not None:
        payload["_meta"]["x_scaler_path"] = str(Path(x_scaler_path))
    if y_scaler_path is not None:
        payload["_meta"]["y_scaler_path"] = str(Path(y_scaler_path))
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value
