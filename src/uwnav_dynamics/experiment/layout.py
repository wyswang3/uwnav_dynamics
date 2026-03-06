from __future__ import annotations

"""
Experiment artifact layout helpers.

This module centralizes how a train yaml maps to runtime directories so train,
eval and CLI wrappers do not each reinvent path concatenation rules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a yaml file and require the top level to be a mapping."""
    yaml_path = Path(path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a dict: {yaml_path}")
    return data


@dataclass(frozen=True)
class RunLayout:
    """
    Materialized directory contract for one training run variant.

    `out_dir` is the experiment root declared in yaml; `variant` is appended once
    to derive the concrete run directory that stores splits, scalers, ckpts and
    evaluation outputs.
    """
    out_dir: Path
    variant: str

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.variant

    @property
    def split_indices_path(self) -> Path:
        return self.run_dir / "split_indices.npz"

    @property
    def scalers_dir(self) -> Path:
        return self.run_dir / "scalers"

    @property
    def x_scaler_path(self) -> Path:
        return self.scalers_dir / "x_scaler.npz"

    @property
    def y_scaler_path(self) -> Path:
        return self.scalers_dir / "y_scaler.npz"

    def eval_dir(self, split_name: str) -> Path:
        return self.run_dir / f"eval_{split_name}"


def run_layout_from_mapping(run_cfg: Mapping[str, Any]) -> RunLayout:
    """Build a `RunLayout` from the `run:` section of a train yaml."""
    return RunLayout(
        out_dir=Path(run_cfg.get("out_dir", "out/ckpts/_unknown")),
        variant=str(run_cfg.get("variant", "default")),
    )


def run_layout_from_train_yaml(path: str | Path) -> RunLayout:
    """Convenience wrapper for callers that only have the train yaml path."""
    cfg = load_yaml_dict(path)
    run_cfg = cfg.get("run", {}) or {}
    if not isinstance(run_cfg, dict):
        raise TypeError("run must be a dict")
    return run_layout_from_mapping(run_cfg)
