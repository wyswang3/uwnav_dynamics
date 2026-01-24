# src/uwnav_dynamics/dataset/dataset_spec.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Any

import os
import yaml


SensorName = Literal["imu", "dvl", "pwm", "volt"]


@dataclass(frozen=True)
class SensorSelection:
    """
    A single sensor selection entry in dataset yaml.

    - file: path relative to raw_base_dir, e.g. "imu/min_imu_tb_*.csv"
    - kind: free-form tag for downstream reader routing, e.g. "min_tb"
    """
    file: str = ""
    kind: str = ""

    def is_enabled(self) -> bool:
        return bool(self.file and self.file.strip())


@dataclass(frozen=True)
class ValidWindowSpec:
    """
    Placeholder for future multi-sensor valid time window intersection.
    Kept in spec layer to avoid breaking changes later.
    """
    anchor: str = "imu"
    margin_start_s: float = 0.0
    margin_end_s: float = 0.0


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    dataset_id: str
    data_root: str = ""   # yaml value; may be empty
    raw_dir: str = "raw"  # yaml value; default "raw"


@dataclass(frozen=True)
class DatasetSpec:
    """
    Strongly-typed dataset spec parsed from configs/dataset/*.yaml.

    This object is responsible for:
      - parsing yaml
      - resolving file system paths (data_root/raw_base/sensor files)

    It must NOT:
      - read CSV contents
      - do preprocessing / alignment / plotting
    """
    meta: DatasetMeta
    selection: Dict[SensorName, SensorSelection]
    valid_window: ValidWindowSpec
    yaml_path: Path

    # -------------------------
    # Constructors
    # -------------------------
    @staticmethod
    def load(yaml_path: str | Path) -> "DatasetSpec":
        yaml_path = Path(yaml_path).expanduser().resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Basic schema check
        if not isinstance(cfg, dict) or "dataset" not in cfg:
            raise ValueError(f"Invalid dataset yaml (missing top-level 'dataset'): {yaml_path}")

        ds = cfg["dataset"]
        if not isinstance(ds, dict):
            raise ValueError(f"Invalid dataset yaml ('dataset' must be a mapping): {yaml_path}")

        name = str(ds.get("name", "")).strip()
        dataset_id = str(ds.get("dataset_id", "")).strip()
        if not name:
            raise ValueError(f"dataset.name is empty in: {yaml_path}")
        if not dataset_id:
            raise ValueError(f"dataset.dataset_id is empty in: {yaml_path}")

        meta = DatasetMeta(
            name=name,
            dataset_id=dataset_id,
            data_root=str(ds.get("data_root", "") or ""),
            raw_dir=str(ds.get("raw_dir", "raw") or "raw"),
        )

        sel = ds.get("selection", {}) or {}
        if not isinstance(sel, dict):
            raise ValueError(f"dataset.selection must be a mapping in: {yaml_path}")

        def _parse_one(sensor: SensorName) -> SensorSelection:
            node = sel.get(sensor, {}) or {}
            if not isinstance(node, dict):
                raise ValueError(f"dataset.selection.{sensor} must be a mapping in: {yaml_path}")
            return SensorSelection(
                file=str(node.get("file", "") or ""),
                kind=str(node.get("kind", "") or ""),
            )

        selection: Dict[SensorName, SensorSelection] = {
            "imu": _parse_one("imu"),
            "dvl": _parse_one("dvl"),
            "pwm": _parse_one("pwm"),
            "volt": _parse_one("volt"),
        }

        vw = ds.get("valid_window", {}) or {}
        if not isinstance(vw, dict):
            raise ValueError(f"dataset.valid_window must be a mapping in: {yaml_path}")

        valid_window = ValidWindowSpec(
            anchor=str(vw.get("anchor", "imu") or "imu"),
            margin_start_s=float(vw.get("margin_start_s", 0.0) or 0.0),
            margin_end_s=float(vw.get("margin_end_s", 0.0) or 0.0),
        )

        return DatasetSpec(meta=meta, selection=selection, valid_window=valid_window, yaml_path=yaml_path)

    # -------------------------
    # Path resolution
    # -------------------------
    def resolve_repo_root(self) -> Path:
        """
        Infer repo root from yaml location:
        repo_root/
          configs/dataset/xxx.yaml

        Strategy:
          - walk up parents and find a directory containing 'configs' and 'data' (or '.git').
          - fallback to yaml_dir/../.. if pattern matches.
        """
        p = self.yaml_path
        for parent in [p.parent, *p.parents]:
            if (parent / "configs").exists() and (parent / "data").exists():
                return parent
            if (parent / ".git").exists():
                return parent
        # Fallback: assume yaml is under configs/dataset/
        return p.parent.parent.parent

    def resolve_data_root(self) -> Path:
        """
        Resolve actual data root by precedence:
          1) env UW_DYN_DATA_ROOT
          2) yaml dataset.data_root
          3) repo_root/data
        """
        env = os.getenv("UW_DYN_DATA_ROOT", "").strip()
        if env:
            return Path(env).expanduser().resolve()

        if self.meta.data_root.strip():
            return Path(self.meta.data_root).expanduser().resolve()

        return (self.resolve_repo_root() / "data").resolve()

    def raw_base_dir(self) -> Path:
        """
        Absolute base directory for this dataset's raw data:
          <data_root>/<raw_dir>/<dataset_id>
        """
        return (self.resolve_data_root() / self.meta.raw_dir / self.meta.dataset_id).resolve()

    def sensor_path(self, sensor: SensorName) -> Optional[Path]:
        """
        Returns absolute path for selected sensor file.
        If selection file is empty, returns None.
        """
        sel = self.selection[sensor]
        if not sel.is_enabled():
            return None
        return (self.raw_base_dir() / sel.file).resolve()

    def all_sensor_paths(self) -> Dict[SensorName, Optional[Path]]:
        return {k: self.sensor_path(k) for k in ("imu", "dvl", "pwm", "volt")}

    def ensure_paths_exist(self, *, require_imu: bool = True) -> None:
        """
        Existence checks only. Does not read file content.
        """
        base = self.raw_base_dir()
        if not base.exists():
            raise FileNotFoundError(f"raw_base_dir not found: {base}")

        paths = self.all_sensor_paths()

        if require_imu and paths["imu"] is None:
            raise ValueError("IMU selection is empty but require_imu=True")

        for sensor, p in paths.items():
            if p is None:
                continue
            if not p.exists():
                raise FileNotFoundError(f"{sensor} file not found: {p}")
