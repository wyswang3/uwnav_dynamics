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
class PwmTimebaseSpec:
    """
    PWM timebase mapping spec.

    Map:
      EstS_pwm = ests0_pwm + (t_s - ts0_pwm)

    Notes:
      - method is kept for future extensibility (e.g. mtime-based, cross-corr, etc.)
      - ests0_pwm is the anchor in sensor time system (EstS).
    """
    method: str = "none"
    ests0_pwm: float = 0.0
    ts0_pwm: float = 0.0
    file_offset_s: float = 0.0
    imu_file_tag: str = ""
    pwm_file_tag: str = ""


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
    meta: "DatasetMeta"
    selection: Dict[SensorName, "SensorSelection"]
    valid_window: "ValidWindowSpec"
    yaml_path: Path
    pwm_timebase: Optional[PwmTimebaseSpec] = None

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

        # ---- meta ----
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

        # ---- selection ----
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

        # ---- valid_window ----
        vw = ds.get("valid_window", {}) or {}
        if not isinstance(vw, dict):
            raise ValueError(f"dataset.valid_window must be a mapping in: {yaml_path}")

        valid_window = ValidWindowSpec(
            anchor=str(vw.get("anchor", "imu") or "imu"),
            margin_start_s=float(vw.get("margin_start_s", 0.0) or 0.0),
            margin_end_s=float(vw.get("margin_end_s", 0.0) or 0.0),
        )

        # ---- pwm_timebase (optional) ----
        pwm_timebase: Optional[PwmTimebaseSpec] = None
        ptb = ds.get("pwm_timebase", None)
        if ptb is not None:
            if not isinstance(ptb, dict):
                raise ValueError(f"dataset.pwm_timebase must be a mapping in: {yaml_path}")

            method = str(ptb.get("method", "none") or "none")

            ests0_pwm = ptb.get("ests0_pwm", None)
            if ests0_pwm is None:
                raise ValueError(f"dataset.pwm_timebase.ests0_pwm is required in: {yaml_path}")

            pwm_timebase = PwmTimebaseSpec(
                method=method,
                ests0_pwm=float(ests0_pwm),
                ts0_pwm=float(ptb.get("ts0_pwm", 0.0) or 0.0),
                file_offset_s=float(ptb.get("file_offset_s", 0.0) or 0.0),
                imu_file_tag=str(ptb.get("imu_file_tag", "") or ""),
                pwm_file_tag=str(ptb.get("pwm_file_tag", "") or ""),
            )

        return DatasetSpec(
            meta=meta,
            selection=selection,
            valid_window=valid_window,
            yaml_path=yaml_path,
            pwm_timebase=pwm_timebase,
        )

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

    def ensure_paths_exist(
        self,
        *,
        require_imu: bool = True,
        require_dvl: bool = False,
        require_pwm: bool = False,
        require_volt: bool = False,
    ) -> None:
        """
        Existence checks only. Does not read file content.

        Parameters
        ----------
        require_imu : bool, default True
            If True, raise if IMU selection is empty or file missing.
        require_dvl : bool, default False
            If True, raise if DVL selection is empty or file missing.
        require_pwm : bool, default False
            If True, raise if PWM selection is empty or file missing.
        require_volt : bool, default False
            If True, raise if Volt selection is empty or file missing.
        """
        base = self.raw_base_dir()
        if not base.exists():
            raise FileNotFoundError(f"raw_base_dir not found: {base}")

        paths = self.all_sensor_paths()

        # helper: check requirement for a single sensor
        def _check_required(sensor: SensorName, required: bool) -> None:
            if not required:
                return
            p = paths[sensor]
            if p is None:
                raise ValueError(f"{sensor.upper()} selection is empty but require_{sensor}=True")
            if not p.exists():
                raise FileNotFoundError(f"{sensor} file not found: {p}")

        _check_required("imu", require_imu)
        _check_required("dvl", require_dvl)
        _check_required("pwm", require_pwm)
        _check_required("volt", require_volt)

        # 对于非必需的传感器，如果设置了路径但文件不存在，也给出早期提示
        for sensor, p in paths.items():
            if p is None:
                continue
            if not p.exists():
                raise FileNotFoundError(f"{sensor} file not found: {p}")
    # still in dataset_spec.py, inside DatasetSpec

    def pwm_csv_path(self) -> Optional[Path]:
        return self.sensor_path("pwm")

    def pwm_reader_kwargs(self) -> Dict[str, object]:
        """
        Returns kwargs for pwm_reader.
        """
        p = self.pwm_csv_path()
        if p is None:
            return {}
        # If PWM enabled, we strongly recommend timebase is provided
        if self.pwm_timebase is None:
            raise ValueError(
                "PWM is enabled in dataset.selection but dataset.pwm_timebase is missing. "
                "Please set dataset.pwm_timebase.ests0_pwm (and optionally ts0_pwm)."
            )

        kw: Dict[str, object] = {
            "csv_path": Path(p),
            "timebase_method": self.pwm_timebase.method,
            "ests0_pwm": float(self.pwm_timebase.ests0_pwm),
            "ts0_pwm": float(self.pwm_timebase.ts0_pwm),
            } 

        return kw
     # 在 DatasetSpec 末尾补充

    def volt_csv_path(self) -> Optional[Path]:
        """Convenience wrapper for volt sensor path."""
        return self.sensor_path("volt")

    def volt_reader_kwargs(self) -> Dict[str, object]:
        """
        Returns kwargs for volt_reader (motor voltage/current logger).
        当前只负责提供 csv_path 和 kind 标签，时间对齐后面再设计专门的 spec。
        """
        p = self.volt_csv_path()
        if p is None:
            return {}

        kind = self.selection["volt"].kind or "motor_data"

        return {
            "csv_path": Path(p),
            "kind": kind,
        }


