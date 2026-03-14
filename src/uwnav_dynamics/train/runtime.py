"""
模块名称：训练运行时辅助工具

模块职责：
负责训练入口中的运行时拼装工作，包括随机种子、CLI override、
device 协调以及 `resolved_train.yaml` 的审计落盘。

主要功能：
1. 定义训练 CLI 可覆盖的最小字段集合 `TrainCliOverrides`。
2. 将 CLI override 纯函数式应用到 canonical train config。
3. 统一 Python / NumPy / PyTorch / DataLoader worker 的全局 seed 机制。
4. 解析运行时 device，并根据设备特性修正数据加载配置。
5. 把最终实际执行配置序列化为 `resolved_train.yaml`，供复现和审计使用。

数据流：
train yaml config + CLI override
    ↓
apply_train_overrides()
    ↓
runtime reconcile(device / pin_memory)
    ↓
resolved_train.yaml
    ↓
run_train.py / pipeline.py

依赖模块：
- uwnav_dynamics.train.config
- uwnav_dynamics.train.data

备注：
- 本模块不负责训练循环，只处理运行时配置与审计辅助信息。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace, dataclass
from functools import partial
from pathlib import Path
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

from uwnav_dynamics.experiment.paths import relative_path_str, to_snapshot_value
from uwnav_dynamics.train.config import TrainYamlConfig
from uwnav_dynamics.train.data import DataConfig


@dataclass(frozen=True)
class TrainCliOverrides:
    """训练 CLI 允许覆盖的运行时字段集合。"""
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


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    """为 DataLoader worker 设置独立但可复现的随机种子。"""
    worker_seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_dataloader_worker_init_fn(seed: int):
    """返回可直接传给 DataLoader 的 worker seeding 回调。"""
    return partial(_seed_worker, base_seed=int(seed))


def make_torch_generator(seed: int) -> torch.Generator:
    """构造一个与全局 seed 对齐的 `torch.Generator`。"""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """为 Python、NumPy、PyTorch 与确定性后端统一设置随机种子。"""
    seed_i = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_i)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def apply_train_overrides(cfg: TrainYamlConfig, overrides: TrainCliOverrides) -> TrainYamlConfig:
    """
    在不修改原配置对象的前提下应用 CLI override。

    该函数刻意保持纯函数语义，用来清晰区分：
    - 从 yaml 解析得到的 canonical config
    - 由 CLI / pipeline 注入的 runtime-only override
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
    """解析运行设备；若请求 CUDA 但不可用，则显式回退到 CPU。"""
    dev_str = str(device_name).lower()
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        dev_str = "cpu"
    return torch.device(dev_str)


def reconcile_data_config_for_device(data_cfg: DataConfig, device: torch.device) -> tuple[DataConfig, str | None]:
    """按设备类型修正数据配置，并返回需要打印给用户的说明信息。"""
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
    split_strategy: str | None = None,
    x_scaler_path: str | Path | None = None,
    y_scaler_path: str | Path | None = None,
    source_snapshot_path: str | Path | None = None,
    split_sizes: dict[str, int] | None = None,
    dropped_window_count: int | None = None,
    path_root: str | Path | None = None,
) -> Path:
    """
    落盘 override 后的最终训练配置快照。

    `resolved_train.yaml` 的目标是提供审计友好的实验记录：
    - 复现实验时不必再次反推 CLI override
    - 说明训练产物与 split/scaler artifact 的映射关系
    - 为后续开源与科研审查保留一份紧凑快照
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_serializable(cfg, path_root=path_root)
    payload["_meta"] = {
        "schema_version": "train_resolved_v2",
        "source_yaml": _snapshot_path_or_str(source_yaml, path_root=path_root),
    }
    if cli_overrides is not None:
        payload["_meta"]["cli_overrides"] = _to_serializable(cli_overrides, path_root=path_root)
    if requested_device is not None:
        payload["_meta"]["requested_device"] = str(requested_device)
    if runtime_device is not None:
        payload["_meta"]["runtime_device"] = str(runtime_device)
    if run_dir is not None:
        payload["_meta"]["run_dir"] = _snapshot_path_or_str(run_dir, path_root=path_root)
    if split_indices_path is not None:
        payload["_meta"]["split_indices_path"] = _snapshot_path_or_str(split_indices_path, path_root=path_root)
    if split_strategy is not None:
        payload["_meta"]["split_strategy"] = str(split_strategy)
    if x_scaler_path is not None:
        payload["_meta"]["x_scaler_path"] = _snapshot_path_or_str(x_scaler_path, path_root=path_root)
    if y_scaler_path is not None:
        payload["_meta"]["y_scaler_path"] = _snapshot_path_or_str(y_scaler_path, path_root=path_root)
    if source_snapshot_path is not None:
        payload["_meta"]["source_snapshot_path"] = _snapshot_path_or_str(source_snapshot_path, path_root=path_root)
    if split_sizes is not None:
        payload["_meta"]["split_sizes"] = {str(k): int(v) for k, v in split_sizes.items()}
    if dropped_window_count is not None:
        payload["_meta"]["dropped_window_count"] = int(dropped_window_count)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path


def _snapshot_path_or_str(path: str | Path, *, path_root: str | Path | None) -> str:
    p = Path(path)
    if path_root is None:
        return p.as_posix() if not p.is_absolute() else str(p)
    return relative_path_str(p, base_dir=path_root)


def _to_serializable(value: Any, *, path_root: str | Path | None = None) -> Any:
    """递归把 dataclass / Path / tuple 等值转换为 YAML 友好的基础类型。"""
    if isinstance(value, Path):
        if path_root is None:
            return value.as_posix() if not value.is_absolute() else str(value)
        return relative_path_str(value, base_dir=path_root)
    if is_dataclass(value):
        return _to_serializable(asdict(value), path_root=path_root)
    if isinstance(value, dict):
        return to_snapshot_value(
            {str(k): _to_serializable(v, path_root=path_root) for k, v in value.items()},
            base_dir=path_root if path_root is not None else Path.cwd(),
        )
    if isinstance(value, tuple):
        return [_to_serializable(v, path_root=path_root) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v, path_root=path_root) for v in value]
    return value
