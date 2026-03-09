"""
模块名称：实验目录布局工具

模块职责：
集中定义 train yaml 与运行目录之间的映射关系，
避免 train、eval 与各类 CLI 各自重复拼接路径规则。

主要功能：
1. 读取 train yaml 并解析 `run.out_dir / run.variant`。
2. 用 `RunLayout` 统一暴露 run_dir、scaler、split 与 eval 子目录位置。
3. 作为 CLI 和训练主流程共享的目录契约真源。

数据流：
train yaml
    ↓
load_yaml_dict() / run_layout_from_train_yaml()
    ↓
RunLayout
    ↓
run_dir / split_indices_path / scaler_path / eval_dir

依赖模块：
- yaml
- pathlib

备注：
- 本模块只负责目录语义，不负责创建训练产物或写入 checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """读取 yaml，并要求顶层结构是 mapping。"""
    yaml_path = Path(path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a dict: {yaml_path}")
    return data


@dataclass(frozen=True)
class RunLayout:
    """描述单个训练 variant 的目录契约。"""
    out_dir: Path
    variant: str

    @property
    def run_dir(self) -> Path:
        """训练主目录，集中存放 ckpt、split、scaler 与 eval 产物。"""
        return Path(self.out_dir) / self.variant

    @property
    def split_indices_path(self) -> Path:
        """训练与评估共享的 split artifact 路径。"""
        return self.run_dir / "split_indices.npz"

    @property
    def scalers_dir(self) -> Path:
        """归一化器目录。"""
        return self.run_dir / "scalers"

    @property
    def x_scaler_path(self) -> Path:
        """输入特征 scaler 路径。"""
        return self.scalers_dir / "x_scaler.npz"

    @property
    def y_scaler_path(self) -> Path:
        """监督目标 scaler 路径。"""
        return self.scalers_dir / "y_scaler.npz"

    def eval_dir(self, split_name: str) -> Path:
        """根据 split 名称推导默认评估目录。"""
        return self.run_dir / f"eval_{split_name}"


def run_layout_from_mapping(run_cfg: Mapping[str, Any]) -> RunLayout:
    """从 train yaml 的 `run:` 段构造 `RunLayout`。"""
    return RunLayout(
        out_dir=Path(run_cfg.get("out_dir", "out/ckpts/_unknown")),
        variant=str(run_cfg.get("variant", "default")),
    )


def run_layout_from_train_yaml(path: str | Path) -> RunLayout:
    """供只有 train yaml 路径的调用方快速解析默认运行目录契约。"""
    cfg = load_yaml_dict(path)
    run_cfg = cfg.get("run", {}) or {}
    if not isinstance(run_cfg, dict):
        raise TypeError("run must be a dict")
    return run_layout_from_mapping(run_cfg)
