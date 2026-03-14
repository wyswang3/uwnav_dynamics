"""
模块名称：实验路径快照工具

模块职责：
负责把训练、评估与实验调度阶段产生的路径信息
统一转换成审计友好的相对路径快照，
避免实验产物中混入不可移植的绝对路径。

主要功能：
1. 将绝对路径按指定基准目录转换成相对路径。
2. 递归序列化包含 `Path` 的 dict/list/tuple 结构。
3. 为 `resolved_train.yaml`、`metrics.yaml`、`summary.csv` 等产物提供统一路径格式。

数据流：
Path / dict / list
    ↓
to_snapshot_value()
    ↓
相对路径字符串
    ↓
yaml / csv / manifest / summary

依赖模块：
- os
- pathlib

备注：
- 已经是相对路径的值会原样保留，避免重复重写用户配置。
- 绝对路径仅在落盘快照时做相对化，不改变运行时真实文件解析逻辑。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def relative_path_str(path: str | Path, *, base_dir: str | Path) -> str:
    """把路径格式化为相对 `base_dir` 的字符串。"""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()

    base = Path(base_dir)
    rel = os.path.relpath(str(p), str(base))
    return Path(rel).as_posix()


def to_snapshot_value(value: Any, *, base_dir: str | Path) -> Any:
    """递归把包含 `Path` 的嵌套结构转换为快照友好的基础类型。"""
    if isinstance(value, Path):
        return relative_path_str(value, base_dir=base_dir)
    if isinstance(value, dict):
        return {str(k): to_snapshot_value(v, base_dir=base_dir) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_snapshot_value(v, base_dir=base_dir) for v in value]
    if isinstance(value, list):
        return [to_snapshot_value(v, base_dir=base_dir) for v in value]
    return value
