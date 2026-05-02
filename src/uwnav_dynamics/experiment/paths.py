"""
模块名称：实验路径快照工具

模块职责：
负责把训练、评估与实验调度阶段产生的路径信息
统一转换成审计友好的相对路径快照，
避免实验产物中混入不可移植的绝对路径。

主要功能：
1. 将绝对路径按指定基准目录转换成相对路径。
2. 递归序列化包含 `Path` 的 dict/list/tuple 结构。
3. 统一解析“相对仓库根目录”与“相对配置文件目录”两类运行时路径。
4. 为 `resolved_train.yaml`、`metrics.yaml`、`summary.csv` 等产物提供统一路径格式。

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
- 对 launcher/config 驱动的实验入口，若路径中显式包含 `..`，默认解释为“相对配置文件目录”；
  否则优先兼容仓库内常见的 repo-root 相对路径（如 `configs/...`、`out/...`）。
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


def infer_repo_root(anchor: str | Path) -> Path:
    """
    从配置文件或源码路径推断当前运行使用的仓库根目录。

    优先查找显式仓库标记；若未命中，则对常见 `configs/`、`src/`
    目录结构做保守回退，最后退化为 anchor 所在目录。
    """
    p = Path(anchor).expanduser().resolve()
    start_dir = p if p.is_dir() else p.parent

    for cand in (start_dir, *start_dir.parents):
        if (cand / "AGENTS.md").exists() or (cand / ".git").exists():
            return cand

    for marker in ("configs", "src"):
        if marker in start_dir.parts:
            idx = len(start_dir.parts) - 1 - list(reversed(start_dir.parts)).index(marker)
            if idx > 0:
                return Path(*start_dir.parts[:idx]).resolve()

    return start_dir


def looks_repo_relative_path(path: str | Path) -> bool:
    """判断一个相对路径是否明显在表达 repo-root 语义。"""
    p = Path(path)
    if p.is_absolute() or len(p.parts) == 0:
        return False
    return p.parts[0] in {
        "apps",
        "configs",
        "data",
        "docs",
        "out",
        "replay_matrix",
        "src",
        "tests",
    }


def resolve_config_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    config_dir: str | Path,
) -> Path:
    """
    统一解析配置文件中的路径字段。

    兼容两类常见来源：
    1. 仓库内手写配置：多使用 repo-root 相对路径。
    2. 由上游 launcher 生成的配置：常写成相对当前配置目录的 `../..` 路径。
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()

    repo_root_path = Path(repo_root)
    config_dir_path = Path(config_dir)
    repo_candidate = (repo_root_path / raw).resolve()
    config_candidate = (config_dir_path / raw).resolve()
    has_parent_hop = any(part == ".." for part in raw.parts)

    if has_parent_hop:
        return config_candidate
    if repo_candidate.exists():
        return repo_candidate
    if config_candidate.exists():
        return config_candidate
    if looks_repo_relative_path(raw):
        return repo_candidate
    return config_candidate


def ensure_path_within_repo_root(
    path: str | Path,
    *,
    repo_root: str | Path,
    field_name: str,
) -> Path:
    """确认运行期路径位于推断出的仓库根目录内。"""
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} resolves outside inferred repo root: path={resolved} repo_root={root}"
        ) from exc
    return resolved


def resolve_repo_output_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    field_name: str,
    config_dir: str | Path | None = None,
) -> Path:
    """
    解析输出路径，并强制它最终位于推断出的仓库根目录内。

    该函数同时兼容：
    - repo-root 相对路径（如 `out/...`）
    - 相对配置文件目录的 `../..` 路径
    - 位于仓库根目录内的绝对路径
    """
    raw = Path(path)
    if config_dir is None:
        config_dir = repo_root
    resolved = resolve_config_path(raw, repo_root=repo_root, config_dir=config_dir)
    return ensure_path_within_repo_root(
        resolved,
        repo_root=repo_root,
        field_name=field_name,
    )


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
