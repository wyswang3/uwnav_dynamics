"""
模块名称：CLI 路径与 checkpoint 解析工具

模块职责：
为训练、评估与流水线 CLI 提供统一的 YAML、run 目录、eval 目录与 checkpoint
路径解析能力，避免各入口重复拼接路径规则。

主要功能：
1. 从 train yaml 解析 canonical `run.out_dir / run.variant` 布局。
2. 解析评估目录与 plots 目录路径，不引入创建目录或文件检查副作用。
3. 在 run_dir 或显式 ckpt 路径下选择 best / last / latest checkpoint。

数据流：
train yaml 或 run_dir
    ↓
experiment.layout.RunLayout
    ↓
run_dir / eval_dir / plots_dir / ckpt_path
    ↓
cli/train.py / cli/eval.py / cli/pipeline.py

依赖模块：
- uwnav_dynamics.experiment.layout

备注：
- 本模块只做路径解析与 checkpoint 选择，不负责目录创建、日志与副作用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from uwnav_dynamics.experiment.layout import load_yaml_dict, run_layout_from_train_yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Backward-compatible alias used by existing CLI entry points."""
    return load_yaml_dict(path)


def ensure_dir(p: Path) -> None:
    """Create a directory tree if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def _find_latest_ckpt(dir_: Path) -> Optional[Path]:
    """Fallback search used when canonical best/last checkpoint names are absent."""
    cand = []
    for ext in ("*.pth", "*.pt"):
        cand += list(dir_.rglob(ext))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def resolve_run_out_dir(train_yaml: Path) -> Tuple[Path, str]:
    """
    约定：
      run.out_dir: e.g. out/ckpts/pooltest02_s1_lstm_v0
      run.variant: e.g. B0_baseline
    训练产物通常会落到 out_dir/variant/ 下（如果你 trainer 里这样设计了）。
    `resolved_train.yaml` 会记录一次训练最终实际采用的 run_dir，但 CLI 对外接口
    仍保持“从 train yaml 推导默认 run_dir”的约定不变。
    """
    layout = run_layout_from_train_yaml(train_yaml)
    return layout.out_dir, layout.variant


def resolve_eval_out_dir(
    train_yaml: Path,
    *,
    split: str,
    out_dir_override: Path | None,
) -> Path:
    """解析评估输出目录；若显式指定则直接复用，否则走 RunLayout 默认约定。"""
    if out_dir_override is not None:
        return Path(out_dir_override)
    layout = run_layout_from_train_yaml(train_yaml)
    return layout.eval_dir(split)


def resolve_eval_plots_dir(eval_out_dir: Path) -> Path:
    """解析评估目录下的 plots 子目录路径，不执行创建操作。"""
    return Path(eval_out_dir) / "plots"


def pick_ckpt(ckpt_or_run_dir: Path) -> Path:
    """
    输入可以是：
      - 具体 ckpt 文件路径
      - run_dir（包含 best/last 的目录）
    """
    p = Path(ckpt_or_run_dir)
    if p.is_file():
        return p

    # 优先 best
    best = p / "best.pth"
    if best.exists():
        return best
    best = p / "best.pt"
    if best.exists():
        return best

    # 其次 last
    last = p / "last.pth"
    if last.exists():
        return last
    last = p / "last.pt"
    if last.exists():
        return last

    # 最后：找最新
    latest = _find_latest_ckpt(p)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under: {p}")
    return latest
