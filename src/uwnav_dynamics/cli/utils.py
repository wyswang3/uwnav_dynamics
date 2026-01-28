# src/uwnav_dynamics/cli/utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_latest_ckpt(dir_: Path) -> Optional[Path]:
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
    若你的实现不是这样，这里也允许通过 CLI override。
    """
    cfg = load_yaml(train_yaml)
    run = cfg.get("run", {})
    out_dir = Path(run.get("out_dir", "out/ckpts/_unknown"))
    variant = str(run.get("variant", "default"))
    return out_dir, variant


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
