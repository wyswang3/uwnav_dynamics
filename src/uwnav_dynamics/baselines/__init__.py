"""
模块名称：经典基线工具包

模块职责：
汇总 trivial 与 classical 基线的训练、推理与评估编排能力，
供 baseline runner CLI 与后续对比实验复用。
"""

from .runner import run_baseline_suite

__all__ = ["run_baseline_suite"]
