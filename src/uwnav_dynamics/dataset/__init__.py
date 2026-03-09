"""
模块名称：数据集工具包

模块职责：
汇总数据集切分与归一化相关的基础工具，
供 dataset build、train 和 eval 共享同一套数值与划分语义。
"""

from .split import make_split_indices, save_split_indices, load_split_indices
from .normalize import fit_scaler, transform, save_scaler, load_scaler

__all__ = [
    "make_split_indices",
    "save_split_indices",
    "load_split_indices",
    "fit_scaler",
    "transform",
    "save_scaler",
    "load_scaler",
]
