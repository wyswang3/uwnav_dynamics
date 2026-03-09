"""
模块名称：实验布局工具包

模块职责：
封装实验运行目录、评估目录和配置解析等布局辅助能力，
为训练、评估与 CLI 入口提供统一目录契约。
"""

from .layout import RunLayout, load_yaml_dict, run_layout_from_mapping, run_layout_from_train_yaml

__all__ = [
    "RunLayout",
    "load_yaml_dict",
    "run_layout_from_mapping",
    "run_layout_from_train_yaml",
]
