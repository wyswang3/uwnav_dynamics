"""
模块名称：可视化工具包

模块职责：
汇总评估、原始数据和对比实验所需的绘图模块，
为科研汇报、调试和实验审查生成稳定图片产物。

主要功能：
1. 作为 `uwnav_dynamics.viz.*` 绘图模块的包入口。
2. 在导入 matplotlib 前设置服务器友好的 cache 目录默认值。
3. 保持用户显式设置的 `MPLCONFIGDIR` 不被覆盖。

数据流：
CLI / 测试 / 批处理脚本导入 viz 子模块
    ↓
设置 matplotlib cache 环境变量
    ↓
各绘图模块读取 eval / replay / sensor artifact 并导出图片

依赖模块：
- os
- pathlib

备注：
- 八卡服务器或容器环境中 HOME 配置目录可能不可写；
  若不预先设置 `MPLCONFIGDIR`，matplotlib 会为每个进程创建临时 cache 并输出 warning。
"""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_server_safe_mpl_config_dir() -> None:
    """在用户未显式指定时，为 matplotlib 设置可写 cache 目录。"""
    if os.environ.get("MPLCONFIGDIR") is not None:
        return
    mpl_config_dir = Path("/tmp") / f"uwnav_dynamics_mplconfig_{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


_ensure_server_safe_mpl_config_dir()
