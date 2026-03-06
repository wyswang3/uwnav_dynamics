"""
模块名称：rollout 数值辅助

模块职责：
提供 rollout 的纯数值运算，
将 `y0` 与增量序列 `dY` 组合成未来状态预测 `y_hat`。

主要功能：
1. 执行 `y_hat = y0 + cumsum(dY)`。
2. 保持 train / eval 共用的 rollout 数学实现。

数据流：
execution layout helper 提供 `y0`
    ↓
`rollout_from_delta(y0, dY)`
    ↓
future state sequence `y_hat`

依赖模块：
- torch

备注：
- 本模块不再负责从 `X` 中提取 `y0`。
- 执行索引解释统一由 `execution_layout.py` 管理。
"""

from __future__ import annotations

import torch


def rollout_from_delta(y0: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    """
    y0: (B, Dout=9)      state at time k
    dY: (B, H, Dout=9)   increments from k->k+1, k+1->k+2, ...
    y_hat: (B, H, Dout)  predicted future states y_{k+1..k+H}
    """
    # cumulative sum over time dimension
    y_hat = y0.unsqueeze(1) + torch.cumsum(dY, dim=1)
    return y_hat
