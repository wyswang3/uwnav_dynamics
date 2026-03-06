"""
模块名称：执行布局契约

模块职责：
统一管理 rollout 执行路径所依赖的索引契约，
确保 train / eval 都只通过 `cfg_model.y_in_idx` 解释 `x_last_state -> y0`。

主要功能：
1. 校验 `u_in_idx / y_in_idx` 的长度、唯一性与范围合法性。
2. 从输入张量 `X` 的最后一个历史时刻中提取 `y0`。
3. 生成最小 execution layout metadata，供评估 artifact 审计使用。

数据流：
train yaml / resolved model config
    ↓
`cfg_model.y_in_idx`
    ↓
execution layout validation
    ↓
train loss rollout / eval rollout 共同调用
    ↓
`metrics.yaml.layout.execution`

依赖模块：
- torch

备注：
- 本模块只处理执行层索引，不承担任何物理语义分组解释。
- `acc / gyro / vel` 等语义请使用 `semantic_output_layout.py`。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _normalize_indices(indices: Sequence[int] | Iterable[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(i) for i in indices)
    if len(out) == 0:
        raise ValueError(f"{name} must be non-empty")
    return out


def validate_feature_indices(
    indices: Sequence[int] | Iterable[int],
    *,
    upper_bound: int,
    expected_len: int | None = None,
    name: str,
) -> tuple[int, ...]:
    normalized = _normalize_indices(indices, name=name)
    if expected_len is not None and len(normalized) != int(expected_len):
        raise ValueError(f"{name} length mismatch: expect {expected_len}, got {len(normalized)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate indices: {normalized}")
    if min(normalized) < 0 or max(normalized) >= int(upper_bound):
        raise ValueError(f"{name} out of range for upper_bound={upper_bound}: {normalized}")
    return normalized


def validate_execution_layout(
    y_in_idx: Sequence[int] | Iterable[int],
    *,
    din: int,
    dout: int,
) -> tuple[int, ...]:
    return validate_feature_indices(
        y_in_idx,
        upper_bound=int(din),
        expected_len=int(dout),
        name="model.y_in_idx",
    )


def build_execution_layout_metadata(y_in_idx: Sequence[int] | Iterable[int]) -> dict[str, object]:
    return {
        "source": "cfg_model.y_in_idx",
        "y_in_idx": [int(i) for i in y_in_idx],
    }


def extract_y0_from_x_last(x: torch.Tensor, y_in_idx: Sequence[int] | Iterable[int]) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"x must be 3D tensor (B,L,Din), got shape={tuple(x.shape)}")
    idx_tuple = validate_feature_indices(
        y_in_idx,
        upper_bound=int(x.shape[-1]),
        name="model.y_in_idx",
    )
    idx = torch.as_tensor(idx_tuple, device=x.device, dtype=torch.long)
    x_last = x[:, -1, :]
    return x_last.index_select(dim=1, index=idx)
