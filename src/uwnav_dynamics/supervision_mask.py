"""
模块名称：监督有效性掩码构造

模块职责：
根据稀疏监督原始掩码与 semantic output layout，
构造与监督张量 `Y` 对齐的 `target_mask`，
供训练 loss 与评估指标在运行时直接消费。

主要功能：
1. 构造“全有效”的 dense supervision mask。
2. 将 `dvl_mask:(N,H)` 广播到 velocity 语义组，生成 `target_mask:(N,H,D)`。
3. 严格校验 mask 形状与 semantic group 的维度合法性。

数据流：
labels.npz["dvl_mask"] + semantic output layout
    ↓
target_mask:(N,H,D)
    ↓
train DataLoader batch / eval metrics

依赖模块：
- numpy
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 本模块只负责监督有效性 mask 构造。
- 不负责 rollout 执行索引，不参与 execution layout contract。
- 不负责物理语义推断；语义分组由 semantic output layout 提供。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uwnav_dynamics.models.utils.semantic_output_layout import SemanticOutputLayout


def build_dense_target_mask(target_shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(v) for v in target_shape)
    if len(shape) != 3:
        raise ValueError(f"target_shape must be (N,H,D), got {shape}")
    return np.ones(shape, dtype=bool)


def build_target_mask_from_dvl_mask(
    dvl_mask: np.ndarray,
    semantic_layout: SemanticOutputLayout,
    *,
    target_shape: Sequence[int],
) -> np.ndarray:
    mask = build_dense_target_mask(target_shape)
    raw = np.asarray(dvl_mask, dtype=bool)
    expected = mask.shape[:2]
    if raw.shape != expected:
        raise ValueError(f"dvl_mask shape mismatch: expect {expected}, got {raw.shape}")

    vel_indices = semantic_layout.group_indices.get("vel")
    if vel_indices is None:
        raise KeyError("semantic layout does not define 'vel' group")
    if len(vel_indices) == 0:
        raise ValueError("semantic layout 'vel' group must be non-empty")
    if min(vel_indices) < 0 or max(vel_indices) >= mask.shape[-1]:
        raise ValueError(
            "semantic layout 'vel' group out of range for target dimension: "
            f"indices={tuple(vel_indices)} target_dim={mask.shape[-1]}"
        )

    mask[:, :, list(vel_indices)] = raw[:, :, None]
    return mask
