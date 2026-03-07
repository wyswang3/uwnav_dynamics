"""
模块名称：监督有效性掩码构造

模块职责：
根据稀疏监督原始掩码与 semantic output layout，
构造与监督张量 `Y` 对齐的 `target_mask`，
供训练 loss 与评估指标在运行时直接消费。

主要功能：
1. 构造“全有效”的 dense supervision mask。
2. 兼容 `dvl_mask:(N,H)` 与 `dvl_mask:(N,H,1)` 两种常见 artifact 形状。
3. 将规范化后的 `dvl_mask` 广播到 velocity 语义组，生成 `target_mask:(N,H,D)`。
4. 对历史 artifact 中“mask=true 但 velocity target 为 NaN”的情况提供保守降级保护。
5. 严格校验 mask 形状与 semantic group 的维度合法性。

数据流：
labels.npz["dvl_mask"] + semantic output layout
    ↓
shape normalize
    ↓
target_mask:(N,H,D)
    ↓
optional finite-target guard on velocity group
    ↓
train DataLoader batch / eval metrics

依赖模块：
- numpy
- uwnav_dynamics.models.utils.semantic_output_layout

备注：
- 本模块只负责监督有效性 mask 构造。
- 不负责 rollout 执行索引，不参与 execution layout contract。
- 不负责物理语义推断；语义分组由 semantic output layout 提供。
- 真实 `labels.npz` 可能保存 `dvl_mask:(N,H,1)`；本模块会在消费端压缩 singleton 末轴，
  避免要求重建历史数据集 artifact。
- 对于历史数据集中 `dvl_mask=true` 但 velocity target 仍为 NaN 的情况，
  本模块只提供“把该位置降级为无监督”的保守兼容保护，不会反向修改原始 artifact。
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


def _normalize_dvl_mask_shape(dvl_mask: np.ndarray, *, expected_shape: tuple[int, int]) -> np.ndarray:
    """
    兼容历史/现有两类 mask artifact：
      - (N,H)
      - (N,H,1)

    只允许末轴为 singleton 的三维 mask。
    若出现其他形状，则显式报错，避免静默吞掉错误数据。
    """
    raw = np.asarray(dvl_mask, dtype=bool)
    if raw.shape == expected_shape:
        return raw
    if raw.ndim == 3 and raw.shape[:2] == expected_shape and raw.shape[-1] == 1:
        return raw[..., 0]
    raise ValueError(f"dvl_mask shape mismatch: expect {expected_shape} or {expected_shape + (1,)}, got {raw.shape}")


def build_target_mask_from_dvl_mask(
    dvl_mask: np.ndarray,
    semantic_layout: SemanticOutputLayout,
    *,
    target_shape: Sequence[int],
) -> np.ndarray:
    mask = build_dense_target_mask(target_shape)
    expected = mask.shape[:2]
    raw = _normalize_dvl_mask_shape(dvl_mask, expected_shape=expected)

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


def refine_velocity_target_mask_with_finite_targets(
    target_mask: np.ndarray,
    y: np.ndarray,
    semantic_layout: SemanticOutputLayout,
) -> tuple[np.ndarray, int]:
    """
    对 velocity 语义组执行“有限值保护”。

    背景：
    - 理想情况下，`dvl_mask=true` 的位置应当对应有限的 velocity target。
    - 但历史/迁移后的 processed dataset 里，可能出现：
        `dvl_mask=true`，但 `Y[..., vel]` 仍为 NaN
    - 这类位置不应继续进入训练 loss，否则会把首个 epoch 直接打成 NaN。

    处理策略：
    - 仅对 velocity 语义组做保守降级：
      若某个时刻的任一 velocity 分量非有限，则该时刻的 velocity supervision 全部置为 False。
    - 不修改 acc/gyro 掩码，不改变原始 artifact。

    Returns:
        refined_mask:
            与输入同 shape 的 bool mask。
        downgraded_count:
            被从 True 降级为 False 的 velocity 监督元素数。
    """
    mask = np.asarray(target_mask, dtype=bool)
    y_arr = np.asarray(y)
    if mask.shape != y_arr.shape:
        raise ValueError(
            "target_mask / Y shape mismatch while refining velocity mask: "
            f"target_mask={mask.shape}, Y={y_arr.shape}"
        )

    vel_indices = semantic_layout.group_indices.get("vel")
    if vel_indices is None or len(vel_indices) == 0:
        raise KeyError("semantic layout does not define a non-empty 'vel' group")

    vel_idx_list = list(vel_indices)
    vel_mask_before = mask[:, :, vel_idx_list]
    finite_vel_step = np.all(np.isfinite(y_arr[:, :, vel_idx_list]), axis=-1)
    vel_mask_after = vel_mask_before & finite_vel_step[:, :, None]

    downgraded_count = int(np.count_nonzero(vel_mask_before & ~vel_mask_after))
    if downgraded_count == 0:
        return mask, 0

    refined = mask.copy()
    refined[:, :, vel_idx_list] = vel_mask_after
    return refined, downgraded_count
