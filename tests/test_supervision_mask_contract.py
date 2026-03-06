"""
模块名称：监督有效性掩码契约测试

模块职责：
验证 PR5 新增的 supervision mask helper
能够把 `dvl_mask` 与 semantic output layout 正确组合成 `target_mask`。

主要功能：
1. 验证输出 `target_mask` 与 `Y` 形状严格对齐。
2. 验证 acc / gyro 维默认全为 True。
3. 验证 vel 维只由 `dvl_mask` 广播决定。

数据流：
dvl_mask + semantic layout
    ↓
target_mask
    ↓
shape / value contract assertions

依赖模块：
- numpy
- uwnav_dynamics.models.utils.semantic_output_layout
- uwnav_dynamics.supervision_mask
"""

from __future__ import annotations

import numpy as np

from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout
from uwnav_dynamics.supervision_mask import build_target_mask_from_dvl_mask


def test_build_target_mask_uses_dvl_mask_only_for_velocity_group():
    semantic_layout = canonical_semantic_output_layout(9)
    dvl_mask = np.asarray(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    target_mask = build_target_mask_from_dvl_mask(
        dvl_mask,
        semantic_layout,
        target_shape=(2, 3, 9),
    )

    assert target_mask.shape == (2, 3, 9)
    assert target_mask.dtype == np.bool_
    assert np.all(target_mask[:, :, 0:6])
    assert np.array_equal(target_mask[:, :, 6], dvl_mask)
    assert np.array_equal(target_mask[:, :, 7], dvl_mask)
    assert np.array_equal(target_mask[:, :, 8], dvl_mask)
