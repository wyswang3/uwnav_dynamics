"""
模块名称：KF 目标列语义布局测试

模块职责：
验证带 `Kf` 前缀的新目标列名仍能被现有 semantic layout 契约识别，
避免 train / eval 在 target_cols 语义校验阶段报错。

主要功能：
1. 验证 `VelKfX/AccKfX/GyroKfX` 这类列名能映射回 canonical 9 维布局。

依赖模块：
- uwnav_dynamics.models.utils.semantic_output_layout
"""

from __future__ import annotations

from uwnav_dynamics.models.utils.semantic_output_layout import resolve_semantic_output_layout


def test_resolve_semantic_layout_accepts_kf_prefixed_target_cols() -> None:
    layout = resolve_semantic_output_layout(
        dout=9,
        target_cols=(
            "AccKfX_body_mps2",
            "AccKfY_body_mps2",
            "AccKfZ_body_mps2",
            "GyroKfX_body_rad_s",
            "GyroKfY_body_rad_s",
            "GyroKfZ_body_rad_s",
            "VelKfX_body_mps",
            "VelKfY_body_mps",
            "VelKfZ_body_mps",
        ),
    )

    assert layout.component_labels[0] == "acc_x"
    assert layout.component_labels[-1] == "vel_z"
    assert layout.validated_against_target_cols is True
