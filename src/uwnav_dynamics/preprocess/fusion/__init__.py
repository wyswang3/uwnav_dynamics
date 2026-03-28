"""
模块名称：融合预处理工具包

模块职责：
在现有 `align -> build_dataset` 主链路之间新增一层因果状态融合，
把 IMU 高频观测、DVL 稀疏速度观测与姿态代理量整理成统一时间轴上的低噪声状态代理量。

主要功能：
1. 读取对齐后的 `train_base.csv` 与 IMU 预处理结果。
2. 执行 KF / ESKF 风格的速度、偏置与姿态代理量融合。
3. 产出可直接进入 dataset build 的 `train_base_kf_v2.csv`。

数据流：
train_base.csv + imu_proc.csv
    ↓
attitude align
    ↓
KF / ESKF causal fusion
    ↓
train_base_kf_v2.csv

依赖模块：
- uwnav_dynamics.preprocess.fusion.kf_eskf

备注：
- 当前实现优先保证训练可用、因果一致与无 NaN；
- 不把该模块表述成高保真物理真值求解器。
"""

from uwnav_dynamics.preprocess.fusion.kf_eskf import (
    KfEskfFusionConfig,
    build_fused_train_base_from_frames,
    load_fusion_config,
    save_fused_train_base_from_csv,
)

__all__ = [
    "KfEskfFusionConfig",
    "build_fused_train_base_from_frames",
    "load_fusion_config",
    "save_fused_train_base_from_csv",
]
