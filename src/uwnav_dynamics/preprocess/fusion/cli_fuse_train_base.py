"""
模块名称：KF / ESKF 融合基础表 CLI

模块职责：
提供一个最小命令行入口，
把现有 `train_base.csv` 与 `imu_proc.csv` 融合成可直接用于训练的 `train_base_kf_v2.csv`。

主要功能：
1. 读取 `configs/fusion/*.yaml`。
2. 解析 `KfEskfFusionConfig`。
3. 调用融合模块生成并写出新的基础训练表。

数据流：
fusion yaml
    ↓
load_fusion_config()
    ↓
save_fused_train_base_from_csv()
    ↓
train_base_kf_v2.csv

依赖模块：
- uwnav_dynamics.preprocess.fusion.kf_eskf
"""

from __future__ import annotations

import argparse

from uwnav_dynamics.preprocess.fusion.kf_eskf import (
    load_fusion_config,
    save_fused_train_base_from_csv,
)


def main() -> int:
    """解析配置并执行融合基础表生成。"""
    ap = argparse.ArgumentParser(
        description="Fuse aligned train_base.csv with IMU attitude into KF/ESKF train_base_kf_v2.csv."
    )
    ap.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        help="Fusion config YAML (e.g. configs/fusion/pooltest02_kf_eskf_v2.yaml)",
    )
    args = ap.parse_args()

    cfg, base_csv, imu_proc_csv, out_csv = load_fusion_config(args.yaml)
    save_fused_train_base_from_csv(
        base_csv=base_csv,
        imu_proc_csv=imu_proc_csv,
        out_csv=out_csv,
        cfg=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
