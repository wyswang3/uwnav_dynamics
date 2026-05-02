"""
模块名称：多传感器对齐 CLI 入口

模块职责：
解析对齐 yaml，并调用 `aligner.py` 生成统一主时间轴下的 `train_base.csv`，
作为数据集构建的直接上游入口。

主要功能：
1. 读取 `configs/align/*.yaml`。
2. 解析 `AlignConfig`。
3. 调用 `save_training_table_imu_main()` 落盘对齐结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from uwnav_dynamics.experiment.paths import infer_repo_root, resolve_config_path, resolve_repo_output_path
from uwnav_dynamics.preprocess.align.aligner import (
    AlignConfig,
    save_training_table_imu_main,
)

def main() -> int:
    """解析对齐配置并执行训练基础表生成。"""
    parser = argparse.ArgumentParser(
        description="Align IMU / PWM / DVL / Power logs to training base table."
    )
    parser.add_argument(
        "-y", "--yaml",
        type=str,
        required=True,
        help="Alignment config YAML (e.g. configs/align/pooltest02.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.yaml).expanduser().resolve()
    repo_root = infer_repo_root(cfg_path)
    config_dir = cfg_path.parent
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    a = cfg_raw["align"]

    imu_csv = resolve_config_path(Path(str(a["imu_proc_csv"])), repo_root=repo_root, config_dir=config_dir)
    pwm_csv = resolve_config_path(Path(str(a["pwm_csv"])), repo_root=repo_root, config_dir=config_dir)
    dvl_csv = (
        resolve_config_path(Path(str(a.get("dvl_proc_csv"))), repo_root=repo_root, config_dir=config_dir)
        if a.get("dvl_proc_csv") not in (None, "")
        else None
    )
    power_csv = (
        resolve_config_path(Path(str(a.get("power_csv"))), repo_root=repo_root, config_dir=config_dir)
        if a.get("power_csv") not in (None, "")
        else None
    )
    out_csv = resolve_repo_output_path(
        Path(str(a["out_csv"])),
        repo_root=repo_root,
        config_dir=config_dir,
        field_name="align.out_csv",
    )

    align_cfg = AlignConfig(
        dt_main_s=float(a.get("dt_main_s", 0.02)),
        t_margin_s=float(a.get("t_margin_s", 0.0)),
        main_axis_csv=a.get("main_axis_csv"),
        main_time_col=str(a.get("main_time_col", "t_s")),
        dvl_max_dt_s=float(a.get("dvl_max_dt_s", 0.10)),
        power_max_dt_s=float(a.get("power_max_dt_s", 0.25)),
        require_pwm=bool(a.get("require_pwm", True)),
        require_dvl=bool(a.get("require_dvl", False)),
        require_power=bool(a.get("require_power", False)),
    )

    save_training_table_imu_main(
        imu_proc_csv=imu_csv,
        pwm_csv=pwm_csv,
        dvl_proc_csv=dvl_csv,
        power_csv=power_csv,
        out_csv=out_csv,
        cfg=align_cfg,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
