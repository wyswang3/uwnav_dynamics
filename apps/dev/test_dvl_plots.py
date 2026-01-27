#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apps/dev/test_dvl_plots.py

功能：
  - 根据 configs/dataset/pooltest02.yaml 找到本次实验的 DVL 原始 CSV；
  - 调用 read_dvl_csv(...) 读取 DVL 数据为 DvlFrame；
  - 调用 save_dvl_bi_be_vel_2rows(...) 绘制「原始」BI/BE 速度曲线；
  - 调用 DVL 预处理 pipeline，输出处理后 CSV；
  - 调用 save_dvl_proc_figures(...) 绘制「预处理后」的 DVL 图（3 个单独图窗）；
  - 在终端打印一些基本信息和预处理诊断，检查读数与管线是否正常。

使用方式（项目根目录下，有 src/ 和 apps/）：

    PYTHONPATH=src python apps/dev/test_dvl_plots.py
"""

from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.io.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.dvl_reader import read_dvl_csv
from uwnav_dynamics.viz.plots.dvl_plots import (
    save_dvl_bi_be_vel_2rows,   # 原始 BI/BE 2 行子图
    save_dvl_proc_figures,      # 预处理后 3 个单图
)

# 新增：DVL 预处理总管线
from uwnav_dynamics.preprocess.dvl.pipeline import (
    DvlPreprocessConfig,
    run_dvl_preprocess_csv,
)

def main() -> int:
    yaml_path = Path("configs/dataset/pooltest02.yaml")

    ds = DatasetSpec.load(yaml_path)
    ds.ensure_paths_exist(require_imu=False)

    dvl_path = ds.sensor_path("dvl")
    if dvl_path is None:
        raise SystemExit(
            "[TEST] ERROR: dataset.selection.dvl.file 为空，请先在 "
            "configs/dataset/pooltest02.yaml 中配置 dvl 路径。"
        )

    dvl_kind = ds.selection["dvl"].kind or "unknown"

    print(f"[TEST] yaml      : {ds.yaml_path}")
    print(f"[TEST] data_root : {ds.resolve_data_root()}")
    print(f"[TEST] raw_base  : {ds.raw_base_dir()}")
    print(f"[TEST] dvl_file  : {dvl_path} (kind={dvl_kind})")

    # 1) 画原始 BI/BE 速度（你原来的代码）
    dvl = read_dvl_csv(dvl_path, kind=dvl_kind)
    print(f"[TEST] DVL loaded: N={dvl.t_s.size}, time_col={dvl.time_col}")
    out_png_raw = save_dvl_bi_be_vel_2rows(
        dvl,
        out_root="out/dvl_plots",
        use_rel_time=False,
    )
    print("[TEST] DVL BI/BE raw-vel plot saved:")
    print(f"       - {out_png_raw}")

    # 2) 预处理：构造 DvlPreprocessConfig，显式告诉它列名
    out_proc_csv = (
        Path("out/dvl_proc") / f"{dvl_path.stem}_proc.csv"
    )

    dvl_cfg_proc = DvlPreprocessConfig(
        vel_scale=1.0,                      # 原始就是 m/s
        min_quality=0,                      # 先不过滤 quality
        use_status_as_valid=True,

        # BI 体速度三轴
        bi_vel_cols=("Vx_body(m_s)", "Vy_body(m_s)", "Vz_body(m_s)"),

        # BE 垂向速度（ENU Up）
        be_up_col="Vu_enu(m_s)",

        # BD 深度
        depth_col="Depth(m)",
        depth_scale=1.0,

        # 有效性字段（视你 CSV 为主）
        quality_col=None,
        status_col="Valid",                 # 或 "ValidFlag"，看你想用哪个
    )

    print("[TEST] run DVL preprocess pipeline ...")
    diag = run_dvl_preprocess_csv(
        in_csv=dvl_path,
        out_csv=out_proc_csv,
        cfg=dvl_cfg_proc,
    )
    print("[TEST] DVL preprocess done, saved to:", out_proc_csv)
    print("[TEST] DVL preprocess diag (repr):")
    print(f"       {diag!r}")

    # 3) 读 processed CSV，画「处理后」三张图
    combined_png = save_dvl_proc_figures(
        out_proc_csv,
        out_root="out/dvl_plots_proc",
        use_rel_time=False,
    )
    print("[TEST] DVL processed combined plot saved:")
    print(f"       - {combined_png}")

    print("[TEST] OK (dvl read + raw plots + preprocess + proc plots)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
