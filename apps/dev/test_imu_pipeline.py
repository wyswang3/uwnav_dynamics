#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.io.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.imu_reader import read_imu_csv
from uwnav_dynamics.analysis.imu_stats import analyze_imu, save_imu_stats_txt

# 原始 IMU 绘图（9 轴 + dt）
from uwnav_dynamics.viz.plots.imu_plot import (
    save_imu_raw_figures,
    save_imu_proc_3rows_from_csv,  # 新增：预处理后 3 行图
)

# IMU 预处理总管线
from uwnav_dynamics.preprocess.imu.pipeline import (
    ImuPreprocessConfig,
    run_imu_preprocess_csv,
)


def main() -> int:
    # 1) 指定 dataset yaml（你后续可以改成 argparse）
    yaml_path = Path("configs/dataset/pooltest02.yaml")

    # 2) load dataset spec & resolve imu file
    ds = DatasetSpec.load(yaml_path)
    ds.ensure_paths_exist(require_imu=True)

    imu_path = ds.sensor_path("imu")
    assert imu_path is not None

    imu_kind = ds.selection["imu"].kind or "unknown"

    print(f"[TEST] yaml: {ds.yaml_path}")
    print(f"[TEST] data_root: {ds.resolve_data_root()}")
    print(f"[TEST] raw_base: {ds.raw_base_dir()}")
    print(f"[TEST] imu_file: {imu_path} (kind={imu_kind})")

    # 3) read imu（原始读取，后面用于统计和原始绘图）
    imu = read_imu_csv(imu_path, kind=imu_kind, g0_mps2=9.78)
    print(f"[TEST] imu loaded: N={imu.t_s.size}, time_col={imu.time_col}")

    # 4) stats -> txt（原始数据质量诊断）
    stats = analyze_imu(imu, bias_window_s=20.0, dt_large_threshold_s=0.05)
    stats_txt = save_imu_stats_txt(stats, out_root="out/imu_stats")
    print(f"[TEST] stats saved: {stats_txt}")

    # 5) plots -> png（原始 IMU 波形和 dt 分布）
    p_raw, p_dt = save_imu_raw_figures(
        imu,
        out_root="out/imu_plots",
        use_rel_time=False,
    )
    print("[TEST] raw plots saved:")
    print(f"       - {p_raw}")
    print(f"       - {p_dt}")

    # ------------------------------------------------------------------
    # 6) IMU 预处理 pipeline：transform -> gravity -> bias -> filter
    # ------------------------------------------------------------------
    print("[TEST] run IMU preprocess pipeline ...")

    # 6.1 预处理配置（先用代码默认值，后续可从 configs/preprocess/imu.yaml 加载）
    imu_cfg = ImuPreprocessConfig()

    # 6.2 处理结果输出路径：out/imu_proc/<原始文件名>_proc.csv
    imu_in_path = Path(imu_path)
    out_dir = Path("out/imu_proc")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{imu_in_path.stem}_proc.csv"

    # 6.3 跑完整 pipeline：原始 CSV -> 预处理 CSV + 诊断信息
    diag = run_imu_preprocess_csv(
        in_csv=imu_in_path,
        out_csv=out_csv,
        cfg=imu_cfg,
    )

    print(f"[TEST] imu preprocess done, saved to: {out_csv}")

    # 6.4 打印部分关键诊断信息，便于 sanity check
    print("[TEST] IMU preprocess diag:")
    print(f"       - N            = {diag.n}")
    print(f"       - t0, t1       = {diag.t0:.3f}, {diag.t1:.3f}  (duration={diag.t1 - diag.t0:.3f}s)")
    print(f"       - dt_med       = {diag.dt_med:.6f} s")
    print(f"       - yaw_source   = {diag.yaw_source_used}")
    print(f"       - bias_window  = {diag.bias_window_s:.2f} s, samples={diag.bias_samples}")
    print(f"       - ba_body_mps2 = {diag.ba_body_mps2}")
    print(f"       - bg_body_rad_s= {diag.bg_body_rad_s}")
    print(f"       - std_a_mps2   = {diag.std_a_body_mps2}")
    print(f"       - std_g_rad_s  = {diag.std_g_body_rad_s}")
    print(f"       - despikes     = {diag.despike_counts}")

    # ------------------------------------------------------------------
    # 7) 预处理后的波形绘图：a_body / ω_body / rpy
    # ------------------------------------------------------------------
    p_proc = save_imu_proc_3rows_from_csv(
        proc_csv=out_csv,
        out_root="out/imu_plots_proc",
        use_rel_time=False,
    )
    print("[TEST] proc plot saved:")
    print(f"       - {p_proc}")

    print("[TEST] OK (raw + stats + plots + preprocess + proc_plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
