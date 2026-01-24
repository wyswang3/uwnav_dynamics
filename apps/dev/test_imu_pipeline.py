# apps/dev/test_imu_pipeline.py
from __future__ import annotations

from pathlib import Path

from uwnav_dynamics.dataset.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.imu_reader import read_imu_csv
from uwnav_dynamics.analysis.imu_stats import analyze_imu, save_imu_stats_txt
from uwnav_dynamics.viz.plots.imu_plot import save_imu_raw_figures


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

    # 3) read imu
    imu = read_imu_csv(imu_path, kind=imu_kind, g0_mps2=9.78)
    print(f"[TEST] imu loaded: N={imu.t_s.size}, time_col={imu.time_col}")

    # 4) stats -> txt
    stats = analyze_imu(imu, bias_window_s=20.0, dt_large_threshold_s=0.05)
    stats_txt = save_imu_stats_txt(stats, out_root="out/imu_stats")
    print(f"[TEST] stats saved: {stats_txt}")

    # 5) plots -> png
    p_raw, p_dt = save_imu_raw_figures(imu, out_root="out/imu_plots", use_rel_time=False)
    print(f"[TEST] plots saved:")
    print(f"       - {p_raw}")
    print(f"       - {p_dt}")

    print("[TEST] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
