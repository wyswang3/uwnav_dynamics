#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apps/dev/test_power_plots.py

功能：
  1) 从 dataset yaml 中读取 Volt32/motor 电机功率日志；
  2) 使用统一风格绘制 8 个电机电流曲线（4×2 子图）；
  3) 导出 8 路电机瞬时电功率数据 P = V * I 到 out/aux_power 下：
        - 尽可能完整保留原始时间戳列（MonoNS/EstNS/MonoS/EstS）
        - 额外提供统一的 t_s（秒）
        - 再附加 P0_W ~ P7_W 作为辅助学习数据集。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uwnav_dynamics.io.dataset_spec import DatasetSpec
from uwnav_dynamics.io.readers.power_reader import read_power_csv
from uwnav_dynamics.viz.plots.power_plots import save_power_currents_8motors


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dev: read motor power data, plot currents, "
                    "and export 8-motor power dataset."
    )
    ap.add_argument(
        "-y", "--yaml",
        type=str,
        default="configs/dataset/pooltest02.yaml",
        help="Dataset yaml path (default: configs/dataset/pooltest02.yaml)",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="out",
        help="Output root directory (default: out)",
    )
    ap.add_argument(
        "--rel-time",
        action="store_true",
        help="Use relative time (t - t0) on plots.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    yaml_path = Path(args.yaml)
    print(f"[TEST] dataset yaml : {yaml_path}")

    ds = DatasetSpec.load(yaml_path)
    raw_base = ds.raw_base_dir()
    print(f"[TEST] raw_base_dir  : {raw_base}")

    # 确认 Volt 文件存在
    ds.ensure_paths_exist(
        require_imu=False,
        require_dvl=False,
        require_pwm=False,
        require_volt=True,
    )

    volt_path = ds.sensor_path("volt")
    print(f"[TEST] volt file     : {volt_path}")

    # ------------------ 1) 读取电机功率数据 ------------------
    volt_kwargs = ds.volt_reader_kwargs()
    if not volt_kwargs:
        print("[TEST] Volt sensor is disabled in dataset.selection; nothing to do.")
        return 1

    power = read_power_csv(**volt_kwargs)

    n = len(power)
    if n == 0:
        raise ValueError("[TEST] PowerFrame has no samples.")

    print(f"[TEST] PowerFrame    : N={n}, time_col={power.time_col}")
    print(f"[TEST] t_s range     : {power.t_s[0]:.3f} ~ {power.t_s[-1]:.3f} s")

    # ------------------ 2) 绘制 8 个电机电流 ------------------
    out_root = Path(args.out_root).expanduser().resolve()
    currents_png = save_power_currents_8motors(
        power,
        out_root=out_root / "power_plots",
        use_rel_time=args.rel_time,
    )
    print(f"[TEST] power currents plot saved:\n       - {currents_png}")

    # ------------------ 3) 使用 reader 中的 power_motors ------------------
    # volt_motors / curr_motors / power_motors 都已经在 reader 中准备好了
    power_w = np.asarray(power.power_motors, dtype=float)
    if power_w.shape[1] != 8:
        raise ValueError(
            f"[TEST] Unexpected power_motors shape: {power_w.shape}; expect (N, 8)."
        )

    # ------------------ 4) 导出辅助学习数据集（保留时间戳） ------------------
    # 4.1 先尽可能还原所有原始时间列：MonoNS, EstNS, MonoS, EstS
    #     使用 PowerFrame.time_cols_raw 中的内容
    df_dict: dict[str, object] = {}

    # 保持一个固定的列顺序：先 NS（纳秒），再 S（秒），最后统一 t_s
    for col in ("MonoNS", "EstNS", "MonoS", "EstS"):
        if col in power.time_cols_raw:
            vals = power.time_cols_raw[col]
            if vals.shape[0] != n:
                raise ValueError(
                    f"[TEST] time column '{col}' length mismatch: "
                    f"{vals.shape[0]} vs {n}"
                )
            df_dict[col] = vals

    # 4.2 统一时间轴 t_s（秒）
    t_s = np.asarray(power.t_s, dtype=float).reshape(-1)
    if t_s.shape[0] != n:
        raise ValueError(
            f"[TEST] t_s length mismatch: {t_s.shape[0]} vs {n}"
        )
    df_dict["t_s"] = t_s

    # 4.3 8 路功率列
    for i in range(8):
        df_dict[f"P{i}_W"] = power_w[:, i]

    df_out = pd.DataFrame(df_dict)

    aux_dir = (out_root / "aux_power").resolve()
    aux_dir.mkdir(parents=True, exist_ok=True)

    stem = power.path.stem  # 原始 motor_data_YYYYMMDD_xxxxxx
    out_csv = aux_dir / f"{stem}_power8.csv"

    df_out.to_csv(out_csv, index=False)

    print(f"[TEST] aux power dataset saved:\n       - {out_csv}")
    print("[TEST] columns:", list(df_out.columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
