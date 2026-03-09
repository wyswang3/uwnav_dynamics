"""
模块名称：功率辅助数据预处理

模块职责：
从数据集选择配置中读取电压/功率相关原始文件，
整理出统一时间轴下的 8 路电机功率辅助表，供训练与分析阶段复用。

主要功能：
1. 按 `DatasetSpec` 解析电压/功率传感器输入。
2. 保留原始时间戳并生成统一的 `t_s` 时间列。
3. 将 8 路功率写成标准 CSV，供后续数据集构建使用。

数据流：
`DatasetSpec` 中的 volt 选择
    -> `power_reader` 解析原始功率数据
    -> 统一时间列与 8 路功率表
    -> `aux_power/*.csv`
    -> dataset build / analysis

依赖模块：
1. `uwnav_dynamics.io.readers.power_reader`
2. `uwnav_dynamics.io.dataset_spec`
3. `numpy`
4. `pandas`

备注：
当前流程只做轻量整理，不引入额外重采样或滤波逻辑。
"""

# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from uwnav_dynamics.io.readers.power_reader import read_power_csv
from uwnav_dynamics.io.dataset_spec import DatasetSpec


@dataclass
class PowerPreprocessConfig:
    """
    Power 预处理参数（当前其实不用太多，将来可加 downsample 等）。
    """
    out_root: str = "out"
    use_rel_time: bool = False   # 预留，将来可以做相对时间


def build_aux_power_from_dataset(
    ds: DatasetSpec,
    cfg: Optional[PowerPreprocessConfig] = None,
) -> Path:
    """
    依据数据集配置生成统一格式的 8 路电机功率辅助 CSV。
    """
    if cfg is None:
        cfg = PowerPreprocessConfig()

    volt_kwargs = ds.volt_reader_kwargs()
    if not volt_kwargs:
        raise ValueError("[POWER-PIPE] Volt sensor not configured in dataset.selection.volt")

    power = read_power_csv(**volt_kwargs)

    # 基础时间与原始时间戳
    t_s = power.t_s.astype(float)
    df_out = pd.DataFrame({"t_s": t_s})

    # 尽可能保留原始时间列
    time_cols_raw = power.time_cols_raw or {}
    for name, arr in time_cols_raw.items():
        if arr is None:
            continue
        df_out[name] = np.asarray(arr)

    # 8 路功率
    P = np.asarray(power.power_motors, dtype=float)
    if P.ndim != 2 or P.shape[1] != 8:
        raise ValueError(f"[POWER-PIPE] power_motors shape should be (N,8), got {P.shape}")

    for i in range(8):
        df_out[f"P{i}_W"] = P[:, i]

    out_root = Path(cfg.out_root).expanduser().resolve()
    aux_dir = (out_root / "aux_power").resolve()
    aux_dir.mkdir(parents=True, exist_ok=True)

    stem = power.path.stem  # motor_data_YYYYMMDD_xxxxxx
    out_csv = aux_dir / f"{stem}_power8.csv"
    df_out.to_csv(out_csv, index=False)

    print(f"[POWER-PIPE] Saved aux power dataset: {out_csv}")
    return out_csv
