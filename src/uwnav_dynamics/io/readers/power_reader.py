# src/uwnav_dynamics/io/readers/power_reader.py
from __future__ import annotations

"""
uwnav_dynamics.io.readers.power_reader

电机电压/电流采集板（Volt32 等）日志读取模块。

原始 CSV 示例：

  MonoNS,EstNS,MonoS,EstS,
  CH0,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9,CH10,CH11,CH12,CH13,CH14,CH15
  1080042607500,1080042607500,1080.0426075,1080.0426075,1.707V,0.004A,...

约定：
  - 一共 16 个通道：CH0 ... CH15
  - 偶数通道（CH0, CH2, ..., CH14）为电压通道（单位 V，字符串末尾带 "V"）
  - 奇数通道（CH1, CH3, ..., CH15）为电流通道（单位 A，字符串末尾带 "A"）
  - 每两个通道为一组，对应一个电机：
      Motor0: (CH0[V], CH1[A])
      Motor1: (CH2[V], CH3[A])
      ...
      Motor7: (CH14[V], CH15[A])

工程处理策略：
  - 电压：不使用测量值，统一视为 NOMINAL_VOLTAGE_V（默认 12.0 V）；
  - 电流：把原始字符串去掉 "A" 后转为 float，再乘以 CURRENT_GAIN（默认 40 倍）。

输出：
  - PowerFrame:
      path          : Path
      time_col      : 选用的时间列名（"EstS"/"MonoS"/"EstNS"/"MonoNS"）
      t_s           : 时间（秒，float，若原始为 *NS 则做 1e-9 换算）
      time_cols_raw : dict[str, np.ndarray]，保存原始时间戳列（若存在），
                      键为 "MonoNS"/"EstNS"/"MonoS"/"EstS" 的子集
      est_ns        : 若存在 EstNS 列，则为 int64 ns 数组，否则 None
      mono_ns       : 若存在 MonoNS 列，则为 int64 ns 数组，否则 None
      volt_motors   : (N, 8) float32，单位 V，当前策略全部为 NOMINAL_VOLTAGE_V
      curr_motors   : (N, 8) float32，单位 A，已经乘以 CURRENT_GAIN
      power_motors  : (N, 8) float32，单位 W，等于 volt_motors * curr_motors
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# 时间列优先级（与 DVL / IMU 保持一致）
_TIME_CANDIDATES: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS")

# 通道/电机相关常数
N_MOTORS: int = 8
N_CHANNELS: int = 16
NOMINAL_VOLTAGE_V: float = 12.0
CURRENT_GAIN: float = 40.0
# 硬件通道 -> 逻辑电机顺序 的重排：
#   - 1 号电机 ← 原 motor_idx 4 （第 5 组，CH8/CH9）
#   - 5 号电机 ← 原 motor_idx 0 （第 1 组，CH0/CH1）
#   其他保持不变
#
# 逻辑 motor 索引 (0..7) → 原始硬件 motor 索引
#   0 → 4
#   1 → 1
#   2 → 2
#   3 → 3
#   4 → 0
#   5 → 5
#   6 → 6
#   7 → 7
MOTOR_PERM = np.asarray([4, 1, 2, 3, 0, 5, 6, 7], dtype=int)

@dataclass
class PowerFrame:
    """
    电机功率采集数据（Volt32 日志解析结果）。

    Attributes
    ----------
    path : Path
        源 CSV 文件路径。
    time_col : str
        用作主时间轴的列名（"EstS"/"MonoS"/"EstNS"/"MonoNS" 之一）。
    t_s : np.ndarray, shape (N,)
        时间（单位：秒）。若 time_col 为 *NS，则已经除以 1e9。
    time_cols_raw : Dict[str, np.ndarray]
        从 CSV 中原样读取的时间戳列（MonoNS/EstNS/MonoS/EstS 的子集）。
    est_ns : Optional[np.ndarray], shape (N,)
        若存在 EstNS 列，则为 int64 纳秒时间戳，否则 None。
    mono_ns : Optional[np.ndarray], shape (N,)
        若存在 MonoNS 列，则为 int64 纳秒时间戳，否则 None。
    volt_motors : np.ndarray, shape (N, 8)
        8 个电机的电压（单位 V）。当前策略：全部固定为 NOMINAL_VOLTAGE_V。
    curr_motors : np.ndarray, shape (N, 8)
        8 个电机的电流（单位 A）。等于原始数值 * CURRENT_GAIN。
    power_motors : np.ndarray, shape (N, 8)
        8 个电机的输出功率（单位 W）。等于 volt_motors * curr_motors。
    """

    path: Path
    time_col: str
    t_s: np.ndarray
    time_cols_raw: Dict[str, np.ndarray]
    est_ns: Optional[np.ndarray]
    mono_ns: Optional[np.ndarray]
    volt_motors: np.ndarray
    curr_motors: np.ndarray
    power_motors: np.ndarray

    def __len__(self) -> int:
        return int(self.t_s.size)


# ----------------------------------------------------------------------
# 内部工具函数
# ----------------------------------------------------------------------


def _pick_time_s(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    从 _TIME_CANDIDATES 中选择一个时间列并转换为秒。

    返回：
      t_s (float64, 秒), time_col 名称
    """
    for c in _TIME_CANDIDATES:
        if c in df.columns:
            col = df[c].to_numpy()
            # 允许是 int/float/string，统一转 float
            t = col.astype("float64")
            if c.endswith("NS"):
                t = t * 1e-9
            return t, c

    raise ValueError(
        f"[POWER-READER] No valid time column found. "
        f"Tried candidates={_TIME_CANDIDATES}. "
        f"Available columns={list(df.columns)}"
    )


def _collect_time_cols_raw(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    从 MonoNS/EstNS/MonoS/EstS 中收集存在的列，原样保存。

    Returns
    -------
    Dict[str, np.ndarray]
        key 为列名，value 为 np.ndarray（dtype 由 pandas 推断）。
    """
    out: Dict[str, np.ndarray] = {}
    for c in ("MonoNS", "EstNS", "MonoS", "EstS"):
        if c in df.columns:
            out[c] = df[c].to_numpy()
    return out


def _parse_channel_numeric(series: pd.Series, unit_suffix: str) -> np.ndarray:
    """
    将形如 '1.707V' 或 '0.004A' 的字符串转换为 float 数值。

    Parameters
    ----------
    series : pd.Series
        原始列（CHk）。
    unit_suffix : str
        末尾单位字符："V" 或 "A"。

    Returns
    -------
    np.ndarray
        float64 数组。解析失败的项为 NaN。
    """
    s = series.astype(str).str.strip()
    # 去掉末尾单位（如果没有该字符也不会报错）
    s_num = s.str.replace(unit_suffix, "", regex=False)
    vals = pd.to_numeric(s_num, errors="coerce").to_numpy(dtype="float64")
    return vals


# ----------------------------------------------------------------------
# 主入口：读 CSV -> PowerFrame
# ----------------------------------------------------------------------


def read_power_csv(
    csv_path: str | Path,
    *,
    kind: str = "motor_data",
) -> PowerFrame:
    """
    读取电机功率采集 CSV 并解析为 PowerFrame。

    Parameters
    ----------
    csv_path : str or Path
        原始 CSV 路径。
    kind : str, default "motor_data"
        预留参数，目前未使用，仅为了与 DatasetSpec.volt_reader_kwargs
        保持接口一致，方便将来扩展不同格式。

    Returns
    -------
    PowerFrame
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"[POWER-READER] csv_path not found: {path}")

    # 读取 CSV：CH 列通常是 object/str，我们自己解析
    df = pd.read_csv(path)

    # ---- 1) 时间轴（主时间列 + 原始时间列备份）----
    t_s, time_col = _pick_time_s(df)
    time_cols_raw = _collect_time_cols_raw(df)

    est_ns_arr: Optional[np.ndarray] = None
    mono_ns_arr: Optional[np.ndarray] = None
    if "EstNS" in df.columns:
        # 尽量按 int64 解析；解析失败的会是 NaN，这里保持 float 再强转会丢精度，
        # 如果你非常在意，可以保留为原 dtype，这里工程上够用。
        est_ns_arr = df["EstNS"].to_numpy(dtype="int64")
    if "MonoNS" in df.columns:
        mono_ns_arr = df["MonoNS"].to_numpy(dtype="int64")

    N = int(t_s.size)

    # ---- 2) 检查 CH0 ~ CH15 是否存在 ----
    ch_cols = [f"CH{i}" for i in range(N_CHANNELS)]
    missing = [c for c in ch_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "[POWER-READER] Missing expected channel columns: "
            f"{missing}. Available columns={list(df.columns)}"
        )

    # ---- 3) 为 8 个电机构造 V / I 数组（原始硬件顺序）----
    volt_motors = np.full((N, N_MOTORS), NOMINAL_VOLTAGE_V, dtype="float32")
    curr_motors = np.zeros((N, N_MOTORS), dtype="float32")

    for motor_idx in range(N_MOTORS):
        v_col = f"CH{2 * motor_idx}"
        i_col = f"CH{2 * motor_idx + 1}"

        # 电压列：解析后仅用于 sanity check（日志提示），数值不直接使用
        v_vals = _parse_channel_numeric(df[v_col], unit_suffix="V")
        if np.isnan(v_vals).mean() > 0.5:
            print(
                f"[POWER-READER] WARNING: column {v_col} seems not to be voltage "
                "(>50% NaN after stripping 'V'). "
                "Check your raw CSV format."
            )
        # 工程约定：电压统一视为 NOMINAL_VOLTAGE_V，已在初始化时填充

        # 电流列：去掉 "A" 后数值 * CURRENT_GAIN
        i_vals = _parse_channel_numeric(df[i_col], unit_suffix="A")
        curr_motors[:, motor_idx] = (i_vals * CURRENT_GAIN).astype("float32")

    # ---- 4) 通道重排：把“硬件顺序”映射到“逻辑电机编号” ----
    # 逻辑含义：
    #   volt_motors_logic[:, i] = volt_motors_hw[:, MOTOR_PERM[i]]
    #   curr_motors_logic[:, i] = curr_motors_hw[:, MOTOR_PERM[i]]
    #
    # 其中 i = 0..7 对应 Motor1..Motor8
    if MOTOR_PERM.shape[0] != N_MOTORS:
        raise ValueError(
            f"[POWER-READER] MOTOR_PERM size mismatch: "
            f"{MOTOR_PERM.shape[0]} vs N_MOTORS={N_MOTORS}"
        )

    volt_motors = volt_motors[:, MOTOR_PERM]
    curr_motors = curr_motors[:, MOTOR_PERM]

    # ---- 5) 计算 8 路功率（已是逻辑电机编号）----
    # 这里假设所有电机使用同一母线电压 NOMINAL_VOLTAGE_V
    power_motors = (volt_motors * curr_motors).astype("float32")

    frame = PowerFrame(
        path=path,
        time_col=time_col,
        t_s=t_s.astype("float64"),
        time_cols_raw=time_cols_raw,
        est_ns=est_ns_arr,
        mono_ns=mono_ns_arr,
        volt_motors=volt_motors,
        curr_motors=curr_motors,
        power_motors=power_motors,
    )

    return frame
