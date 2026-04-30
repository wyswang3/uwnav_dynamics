"""
模块名称：Power 日志读取器

模块职责：
负责读取 Volt32 等电压电流日志，
整理为统一的 `PowerFrame`，供辅助功率特征构建和质量审查使用。

主要功能：
1. 解析 16 通道原始字符串并整理成 8 路电机电流/功率。
2. 统一时间列并保留原始时间戳列。
3. 根据当前工程约定生成固定电压和放大后的电流值。
"""

from __future__ import annotations

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
# 论文与当前工程默认把母线电压近似视为 24 V。
# 若后续要切回真实逐时电压测量，应在此处替换为显式电压通道解析。
NOMINAL_VOLTAGE_V: float = 24.0
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
    """按统一优先级选择时间列，并输出秒级时间轴。"""
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
    """收集原始时间戳列，供调试与后续导出复用。"""
    out: Dict[str, np.ndarray] = {}
    for c in ("MonoNS", "EstNS", "MonoS", "EstS"):
        if c in df.columns:
            out[c] = df[c].to_numpy()
    return out


def _parse_channel_numeric(series: pd.Series, unit_suffix: str) -> np.ndarray:
    """把带单位后缀的字符串通道解析成浮点数组。"""
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
    """读取电压电流日志，并生成 8 路电机功率序列。"""
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
