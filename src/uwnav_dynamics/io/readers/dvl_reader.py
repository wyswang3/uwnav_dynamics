# src/uwnav_dynamics/io/readers/dvl_reader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Tuple

import numpy as np
import pandas as pd

DvlSrc = Literal["BI", "BE", "BD", "BS", "TS", "SA"]


@dataclass
class DvlFrame:
    """
    DVL 时间序列（已按 nav_state_tb CSV 读入并统一时间基）。

    坐标与物理语义（约定）：
      - 体坐标 velocity：Vx_body, Vy_body, Vz_body (m/s)，体坐标系 = FRD
      - ENU velocity：Ve_enu, Vn_enu, Vu_enu (m/s)
      - ENU position：E, N, U (m)
      - Depth(m)：由 BD 行给出（其余行通常为 0 或 NaN）

    属性说明
    --------
    path : Path
        原始 CSV 路径。
    time_col : str
        选用的时间列名（通常 "EstS"），用于 t_s。
    t_s : (N,) float
        时间序列，单位 [s]。
    dt_s : (N,) float
        相邻时间差，首个元素等于第二个元素。
    src : (N,) str
        行类型：BI/BE/BD/BS/TS/SA 等。
    v_body_mps : (N,3) float
        体坐标速度 [m/s]，对于非 BI/BS 行通常为 0 或 NaN。
    v_enu_mps : (N,3) float
        ENU 速度 [m/s]，对于非 BE 行通常为 0 或 NaN。
    pos_enu_m : (N,3) float
        ENU 位置 [m]，来自 E/N/U 列。
    depth_m : (N,) float
        深度 [m]，来自 Depth(m)，非 BD 行通常为 0 或 NaN。
    valid : (N,) bool
        Valid 列，代表该观测是否有效。
    is_water_mass : (N,) bool
        IsWaterMass 列，True 表示水团跟踪模式。
    raw_df : pd.DataFrame
        原始 CSV（去掉 NaN 时间行后的版本），作为 debug/扩展备用。
    """
    path: Path
    time_col: str
    t_s: np.ndarray
    dt_s: np.ndarray
    src: np.ndarray
    v_body_mps: np.ndarray
    v_enu_mps: np.ndarray
    pos_enu_m: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    is_water_mass: np.ndarray
    raw_df: pd.DataFrame

    # --------- 常用 mask / 视图 ---------
    def mask_kind(self, kind: DvlSrc, *, require_valid: bool = True) -> np.ndarray:
        """
        返回指定 Src 类型（例如 "BI"）的布尔掩码。
        如果 require_valid=True，则同时要求 Valid==True。
        """
        m = (self.src == kind)
        if require_valid and ("valid" in dir(self)):
            m = m & self.valid
        return m

    def view_kind(self, kind: DvlSrc, *, require_valid: bool = True) -> "DvlKindView":
        """
        提取某一类 Src（BI/BE/BD）的子序列视图。
        """
        m = self.mask_kind(kind, require_valid=require_valid)
        return DvlKindView(
            kind=kind,
            t_s=self.t_s[m],
            v_body_mps=self.v_body_mps[m] if kind in ("BI", "BS") else None,
            v_enu_mps=self.v_enu_mps[m] if kind in ("BE",) else None,
            pos_enu_m=self.pos_enu_m[m],
            depth_m=self.depth_m[m] if kind in ("BD",) else None,
            valid=self.valid[m],
            is_water_mass=self.is_water_mass[m],
        )


@dataclass
class DvlKindView:
    """
    某一类 Src（BI / BE / BD）的子视图。

    用途示例：
      - BI: 体速度真值，用于 ESKF 或动力学建模中的 “ground truth v_body”
      - BE: ENU 速度真值，用于 ESKF 观测或轨迹评估
      - BD: 深度真值，用于 ESKF 的 depth 观测或轨迹评估
    """
    kind: DvlSrc
    t_s: np.ndarray                # (K,)
    v_body_mps: Optional[np.ndarray]  # (K,3) or None
    v_enu_mps: Optional[np.ndarray]   # (K,3) or None
    pos_enu_m: np.ndarray          # (K,3)
    depth_m: Optional[np.ndarray]  # (K,) or None
    valid: np.ndarray              # (K,)
    is_water_mass: np.ndarray      # (K,)

def _pick_time_column(
    df: pd.DataFrame,
    candidates: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS"),
) -> Tuple[str, np.ndarray]:
    """
    类似 IMU 读取逻辑：
      - 优先用秒级时间（EstS / MonoS）
      - 若只剩纳秒级（EstNS / MonoNS），则转为秒

    返回 (time_col_name, t_s)
    """
    for c in ("EstS", "MonoS"):
        if c in df.columns and not df[c].isna().all():
            t = df[c].to_numpy(dtype=float)
            return c, t

    for c in ("EstNS", "MonoNS"):
        if c in df.columns and not df[c].isna().all():
            ns = df[c].to_numpy(dtype=float)
            return c, ns * 1e-9

    raise ValueError(
        f"No valid time column found in DVL CSV; "
        f"candidates={candidates}"
    )

def read_dvl_csv(
    path: str | Path,
    *,
    kind: str = "nav_state_tb",
    drop_nan_time: bool = True,
) -> DvlFrame:
    """
    读取 DVL nav_state_tb CSV，转换为 DvlFrame。

    参数
    ----
    path : str | Path
        CSV 文件路径，例如:
        data/raw/2026-01-10_pooltest02/dvl/dvl_nav_state_tb_20260110_192211.csv

    kind : str
        预留将来支持多种 DVL 文件格式，此处仅用于日志输出。

    drop_nan_time : bool
        若 True，则对选定时间列为 NaN 的行进行剔除。

    返回
    ----
    DvlFrame
        强类型 DVL 时间序列。
    """
    path = Path(path).expanduser().resolve()
    df_raw = pd.read_csv(path)

    if df_raw.empty:
        raise ValueError(f"DVL CSV is empty: {path}")

    # 1) 选时间列并构造 t_s
    time_col, t_s = _pick_time_column(df_raw)
    if drop_nan_time:
        mask_t = np.isfinite(t_s)
        df = df_raw.loc[mask_t].reset_index(drop=True)
        t_s = t_s[mask_t]
    else:
        df = df_raw.copy()

    # 再算一次 dt
    if t_s.size < 2:
        raise ValueError(f"Need at least 2 DVL samples, got {t_s.size}")
    dt_s = np.diff(t_s)
    dt_s = np.concatenate([dt_s[:1], dt_s])

    # 2) Src
    if "Src" not in df.columns:
        raise ValueError(f"DVL CSV missing 'Src' column: {path}")
    src = df["Src"].astype(str).to_numpy()

    # 3) Body velocity (Vx_body/Vy_body/Vz_body)，没有的话就填 NaN
    def _get_vec3(col_x: str, col_y: str, col_z: str) -> np.ndarray:
        if all(c in df.columns for c in (col_x, col_y, col_z)):
            return df[[col_x, col_y, col_z]].to_numpy(dtype=float)
        # 若某些格式缺少，则统一填 NaN，方便上层识别
        n = df.shape[0]
        return np.full((n, 3), np.nan, dtype=float)

    v_body_mps = _get_vec3("Vx_body(m_s)", "Vy_body(m_s)", "Vz_body(m_s)")
    v_enu_mps = _get_vec3("Ve_enu(m_s)", "Vn_enu(m_s)", "Vu_enu(m_s)")

    # 4) 位置/深度
    pos_enu_m = _get_vec3("E(m)", "N(m)", "U(m)")
    depth_m = (
        df["Depth(m)"].to_numpy(dtype=float)
        if "Depth(m)" in df.columns
        else np.full(df.shape[0], np.nan, dtype=float)
    )

    # 5) Valid / IsWaterMass
    def _get_bool(col: str, default: bool = False) -> np.ndarray:
        if col not in df.columns:
            return np.full(df.shape[0], default, dtype=bool)
        v = df[col]
        if v.dtype == bool:
            return v.to_numpy()
        # 可能是 "True"/"False" 或 0/1
        if v.dtype == object:
            return v.astype(str).str.lower().isin(("true", "1", "t", "yes")).to_numpy()
        # numeric
        return (v.to_numpy(dtype=float) != 0.0)

    valid = _get_bool("Valid", default=True)
    is_water_mass = _get_bool("IsWaterMass", default=False)

    return DvlFrame(
        path=path,
        time_col=time_col,
        t_s=t_s,
        dt_s=dt_s,
        src=src,
        v_body_mps=v_body_mps,
        v_enu_mps=v_enu_mps,
        pos_enu_m=pos_enu_m,
        depth_m=depth_m,
        valid=valid,
        is_water_mass=is_water_mass,
        raw_df=df,
    )
