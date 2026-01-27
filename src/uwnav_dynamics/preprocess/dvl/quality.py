# src/uwnav_dynamics/preprocess/dvl/quality.py
from __future__ import annotations

"""
uwnav_dynamics.preprocess.dvl.quality

DVL 质量整理 / 分类模块（BI/BS/BE/BD 拆分 + 列裁剪）。

目标场景：
  - 为“水下动力学建模 / 神经网络系统辨识”提供干净的一致数据；
  - 明确区分：
      * BI / BS: 体坐标系下速度 v_body (m/s)
      * BE     : 全局 ENU 坐标系下速度 v_enu (m/s)
      * BD     : 深度测量 Depth(m)
  - 删除对建模价值不大的冗余列（De_enu, E, N, U, Valid* 等）。

输入原始 CSV 示例列（来自采集程序）：
  - 时间戳:
      MonoNS, EstNS, MonoS, EstS
  - 标识:
      SensorID, Src
  - 速度:
      Vx_body(m_s), Vy_body(m_s), Vz_body(m_s)
      Ve_enu(m_s),  Vn_enu(m_s), Vu_enu(m_s)
  - 位置 / 深度:
      De_enu(m), Dn_enu(m), Du_enu(m)
      Depth(m), E(m), N(m), U(m)
  - 其他:
      Valid, ValidFlag, IsWaterMass

Src 典型取值：
  - "BI" : Bottom-track, body frame velocity    -> v_body
  - "BS" : 也是体坐标速度（状态/辅助），可归入体速度组   -> v_body
  - "BE" : ENU 坐标系下速度，仅平面速度可靠，垂向可参考 -> v_enu
  - "BD" : Depth-only 测量，深度在 Depth(m)

本模块输出的「清洗结果 CSV」包含：
  - 时间：
      MonoNS, EstNS, MonoS, EstS（若存在则保留）
      t_s           : 选定的“主时间轴”（优先 EstS, 次之 MonoS, 再 NS）
  - 源标记：
      Src           : 原始 Src 字段（BI/BS/BE/BD/TS/SA 等）
  - 体速度（FRD）：
      VelX_body_mps, VelY_body_mps, VelZ_body_mps
        * 对 BI/BS 行：来自 Vx_body(m_s)/Vy_body(m_s)/Vz_body(m_s)
        * 对 BE/BD/其他行：设为 NaN
  - ENU 速度：
      VelE_enu_mps, VelN_enu_mps, VelU_enu_mps
        * 对 BE 行：来自 Ve_enu(m_s)/Vn_enu(m_s)/Vu_enu(m_s)
        * 对 BI/BS/BD/其他行：设为 NaN
  - 深度：
      Depth_m       : 对 BD 行来自 Depth(m)，其他行 NaN

注意：
  - 本模块不做数值门控（threshold gating），仅做“分类 + 列裁剪 + 语义统一”；
  - 后续若需要根据速度大小 / ValidFlag / 底追状态做门控，可以在此模块基础上扩展。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 配置与诊断
# =============================================================================


@dataclass
class DvlQualityConfig:
    """
    DVL 质量整理配置。
    ...
    """
    time_col_candidates: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS")
    keep_src: Tuple[str, ...] = ("BI", "BS", "BE", "BD")
    drop_other_src: bool = False

    # ---------- 新增：简单去尖刺参数 ----------
    # 是否启用去尖刺
    enable_despike: bool = True

    # BI/BS 体速度允许的最大模长（超过视为离群点）
    max_body_speed_mps: float = 2.0

    # BE 垂向速度允许的最大绝对值（超过视为离群点）
    max_be_vert_mps: float = 1.0

    # BD 深度允许的单步最大跳变（差分），超过则视为离群点
    max_depth_jump_m: float = 0.5

    # -------- 新增：深度绝对值物理范围 ----------
    # 水池实验物理上不可能超过的深度（比如 10 m）
    max_depth_abs_m: float = 6.0

@dataclass
class DvlQualityDiag:
    """
    DVL 质量整理诊断信息。

    用于了解分类结果和未来门控策略设计。
    """
    n_total: int
    n_kept: int
    counts_by_src: Dict[str, int]
    n_BI: int
    n_BS: int
    n_BE: int
    n_BD: int


# =============================================================================
# 内部工具函数
# =============================================================================


def _select_time_column(df: pd.DataFrame, cfg: DvlQualityConfig) -> str:
    """
    根据配置在 df 中选择一个时间列作为 t_s。
    优先顺序：cfg.time_col_candidates。
    """
    for c in cfg.time_col_candidates:
        if c in df.columns and not df[c].isna().all():
            return c
    raise ValueError(
        f"[DVL-QUALITY] No valid time column found, tried: {cfg.time_col_candidates}"
    )


def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除对动力学建模价值不大的冗余列：
      - De_enu(m), Dn_enu(m), Du_enu(m)
      - E(m), N(m), U(m)
      - Valid, ValidFlag, IsWaterMass
    """
    drop_cols = [
        "De_enu(m)",
        "Dn_enu(m)",
        "Du_enu(m)",
        "E(m)",
        "N(m)",
        "U(m)",
        "Valid",
        "ValidFlag",
        "IsWaterMass",
    ]
    cols_exist = [c for c in drop_cols if c in df.columns]
    if cols_exist:
        df = df.drop(columns=cols_exist)
    return df

def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    对 1D 序列的 NaN 进行线性插值（两端用最近的有效值外推），
    返回新数组，不原地修改。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        return x

    if not np.isnan(x).any():
        return x

    s = pd.Series(x)
    # limit_direction="both"：两端也用最近有效值外推
    x_filled = s.interpolate(limit_direction="both").to_numpy(dtype=float)
    return x_filled


def _interp_nan_2d_cols(X: np.ndarray) -> np.ndarray:
    """
    对 (N,3) 这种 2D 向量序列的每个分量单独做插值兜底。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return X
    N, D = X.shape
    if D == 0 or N == 0:
        return X

    out = X.copy()
    for j in range(D):
        col = out[:, j]
        if np.isnan(col).any():
            out[:, j] = _interp_nan_1d(col)
    return out

# =============================================================================
# 主清洗逻辑：DataFrame -> DataFrame
# =============================================================================
def clean_dvl_dataframe(
    df_raw: pd.DataFrame,
    cfg: Optional[DvlQualityConfig] = None,
) -> Tuple[pd.DataFrame, DvlQualityDiag]:
    """
    对原始 DVL DataFrame 进行分类与列裁剪 + 基础质量控制（去尖刺）。

    输入
    ----
    df_raw :
        直接来自原始 CSV 的 DataFrame。
    cfg :
        DvlQualityConfig；若为 None，则使用默认配置。

    输出
    ----
    (df_out, diag) :
        df_out : 清洗后的 DataFrame（列名见模块说明）
        diag   : DvlQualityDiag，记录各类统计信息
    """
    if cfg is None:
        cfg = DvlQualityConfig()

    df = df_raw.copy()

    # ------------------------------------------------------------------
    # 1) 选择时间列，并构造 t_s
    # ------------------------------------------------------------------
    t_col = _select_time_column(df, cfg)
    t_s = df[t_col].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 2) 可选：只保留关心的 Src 行（BI/BS/BE/BD）
    # ------------------------------------------------------------------
    if "Src" not in df.columns:
        raise ValueError("[DVL-QUALITY] Input DataFrame missing 'Src' column.")

    src = df["Src"].astype(str).fillna("")

    counts_by_src: Dict[str, int] = src.value_counts(dropna=False).to_dict()

    if cfg.drop_other_src:
        mask_keep = src.isin(cfg.keep_src)
        df = df.loc[mask_keep].reset_index(drop=True)
        src = src.loc[mask_keep].reset_index(drop=True)
        t_s = t_s[mask_keep.to_numpy(dtype=bool)]

    # 重新记一遍大小
    n_total = int(len(df_raw))
    n_kept = int(len(df))

    # ------------------------------------------------------------------
    # 3) 提取原始 velocity / depth 列
    # ------------------------------------------------------------------
    required_vel_cols = [
        "Vx_body(m_s)",
        "Vy_body(m_s)",
        "Vz_body(m_s)",
        "Ve_enu(m_s)",
        "Vn_enu(m_s)",
        "Vu_enu(m_s)",
    ]
    for c in required_vel_cols:
        if c not in df.columns:
            raise ValueError(f"[DVL-QUALITY] Required column missing: {c}")

    body_raw = df[["Vx_body(m_s)", "Vy_body(m_s)", "Vz_body(m_s)"]].to_numpy(dtype=float)
    enu_raw = df[["Ve_enu(m_s)", "Vn_enu(m_s)", "Vu_enu(m_s)"]].to_numpy(dtype=float)

    depth_raw: Optional[np.ndarray] = None
    if "Depth(m)" in df.columns:
        depth_raw = df["Depth(m)"].to_numpy(dtype=float)

    N = len(df)

    # 初始化输出数组：默认 NaN
    vel_body = np.full((N, 3), np.nan, dtype=float)  # VelX_body_mps, VelY_body_mps, VelZ_body_mps
    vel_enu = np.full((N, 3), np.nan, dtype=float)   # VelE_enu_mps, VelN_enu_mps, VelU_enu_mps
    depth_m = np.full((N,), np.nan, dtype=float)     # Depth_m

    # ------------------------------------------------------------------
    # 4) 分类掩码
    # ------------------------------------------------------------------
    src_arr = src.to_numpy(dtype=str)
    mask_BI = (src_arr == "BI")
    mask_BS = (src_arr == "BS")
    mask_BE = (src_arr == "BE")
    mask_BD = (src_arr == "BD")

    # ------------------------------------------------------------------
    # 5) BI / BS：体坐标速度
    # ------------------------------------------------------------------
    mask_body = mask_BI | mask_BS
    vel_body[mask_body] = body_raw[mask_body]

    # ------------------------------------------------------------------
    # 6) BE：只保留 ENU 垂向速度（U 分量）
    # ------------------------------------------------------------------
    vel_enu[mask_BE, 2] = enu_raw[mask_BE, 2]  # VelU_enu_mps

    # ------------------------------------------------------------------
    # 7) BD：深度
    # ------------------------------------------------------------------
    if depth_raw is not None:
        depth_m[mask_BD] = depth_raw[mask_BD]

    # ------------------------------------------------------------------
    # 7.5) 去尖刺 + 插值兜底（体速度 / BE 垂向速度 / 深度）
    # ------------------------------------------------------------------
    if cfg.enable_despike:
        # ---- 7.5.1 BI/BS 体速度：按速度模长做阈值 ----
        speed_body = np.linalg.norm(vel_body, axis=1)
        mask_valid_body = np.isfinite(speed_body)
        mask_spike_body = np.zeros_like(speed_body, dtype=bool)

        if np.any(mask_valid_body):
            mask_spike_body[mask_valid_body] = (
                speed_body[mask_valid_body] > cfg.max_body_speed_mps
            )
            if mask_spike_body.any():
                print(
                    f"[DVL-QUALITY] Despike body velocity: "
                    f"{int(mask_spike_body.sum())} samples | "
                    f"max_body_speed_mps={cfg.max_body_speed_mps}"
                )
                vel_body[mask_spike_body, :] = np.nan

        # ---- 7.5.2 BE 垂向速度：按绝对值阈值 ----
        vup = vel_enu[:, 2]
        mask_valid_be = np.isfinite(vup)
        mask_spike_be = np.zeros_like(vup, dtype=bool)

        if np.any(mask_valid_be):
            mask_spike_be[mask_valid_be] = (
                np.abs(vup[mask_valid_be]) > cfg.max_be_vert_mps
            )
            if mask_spike_be.any():
                print(
                    f"[DVL-QUALITY] Despike BE vertical velocity: "
                    f"{int(mask_spike_be.sum())} samples | "
                    f"max_be_vert_mps={cfg.max_be_vert_mps}"
                )
                vel_enu[mask_spike_be, 2] = np.nan

        # ---- 7.5.3 BD 深度：绝对值范围 + 单步跳变阈值 ----
        if np.isfinite(depth_m).any():
            # 绝对值范围（物理不可能深度）
            if getattr(cfg, "max_depth_abs_m", None) is not None:
                mask_bad_abs = np.abs(depth_m) > cfg.max_depth_abs_m
                if mask_bad_abs.any():
                    print(
                        f"[DVL-QUALITY] Depth abs out-of-range: "
                        f"{int(mask_bad_abs.sum())} samples | "
                        f"max_depth_abs_m={cfg.max_depth_abs_m}"
                    )
                    depth_m[mask_bad_abs] = np.nan

            # 单步跳变
            depth_diff = np.diff(depth_m)
            depth_diff = np.concatenate([[0.0], depth_diff])

            mask_valid_depth = np.isfinite(depth_m) & np.isfinite(depth_diff)
            mask_spike_depth = np.zeros_like(depth_m, dtype=bool)

            if np.any(mask_valid_depth):
                mask_spike_depth[mask_valid_depth] = (
                    np.abs(depth_diff[mask_valid_depth]) > cfg.max_depth_jump_m
                )
                if mask_spike_depth.any():
                    print(
                        f"[DVL-QUALITY] Despike depth: "
                        f"{int(mask_spike_depth.sum())} samples | "
                        f"max_depth_jump_m={cfg.max_depth_jump_m}"
                    )
                    depth_m[mask_spike_depth] = np.nan

        # ---- 7.5.4 对 NaN 做插值兜底，保证序列连续 ----
        vel_body = _interp_nan_2d_cols(vel_body)     # 每一列线性插值
        vel_enu[:, 2] = _interp_nan_1d(vel_enu[:, 2])
        depth_m = _interp_nan_1d(depth_m)

    # ------------------------------------------------------------------
    # 8) 构造输出 DataFrame
    # ------------------------------------------------------------------
    time_cols_to_keep = [c for c in ("MonoNS", "EstNS", "MonoS", "EstS") if c in df.columns]
    df_out = df[time_cols_to_keep].copy()
    df_out["t_s"] = t_s
    df_out["Src"] = src_arr

    # 体速度（FRD）
    df_out["VelX_body_mps"] = vel_body[:, 0]
    df_out["VelY_body_mps"] = vel_body[:, 1]
    df_out["VelZ_body_mps"] = vel_body[:, 2]

    # ENU 速度（目前只填 U 分量，其余保持 NaN）
    df_out["VelE_enu_mps"] = vel_enu[:, 0]
    df_out["VelN_enu_mps"] = vel_enu[:, 1]
    df_out["VelU_enu_mps"] = vel_enu[:, 2]

    # 深度
    df_out["Depth_m"] = depth_m

    # ------------------------------------------------------------------
    # 9) 删除冗余列（De_enu/E/N/U/Valid* 等）
    # ------------------------------------------------------------------
    df_out = _drop_unused_columns(df_out)

    # ------------------------------------------------------------------
    # 10) 构造诊断信息
    # ------------------------------------------------------------------
    diag = DvlQualityDiag(
        n_total=n_total,
        n_kept=n_kept,
        counts_by_src=counts_by_src,
        n_BI=int(mask_BI.sum()),
        n_BS=int(mask_BS.sum()),
        n_BE=int(mask_BE.sum()),
        n_BD=int(mask_BD.sum()),
    )

    return df_out, diag

# =============================================================================
# CSV I/O 封装：从 CSV -> 清洗 CSV
# =============================================================================
def run_dvl_quality_csv(
    in_csv: Path | str,
    out_csv: Path | str,
    cfg: Optional[DvlQualityConfig] = None,
) -> DvlQualityDiag:
    """
    从原始 DVL CSV 读取数据，执行 clean_dvl_dataframe，
    并将清洗结果写入 out_csv。

    一般用于离线预处理管线：
      raw CSV (driver) -> quality CSV (BI/BE/BD 拆分 & 列裁剪)
    """
    in_path = Path(in_csv).expanduser().resolve()
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_path)
    df_clean, diag = clean_dvl_dataframe(df_raw, cfg)

    df_clean.to_csv(out_path, index=False)

    return diag
