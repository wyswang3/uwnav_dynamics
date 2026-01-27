from __future__ import annotations

"""
uwnav_dynamics.preprocess.dvl.pipeline

DVL 预处理总管线（简化版，先保证「可用的 CSV 输出」）：

  原始 CSV  ->  选择时间列（EstS/MonoS/...）
             -> 选择 BI 速度列（若干命名候选 + 显式配置）
             -> 单位统一到 m/s
             -> 简单有效性标记（按 status / quality）
             -> 写出「动力学建模友好」的 processed CSV
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from uwnav_dynamics.preprocess.dvl.quality import (
    clean_dvl_dataframe,
    DvlQualityConfig,
)

# =============================================================================
# 配置与数据结构
# =============================================================================
@dataclass
class DvlPreprocessConfig:
    """
    DVL 预处理配置。

    vel_scale:
        将原始「速度」单位换算到 m/s 的比例。
        - 原始为 m/s   -> 1.0
        - 原始为 cm/s -> 0.01
        - 原始为 mm/s -> 0.001

    min_quality:
        质量阈值，quality < min_quality 的样本标记为 invalid。

    use_status_as_valid:
        若为 True，则尝试用 status 列（例如 ValidFlag/Status）进一步过滤。

    quality_col:
        原始 CSV 中的质量字段列名；若为 None 则忽略质量。

    status_col:
        原始 CSV 中的有效性状态列名，如 "ValidFlag"、"Status"；若为 None 则忽略状态字段。

    bi_vel_cols:
        显式指定 BI 体坐标速度三列的列名 (vx, vy, vz)。
        若为 None，则回退到自动 candidates 匹配。

    be_up_col:
        ENU 垂向速度列名（例如 "Vu_enu(m_s)"）。若为 None 则不输出该列。

    depth_col:
        深度列名（例如 "Depth(m)"）。若为 None 则不输出该列。

    depth_scale:
        深度单位换算比例（原始为 m -> 1.0；若为 cm 则可设 0.01）。
    """
    vel_scale: float = 1.0
    min_quality: int = 0
    use_status_as_valid: bool = True

    quality_col: Optional[str] = None
    status_col: Optional[str] = None

    bi_vel_cols: Optional[Tuple[str, str, str]] = None

    be_up_col: Optional[str] = None
    depth_col: Optional[str] = None
    depth_scale: float = 1.0


@dataclass
class DvlProcessedFrame:
    """
    预处理后的 DVL 时间序列。

    全部在体坐标系 FRD 下，单位 m/s。
    """
    t_s: np.ndarray             # (N,)
    dt_s: np.ndarray            # (N,)
    v_body_mps: np.ndarray      # (N,3) -> [VelBx, VelBy, VelBz]
    speed_mps: np.ndarray       # (N,)
    valid: np.ndarray           # (N,) bool


@dataclass
class DvlPreprocessDiag:
    """
    预处理诊断信息：便于日志打印和调参。
    """
    n: int
    t0: float
    t1: float
    dt_med: float
    n_valid: int
    n_invalid: int
    vel_scale: float
    min_quality: int


# =============================================================================
# 核心数组级预处理
# =============================================================================

def preprocess_dvl_arrays(
    t_s: np.ndarray,
    v_body_raw: np.ndarray,
    quality: Optional[np.ndarray],
    status: Optional[np.ndarray],
    cfg: DvlPreprocessConfig,
) -> Tuple[DvlProcessedFrame, DvlPreprocessDiag]:
    """
    纯数组级 DVL 预处理：

      - 统一时间轴（计算 dt / dt_med）
      - 速度单位转换到 m/s
      - 计算速度模长
      - 基于 quality / status 得到 valid mask
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("DVL preprocessing needs at least 2 samples.")
    N = t.size

    v_raw = np.asarray(v_body_raw, dtype=float)
    if v_raw.shape != (N, 3):
        raise ValueError(f"v_body_raw must be (N,3), got {v_raw.shape}")

    # 时间差
    dt = np.diff(t)
    dt = np.concatenate([dt[:1], dt])
    dt_med = float(np.median(dt))

    # 单位转换
    v_mps = v_raw * float(cfg.vel_scale)

    # 速度模长
    speed = np.linalg.norm(v_mps, axis=1)

    # 有效性判定
    valid = np.ones(N, dtype=bool)

    if quality is not None:
        q = np.asarray(quality, dtype=float).reshape(-1)
        if q.size != N:
            raise ValueError("quality size mismatch with t_s.")
        valid &= (q >= cfg.min_quality)

    if cfg.use_status_as_valid and status is not None:
        s = np.asarray(status, dtype=float).reshape(-1)
        if s.size != N:
            raise ValueError("status size mismatch with t_s.")
        # 这里简单认为 status > 0 为有效，你可以按实际协议改成 ==1 / ==3 等
        valid &= (s > 0)

    n_valid = int(valid.sum())
    n_invalid = int((~valid).sum())

    frame = DvlProcessedFrame(
        t_s=t,
        dt_s=dt,
        v_body_mps=v_mps,
        speed_mps=speed,
        valid=valid,
    )

    diag = DvlPreprocessDiag(
        n=N,
        t0=float(t[0]),
        t1=float(t[-1]),
        dt_med=dt_med,
        n_valid=n_valid,
        n_invalid=n_invalid,
        vel_scale=cfg.vel_scale,
        min_quality=cfg.min_quality,
    )

    return frame, diag

# =============================================================================
# CSV I/O：原始 CSV -> 预处理 CSV
# =============================================================================

def _pick_time_column(
    df: pd.DataFrame,
    candidates: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS"),
) -> str:
    """
    从候选列表中挑选一个非全 NaN 的时间列。
    """
    for c in candidates:
        if c in df.columns and not df[c].isna().all():
            return c
    raise ValueError(f"No valid time column in DVL CSV, candidates={candidates}")


def _pick_velocity_columns(
    df: pd.DataFrame,
    cfg: DvlPreprocessConfig,
) -> Tuple[str, str, str]:
    """
    选择体坐标 BI 速度三列。

    优先级：
      1) 若 cfg.bi_vel_cols 显式配置，则优先使用；
      2) 否则，在若干候选命名中自动匹配。
    """
    # 1) 显式配置优先
    if cfg.bi_vel_cols is not None:
        vx, vy, vz = cfg.bi_vel_cols
        missing = [c for c in (vx, vy, vz) if c not in df.columns]
        if missing:
            raise ValueError(
                "[DVL-PIPE] bi_vel_cols configured as "
                f"{cfg.bi_vel_cols}, but missing columns={missing}. "
                f"Available columns={list(df.columns)}"
            )
        return vx, vy, vz

    # 2) 自动候选（按你当前 CSV 优先）
    candidates: Tuple[Tuple[str, str, str], ...] = (
        ("Vx_body(m_s)", "Vy_body(m_s)", "Vz_body(m_s)"),  # 你的真实列
        ("VelBx", "VelBy", "VelBz"),
        ("VelX", "VelY", "VelZ"),
        ("Vx", "Vy", "Vz"),
        ("vx_bi_mps", "vy_bi_mps", "vz_bi_mps"),
        ("Vx_body_mps", "Vy_body_mps", "Vz_body_mps"),
    )

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    # 仍找不到，报详细错误
    raise ValueError(
        "[DVL-PIPE] No velocity columns found in DVL CSV. "
        f"Tried candidates={candidates}. "
        "Consider setting DvlPreprocessConfig.bi_vel_cols explicitly. "
        f"Available columns={list(df.columns)}"
    )


def _build_processed_df(
    frame: DvlProcessedFrame,
    df_raw_time: pd.DataFrame,
    vup_raw: Optional[np.ndarray],
    depth_raw: Optional[np.ndarray],
    cfg: DvlPreprocessConfig,
) -> pd.DataFrame:
    """
    构造最终写出的 DVL processed DataFrame。

    策略：
      - 只保留原始「时间戳列」用于对比；
      - 写入统一命名 / 单位的体坐标速度：
          VelBx_body_mps, VelBy_body_mps, VelBz_body_mps, Speed_body_mps
      - 可选附加：
          VelU_enu_mps（BE 垂向速度，ENU Up，单位 m/s）
          Depth_m     （BD 深度，单位 m）
      - valid 为 0/1 掩码。
    """
    df_out = df_raw_time.copy()

    v = frame.v_body_mps
    if v.shape[1] != 3:
        raise ValueError(f"v_body_mps must be (N,3), got {v.shape}")

    df_out["t_s"] = frame.t_s
    df_out["dt_s"] = frame.dt_s
    df_out["VelBx_body_mps"] = v[:, 0]
    df_out["VelBy_body_mps"] = v[:, 1]
    df_out["VelBz_body_mps"] = v[:, 2]
    df_out["Speed_body_mps"] = frame.speed_mps
    df_out["valid"] = frame.valid.astype(int)

    # ---- BE 垂向速度（ENU Up）----
    if vup_raw is not None:
        vup_raw = np.asarray(vup_raw, dtype=float).reshape(-1)
        if vup_raw.size != frame.t_s.size:
            raise ValueError(
                f"[DVL-PIPE] be_up array length mismatch: "
                f"{vup_raw.size} vs {frame.t_s.size}"
            )
        # 默认认为单位与 BI 速度一致，共用 vel_scale
        df_out["VelU_enu_mps"] = vup_raw * float(cfg.vel_scale)

    # ---- BD 深度 ----
    if depth_raw is not None:
        depth_raw = np.asarray(depth_raw, dtype=float).reshape(-1)
        if depth_raw.size != frame.t_s.size:
            raise ValueError(
                f"[DVL-PIPE] depth array length mismatch: "
                f"{depth_raw.size} vs {frame.t_s.size}"
            )
        df_out["Depth_m"] = depth_raw * float(cfg.depth_scale)

    return df_out


def run_dvl_preprocess_csv(
    in_csv: Path | str,
    out_csv: Path | str,
    cfg: DvlPreprocessConfig,
    time_col_candidates: Tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS"),
) -> DvlPreprocessDiag:
    """
    从原始 DVL CSV 跑完整预处理，并写出 processed CSV。

    当前策略：
      - 只保留原始时间戳列（MonoNS/EstNS/MonoS/EstS 中存在的那些）；
      - 速度等只保留「预处理后」的版本；
      - 附带 ENU 垂向速度 VelU_enu_mps 与 Depth_m（若配置了对应列名）。
    """
    in_path = Path(in_csv)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_path)

    # 1) 挑选时间列，并按时间去 NaN
    t_col = _pick_time_column(df_raw, candidates=time_col_candidates)
    t_raw = df_raw[t_col].to_numpy(dtype=float)
    mask_time = np.isfinite(t_raw)
    if not mask_time.all():
        df = df_raw.loc[mask_time].reset_index(drop=True)
    else:
        df = df_raw.copy()

    t_s = df[t_col].to_numpy(dtype=float)

    # 2) 只保留时间戳相关原始列（方便对比）
    time_cols_to_keep = [c for c in ("MonoNS", "EstNS", "MonoS", "EstS") if c in df.columns]
    df_time = df[time_cols_to_keep].copy()

    # 3) 选择体坐标 BI 速度列
    vx_col, vy_col, vz_col = _pick_velocity_columns(df, cfg)
    v_raw = df[[vx_col, vy_col, vz_col]].to_numpy(dtype=float)

    # 3b) 取 BE 垂向速度原始列（可选）
    vup_raw: Optional[np.ndarray] = None
    if cfg.be_up_col and cfg.be_up_col in df.columns:
        vup_raw = df[cfg.be_up_col].to_numpy(dtype=float)

    # 3c) 取 BD 深度原始列（可选 + 去尖刺）
    depth_raw: Optional[np.ndarray] = None
    if cfg.depth_col and cfg.depth_col in df.columns:
        # 使用 quality 模块做一次 depth 清洗（带去尖刺 + 插值）
        q_cfg = DvlQualityConfig(
            drop_other_src=False,   # 不删其他 Src，保证行数对齐
            # 深度去尖刺开关和阈值（需要在 DvlQualityConfig 中定义）
            enable_despike=True,
            max_depth_abs_m=10.0,   # 例如水池最大 10 m，可按实际改
            max_depth_jump_m=0.5,   # 单步跳变超过 0.5 m 视为尖刺
        )

        # 注意：这里用的是当前的 df（已经做了时间掩码），避免长度不一致
        df_clean, _qdiag = clean_dvl_dataframe(df, q_cfg)

        if "Depth_m" in df_clean.columns:
            depth_raw = df_clean["Depth_m"].to_numpy(dtype=float)
        else:
            # 兜底逻辑：如果因某种原因没有 Depth_m，就退回原始 Depth(m)
            depth_raw = df[cfg.depth_col].to_numpy(dtype=float)


    # 4) 可选 quality / status 列
    if cfg.quality_col and cfg.quality_col in df.columns:
        quality = df[cfg.quality_col].to_numpy(dtype=float)
    else:
        quality = None

    if cfg.status_col and cfg.status_col in df.columns:
        status = df[cfg.status_col].to_numpy(dtype=float)
    else:
        status = None

    # 5) 数组级预处理（只负责 BI + valid）
    frame, diag = preprocess_dvl_arrays(
        t_s=t_s,
        v_body_raw=v_raw,
        quality=quality,
        status=status,
        cfg=cfg,
    )

    # 6) 构建输出 DataFrame 并写 CSV：这里附带 BE/BD
    df_out = _build_processed_df(frame, df_time, vup_raw, depth_raw, cfg)
    df_out.to_csv(out_path, index=False)

    return diag
