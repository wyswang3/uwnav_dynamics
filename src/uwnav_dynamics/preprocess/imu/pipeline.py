from __future__ import annotations

"""
uwnav_dynamics.preprocess.imu.pipeline

IMU 预处理总管线（pipeline）：

  原始 CSV / 数组  ->  坐标 / 单位统一 (transform)
                    -> 重力补偿 (gravity)
                    -> 静止窗零偏估计与去 bias (bias)
                    -> 去毛刺 + 一阶低通滤波 (filter)
                    -> 预处理结果 CSV（物理一致、可用于建模与控制）

一、整体流程回顾

1) 原始数据（来自采集程序 / 设备 CSV）：
   - 坐标系：IMU 自身 RFU（X右 Y前 Z上）
   - 单位：
       * 加速度：g
       * 角速度：deg/s
       * 姿态角：deg
   - 时间：MonoS / EstS / MonoNS / EstNS 中至少一列

2) transform.py
   - RFU -> FRD（体坐标系）
   - g -> m/s²，deg/s -> rad/s
   - roll / pitch / yaw : deg -> rad，yaw 符号统一，wrap 到 (-π, π]
   - 给出 dt 序列与 dt 中位数

3) gravity.py
   - 在 ENU 坐标系中定义重力向量：
       g_n = [0, 0, -g0]^T
   - 根据 (roll, pitch, yaw) 计算 R_nb（body -> nav）
   - 计算体坐标下重力 g_b = R_bn * g_n
   - 从测量加速度中减去 g_b，得到线加速度（仍含 bias）：

       a_lin_body_plus_bias = acc_body - g_b

4) bias.py
   - 在静止窗内估计：
       * 加速度零偏 ba_body_mps2
       * 陀螺零偏   bg_body_rad_s
       * 噪声 std 与随机游走近似 rw
   - apply_bias(...) 对整个序列减去估计零偏

5) filter.py
   - 分通道做去毛刺（单步差分阈值）
   - 一阶 IIR 低通（加速度 / 角速度 / yaw）
   - yaw 先 unwrap 再滤波，最后 wrap 回 (-π, π]

6) 本模块（pipeline.py）
   - 整合以上步骤，统一输出：

     a) ImuProcessedFrame（用于后续建模 / 控制）
     b) ImuPreprocessDiag（记录 dt / bias / 噪声统计 / 去毛刺计数等）
     c) 可选：将结果写入 CSV 文件（物理含义清晰的字段）

二、输出 CSV 字段约定（建议）

默认输出 IMU 预处理结果 CSV 包含以下列（可根据需要扩展）：

  - t_s                       : 时间（秒）
  - dt_s                      : 相邻 dt（秒）
  - roll_rad, pitch_rad       : 机体姿态角（rad）
  - yaw_rad                   : yaw（rad, (-π, π]）
  - AccX_body_mps2            : 体坐标系 FRD 下 X（Forward）向线加速度 [m/s²]
  - AccY_body_mps2            : 体坐标系 FRD 下 Y（Right）向线加速度 [m/s²]
  - AccZ_body_mps2            : 体坐标系 FRD 下 Z（Down）向线加速度 [m/s²]
  - AccE_enu_mps2             : ENU 下 East 方向线加速度 [m/s²]
  - AccN_enu_mps2             : ENU 下 North 方向线加速度 [m/s²]
  - AccU_enu_mps2             : ENU 下 Up    方向线加速度 [m/s²]
  - GyroX_body_rad_s          : 体坐标系 FRD 下 X（Forward）轴角速度 [rad/s]
  - GyroY_body_rad_s          : 体坐标系 FRD 下 Y（Right）轴角速度 [rad/s]
  - GyroZ_body_rad_s          : 体坐标系 FRD 下 Z（Down）轴角速度 [rad/s]

这些字段可以直接用于后续：
  - 数据驱动动力学建模（LSTM / SSM 等）
  - 在线导航 / 控制（作为滤波与控制的输入）
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any,Optional

import numpy as np
import pandas as pd

from .transform import ImuTransformConfig, transform_raw
from .gravity import ImuGravityConfig, compensate_gravity, rpy_to_R_nb
from .bias import ImuBiasConfig, estimate_bias, apply_bias
from .filter import ImuFilterConfig, filter_signals

# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class ImuPreprocessConfig:
    """
    IMU 预处理总配置：将 transform / gravity / bias / filter 四部分配置组合在一起。

    通常在 configs/preprocess/imu.yaml 中进行 YAML 化管理，
    此处为运行时的 Python 映射。
    """
    transform: ImuTransformConfig = ImuTransformConfig()
    gravity: ImuGravityConfig = ImuGravityConfig()
    bias: ImuBiasConfig = ImuBiasConfig()
    filt: ImuFilterConfig = ImuFilterConfig()

@dataclass
class ImuProcessedFrame:
    """
    预处理后的 IMU 时间序列数据，用于后续建模与控制。

    所有坐标系与单位：
      - 体坐标系：FRD
      - 全局坐标系：ENU
      - 加速度：m/s²
      - 角速度：rad/s
      - 角度：rad
    """
    t_s: np.ndarray                 # (N,) 时间戳
    dt_s: np.ndarray                # (N,) 相邻 dt
    roll_rad: np.ndarray            # (N,)
    pitch_rad: np.ndarray           # (N,)
    yaw_rad: np.ndarray             # (N,)
    a_lin_body_mps2: np.ndarray     # (N,3) FRD，去重力 / 去 bias / 滤波后
    a_lin_enu_mps2: np.ndarray      # (N,3) ENU，去重力 / 去 bias / 滤波后
    gyro_body_rad_s: np.ndarray     # (N,3) FRD，去 bias / 滤波后

@dataclass
class ImuPreprocessDiag:
    """
    IMU 预处理诊断信息，用于调参与质量评估。

    包含：
      - dt 统计
      - bias 估计结果
      - 噪声 / 随机游走估计
      - 去毛刺计数
      - 所使用的 yaw 源字段等
    """
    n: int
    t0: float
    t1: float
    dt_med: float
    yaw_source_used: str

    bias_window_s: float
    bias_samples: int
    ba_body_mps2: np.ndarray
    bg_body_rad_s: np.ndarray
    std_a_body_mps2: np.ndarray
    std_g_body_rad_s: np.ndarray
    rw_a_body_mps2_sqrt_s: np.ndarray
    rw_g_body_rad_s_sqrt_s: np.ndarray

    despike_counts: Dict[str, int]
    filter_cfg: ImuFilterConfig

# =============================================================================
# 核心预处理函数（数组级）
# =============================================================================
def _dbg_nan(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    n_nan = np.isnan(arr).sum()
    print(f"[IMU-PIPE][DEBUG] {name}: shape={arr.shape}, n_nan={n_nan}")

def preprocess_imu_arrays(
    t_s: np.ndarray,
    acc_rfu_g: np.ndarray,
    gyro_rfu_dps: np.ndarray,
    angx_deg: np.ndarray,
    angy_deg: np.ndarray,
    yawdeg: np.ndarray | None,
    angz_deg: np.ndarray | None,
    cfg: ImuPreprocessConfig,
) -> Tuple[ImuProcessedFrame, ImuPreprocessDiag]:
    """
    以数组形式输入原始 IMU 数据，执行完整预处理管线。

    注意：本函数服务于“动力学建模用的数据预处理”，
    目标是产出尽量无 NaN、物理一致的加速度 / 角速度序列。
    """
    # 0) 基本检查
    t = np.asarray(t_s, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("Need at least 2 IMU samples for preprocessing.")
    N = t.size

    # 1) transform：RFU->FRD，单位转换，yaw 选择与 wrap
    tf = transform_raw(
        t_s=t,
        acc_rfu_g=acc_rfu_g,
        gyro_rfu_dps=gyro_rfu_dps,
        angx_deg=angx_deg,
        angy_deg=angy_deg,
        yawdeg=yawdeg,
        angz_deg=angz_deg,
        cfg=cfg.transform,
    )
    dt_s = tf["dt_s"]               # (N,)
    dt_med = float(tf["dt_med"])    # float
    acc_body_mps2 = tf["acc_body_mps2"]        # (N,3)
    gyro_body_rad_s = tf["gyro_body_rad_s"]    # (N,3)
    roll_rad = tf["roll_rad"]                  # (N,)
    pitch_rad = tf["pitch_rad"]                # (N,)
    yaw_rad_raw = tf["yaw_rad"]                # (N,)
    yaw_source_used = str(tf["yaw_source_used"])

    _dbg_nan("acc_body_mps2 (after transform)", acc_body_mps2)
    _dbg_nan("gyro_body_rad_s (after transform)", gyro_body_rad_s)
    _dbg_nan("yaw_rad_raw (after transform)", yaw_rad_raw)

    # 2) gravity：重力补偿，得到线加速度（仍含 bias）
    grav = compensate_gravity(
        acc_body_mps2=acc_body_mps2,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad_raw,
        cfg=cfg.gravity,
    )
    a_lin_body_plus_bias = grav["a_lin_body_mps2"]   # (N,3)
    R_nb_raw = grav["R_nb"]                          # (N,3,3)  # 目前仅用于调试 / 可视化
    _dbg_nan("a_lin_body_plus_bias (after gravity)", a_lin_body_plus_bias)

    # 3) bias：静止窗统计，估计零偏 / 噪声 / 随机游走
    bias_info = estimate_bias(
        t_s=t,
        a_lin_body_mps2=a_lin_body_plus_bias,
        gyro_body_rad_s=gyro_body_rad_s,
        cfg=cfg.bias,
    )
    ba = bias_info["ba_body_mps2"]       # (3,)
    bg = bias_info["bg_body_rad_s"]      # (3,)
    bias_window_s = float(bias_info["t_window_s"])
    bias_samples = int(bias_info["n_window"])

    print(f"[IMU-PIPE][DEBUG] ba_body_mps2 (bias) = {ba}")
    print(f"[IMU-PIPE][DEBUG] bg_body_rad_s (bias) = {bg}")

    # 4) 去 bias（作为“基础版本”，滤波失败时退回到这一版）
    a_lin_body_nobias, gyro_body_nobias = apply_bias(
        a_lin_body_mps2=a_lin_body_plus_bias,
        gyro_body_rad_s=gyro_body_rad_s,
        ba_body_mps2=ba,
        bg_body_rad_s=bg,
    )
    _dbg_nan("a_lin_body_nobias (after bias)", a_lin_body_nobias)
    _dbg_nan("gyro_body_nobias (after bias)", gyro_body_nobias)

    # 5) filter：去毛刺 + 一阶低通（a_lin / gyro / yaw）
    filt = filter_signals(
        dt_med=dt_med,
        a_lin_body_mps2=a_lin_body_nobias,
        gyro_body_rad_s=gyro_body_nobias,
        yaw_rad=yaw_rad_raw,
        cfg=cfg.filt,
    )
    a_lin_body_filt = filt.get("a_lin_body_mps2_filt")
    gyro_body_filt = filt.get("gyro_body_rad_s_filt")
    yaw_rad_filt = filt.get("yaw_rad_filt")
    despike_counts = filt.get("despike_counts", {})

    _dbg_nan("a_lin_body_filt (raw from filter)", a_lin_body_filt)
    _dbg_nan("gyro_body_filt (raw from filter)", gyro_body_filt)
    _dbg_nan("yaw_rad_filt (raw from filter)", yaw_rad_filt)

    # ------------------------------------------------------------------
    # 5.1 逐样本兜底：一旦某个样本滤波结果为 NaN，就退回 bias-free 版本
    # ------------------------------------------------------------------
    def _sanitize_vec_series(
        name: str,
        filt_arr: Optional[np.ndarray],
        base_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量序列兜底：
        - 若 filt_arr 为 None：整体退回 base_arr；
        - 若部分样本有 NaN：仅这些样本退回 base_arr，其余保留滤波结果。

        返回：
          out   : (N,3) 最终可用序列
          mask_fallback : (N,) bool，True 表示该样本使用了 fallback
        """
        base_arr = np.asarray(base_arr, dtype=float)
        if filt_arr is None:
            print(
                f"[IMU-PIPE] WARNING: filtered '{name}' is None, "
                f"fallback to bias-free signal (all {base_arr.shape[0]} samples)."
            )
            return base_arr.copy(), np.ones(base_arr.shape[0], dtype=bool)

        fa = np.asarray(filt_arr, dtype=float)
        if fa.shape != base_arr.shape:
            raise ValueError(
                f"[IMU-PIPE] '{name}' shape mismatch: "
                f"filt={fa.shape}, base={base_arr.shape}"
            )

        # good: 该样本三个分量都 finite；bad: 这一行有 NaN / inf
        good = np.isfinite(fa).all(axis=1)
        out = fa.copy()
        out[~good] = base_arr[~good]

        n_bad = int((~good).sum())
        if n_bad > 0:
            print(
                f"[IMU-PIPE] NOTE: filtered '{name}' has {n_bad} NaN samples, "
                f"fallback to bias-free signal on these samples."
            )
        return out, ~good

    def _sanitize_scalar_series(
        name: str,
        filt_arr: Optional[np.ndarray],
        base_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        标量序列兜底（如 yaw）：
        - 若 filt_arr 为 None：整体退回 base_arr；
        - 若部分样本有 NaN：仅这些样本退回 base_arr。
        """
        base_arr = np.asarray(base_arr, dtype=float).reshape(-1)
        if filt_arr is None:
            print(
                f"[IMU-PIPE] WARNING: filtered '{name}' is None, "
                f"fallback to raw (all {base_arr.size} samples)."
            )
            return base_arr.copy(), np.ones(base_arr.size, dtype=bool)

        fa = np.asarray(filt_arr, dtype=float).reshape(-1)
        if fa.shape != base_arr.shape:
            raise ValueError(
                f"[IMU-PIPE] '{name}' shape mismatch: "
                f"filt={fa.shape}, base={base_arr.shape}"
            )

        good = np.isfinite(fa)
        out = fa.copy()
        out[~good] = base_arr[~good]

        n_bad = int((~good).sum())
        if n_bad > 0:
            print(
                f"[IMU-PIPE] NOTE: filtered '{name}' has {n_bad} NaN samples, "
                f"fallback to raw on these samples."
            )
        return out, ~good

    a_lin_body_filt, fb_a = _sanitize_vec_series(
        "a_lin_body_mps2", a_lin_body_filt, a_lin_body_nobias
    )
    gyro_body_filt, fb_g = _sanitize_vec_series(
        "gyro_body_rad_s", gyro_body_filt, gyro_body_nobias
    )
    yaw_rad_filt, fb_yaw = _sanitize_scalar_series(
        "yaw_rad", yaw_rad_filt, yaw_rad_raw
    )

    _dbg_nan("a_lin_body_filt (after sanitize)", a_lin_body_filt)
    _dbg_nan("gyro_body_filt (after sanitize)", gyro_body_filt)
    _dbg_nan("yaw_rad_filt (after sanitize)", yaw_rad_filt)

    # 6) 计算 ENU 下线加速度（使用“兜底后”的 yaw & a_body）
    R_nb_filt = rpy_to_R_nb(
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad_filt,
    )
    a_lin_enu_filt = np.empty_like(a_lin_body_filt)
    for i in range(N):
        a_lin_enu_filt[i] = R_nb_filt[i] @ a_lin_body_filt[i]

    # 7) 打包 ImuProcessedFrame
    processed = ImuProcessedFrame(
        t_s=t,
        dt_s=dt_s,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad_filt,
        a_lin_body_mps2=a_lin_body_filt,
        a_lin_enu_mps2=a_lin_enu_filt,
        gyro_body_rad_s=gyro_body_filt,
    )

    # 8) 打包 ImuPreprocessDiag
    diag = ImuPreprocessDiag(
        n=N,
        t0=float(t[0]),
        t1=float(t[-1]),
        dt_med=dt_med,
        yaw_source_used=yaw_source_used,
        bias_window_s=bias_window_s,
        bias_samples=bias_samples,
        ba_body_mps2=bias_info["ba_body_mps2"],
        bg_body_rad_s=bias_info["bg_body_rad_s"],
        std_a_body_mps2=bias_info["std_a_body_mps2"],
        std_g_body_rad_s=bias_info["std_g_body_rad_s"],
        rw_a_body_mps2_sqrt_s=bias_info["rw_a_body_mps2_sqrt_s"],
        rw_g_body_rad_s_sqrt_s=bias_info["rw_g_body_rad_s_sqrt_s"],
        despike_counts=despike_counts,
        filter_cfg=cfg.filt,
    )

    return processed, diag

# =============================================================================
# CSV I/O 辅助：从原始 CSV 跑 pipeline，并输出预处理结果 CSV
# =============================================================================
def _build_processed_df(frame: ImuProcessedFrame) -> pd.DataFrame:
    """
    将 ImuProcessedFrame 转换为 pandas DataFrame，便于写入 CSV。

    当前版本（动力学建模导向）只导出：
      - 时间：t_s, dt_s
      - 姿态：roll_rad, pitch_rad, yaw_rad
      - 体坐标线性加速度：AccX/Y/Z_body_mps2
      - 体坐标角速度：GyroX/Y/Z_body_rad_s

    ENU 线加速度 a_lin_enu_mps2 目前仅保留在内存结构中，
    不写入 CSV（避免出现空列；等实现稳定后再开放导出）。
    """
    t = frame.t_s
    dt_s = frame.dt_s
    roll = frame.roll_rad
    pitch = frame.pitch_rad
    yaw = frame.yaw_rad

    a_b = frame.a_lin_body_mps2  # (N,3)
    g_b = frame.gyro_body_rad_s  # (N,3)

    df = pd.DataFrame(
        {
            "t_s": t,
            "dt_s": dt_s,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "AccX_body_mps2": a_b[:, 0],
            "AccY_body_mps2": a_b[:, 1],
            "AccZ_body_mps2": a_b[:, 2],
            "GyroX_body_rad_s": g_b[:, 0],
            "GyroY_body_rad_s": g_b[:, 1],
            "GyroZ_body_rad_s": g_b[:, 2],
        }
    )
    return df

def run_imu_preprocess_csv(
    in_csv: Path | str,
    out_csv: Path | str,
    cfg: ImuPreprocessConfig,
    time_col_candidates: tuple[str, ...] = ("EstS", "MonoS", "EstNS", "MonoNS"),
) -> ImuPreprocessDiag:
    """
    从原始 IMU CSV 读取数据，执行完整预处理管线，并将结果写入 CSV。

    输出 CSV 设计为「精简版物理一致表」：

      - 原始时间戳列：在原始 CSV 中存在的 MonoNS / EstNS / MonoS / EstS
      - 预处理结果：t_s, dt_s, roll/pitch/yaw, a_body, a_enu, gyro_body

    注意：
      - 不再保留原始的 AccX/AccY/AccZ/GyroX/... 等九轴数据；
      - 原始 raw CSV 作为“底稿”保存在 data/raw 中；
      - 预处理 CSV 则专注于“建模与控制需要的物理一致量”。
    """
    in_path = Path(in_csv)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读原始 CSV
    df_in = pd.read_csv(in_path)

    # 2) 选择时间列（优先 EstS，其次 MonoS，再 EstNS / MonoNS）
    t_col: str | None = None
    for c in time_col_candidates:
        if c in df_in.columns and not df_in[c].isna().all():
            t_col = c
            break
    if t_col is None:
        raise ValueError(
            f"No valid time column found in {in_path}; "
            f"candidates={time_col_candidates}"
        )

    # 3) 根据时间列剔除 NaN 行（确保 dt_med 等统计正常）
    t_raw = df_in[t_col].to_numpy(dtype=float)
    mask_time = np.isfinite(t_raw)
    if not mask_time.all():
        df = df_in.loc[mask_time].reset_index(drop=True)
    else:
        df = df_in.copy()

    # 重新拿一次时间列（已去 NaN）
    t_s = df[t_col].to_numpy(dtype=float)

    # 4) 准备原始 IMU 列（用于 transform）
    def _col_or_nan(name: str) -> np.ndarray | None:
        if name in df.columns and not df[name].isna().all():
            return df[name].to_numpy(dtype=float)
        return None

    acc_rfu_g = df[["AccX", "AccY", "AccZ"]].to_numpy(dtype=float)
    gyro_rfu_dps = df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)
    angx_deg = df["AngX"].to_numpy(dtype=float)
    angy_deg = df["AngY"].to_numpy(dtype=float)

    yawdeg = _col_or_nan("YawDeg")
    angz_deg = _col_or_nan("AngZ")

    # 5) 调用数组级预处理
    processed, diag = preprocess_imu_arrays(
        t_s=t_s,
        acc_rfu_g=acc_rfu_g,
        gyro_rfu_dps=gyro_rfu_dps,
        angx_deg=angx_deg,
        angy_deg=angy_deg,
        yawdeg=yawdeg,
        angz_deg=angz_deg,
        cfg=cfg,
    )

    # 6) 构造“预处理结果 DataFrame”（纯 processed 列）
    df_proc = _build_processed_df(processed)

    # 7) 只从原始 df 中抽取“时间相关列”，其余原始列一律不保留
    time_cols: list[str] = []
    for c in ("MonoNS", "EstNS", "MonoS", "EstS"):
        if c in df.columns:
            time_cols.append(c)

    df_time = df[time_cols].copy() if time_cols else pd.DataFrame(index=df_proc.index)

    # 8) 最终输出表 = [原始时间戳列] + [预处理物理一致列]
    df_out = pd.concat([df_time, df_proc], axis=1)

    # 9) 写出 CSV
    df_out.to_csv(out_path, index=False)

    return diag

