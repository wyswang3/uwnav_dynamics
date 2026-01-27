from __future__ import annotations

"""
uwnav_dynamics.preprocess.imu.transform

IMU 原始数据的坐标 / 单位 / 角度整理模块。

一、坐标系约定（全项目统一）

1) 传感器坐标系（sensor frame, IMU 自身）
   - 采用 RFU（Right–Forward–Up）右手坐标系：
     - X_s：向右（Right）
     - Y_s：向前（Forward）
     - Z_s：向上（Up）

   原始加速度 / 角速度数据均在该坐标系下给出：
     - AccX, AccY, AccZ      -> [g]
     - GyroX, GyroY, GyroZ   -> [deg/s]

2) 体坐标系（body frame）
   - 采用 FRD（Forward–Right–Down）右手坐标系：
     - X_b：向前（Forward）
     - Y_b：向右（Right）
     - Z_b：向下（Down）

   这是后续控制与动力学建模使用的「机器人体坐标系」。
   本模块完成的核心工作之一：RFU -> FRD 坐标变换。

3) 全局导航坐标系（navigation frame）
   - 采用 ENU（East–North–Up）右手坐标系：
     - X_n：East（东）
     - Y_n：North（北）
     - Z_n：Up（上）

   本模块不直接进行 body->ENU 的旋转，
   但：
     - yaw 的符号 / 范围设置与 ENU 语义保持一致：
       * yaw > 0：通常表示从 East 朝 North 逆时针旋转（数学正方向）
       * yaw 范围统一为 (-π, π]

二、单位约定

原始数据单位：
  - 加速度：AccX/Y/Z 为「g」（1g ≈ 9.78 m/s²）
  - 角速度：GyroX/Y/Z 为「deg/s」
  - 姿态角：AngX/AngY/AngZ、YawDeg 等为「deg」

本模块输出单位：
  - 加速度（体坐标系 FRD）：[m/s²]
  - 角速度（体坐标系 FRD）：[rad/s]
  - 姿态角（roll, pitch, yaw）：[rad]，且 yaw wrap 到 (-π, π]

三、模块职责

本模块只做「确定性的几何与单位变换」，不做：
  - 重力补偿（由 gravity.py 完成）
  - 零偏估计（由 bias.py 完成）
  - 滤波与去毛刺（由 filter.py 完成）

输出结果将作为后续 gravity / bias / filter 的输入。
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ImuTransformConfig:
    """
    IMU 原始数据坐标 / 单位变换配置。

    属性说明
    ----------
    g0 : float
        重力加速度标称值，用于 g -> m/s² 的单位转换。
        对海平面附近，可以约取 9.78 或 9.81。

    yaw_source : str
        选择作为「航向角」的原始字段：
          - "YawDeg" : 优先使用独立的 YawDeg 字段
          - "AngZ"   : 若没有 YawDeg，可使用 AngZ 作为 yaw 源

    yaw_sign : float
        yaw 符号修正因子：
          - +1.0：原始 yaw 的正方向与 ENU 数学正方向一致
          - -1.0：原始 yaw 与 ENU 定义相反时，用 -1.0 翻转

        最终输出 yaw_rad = yaw_sign * yaw_raw_rad，并 wrap 到 (-π, π]。
    """
    g0: float = 9.78
    yaw_source: str = "YawDeg"  # "YawDeg" | "AngZ"
    yaw_sign: float = 1.0       # 若厂家 yaw 与数学 yaw 方向相反，可设为 -1.0


def wrap_pm_pi(a_rad: np.ndarray) -> np.ndarray:
    """
    将角度 wrap 到 (-π, π] 范围。

    参数
    ----------
    a_rad : np.ndarray
        任意实数角度（rad）。

    返回
    ----------
    np.ndarray
        wrap 后角度，范围在 (-π, π]。
    """
    a = np.asarray(a_rad, dtype=float)
    # 使用 atan2 保证稳定 wrap
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rfu_to_frd(v_rfu: np.ndarray) -> np.ndarray:
    """
    将向量从 RFU（Right-Forward-Up）坐标系变换到 FRD（Forward-Right-Down）。

    RFU 定义：
      - X_s: Right
      - Y_s: Forward
      - Z_s: Up

    FRD 定义：
      - X_b: Forward
      - Y_b: Right
      - Z_b: Down

    因此，有：
      X_b = Y_s
      Y_b = X_s
      Z_b = -Z_s

    参数
    ----------
    v_rfu : np.ndarray
        形状 (N, 3) 或 (3,) 的数组，RFU 坐标向量。

    返回
    ----------
    np.ndarray
        同形状数组，已转换到 FRD 坐标。
    """
    arr = np.asarray(v_rfu, dtype=float)
    if arr.ndim == 1:
        x_s, y_s, z_s = arr[..., 0], arr[..., 1], arr[..., 2]
        v_frd = np.empty_like(arr)
        v_frd[0] = y_s
        v_frd[1] = x_s
        v_frd[2] = -z_s
        return v_frd

    # (N, 3)
    x_s = arr[:, 0]
    y_s = arr[:, 1]
    z_s = arr[:, 2]
    out = np.empty_like(arr)
    out[:, 0] = y_s      # X_b (Forward)  <- Y_s (Forward)
    out[:, 1] = x_s      # Y_b (Right)    <- X_s (Right)
    out[:, 2] = -z_s     # Z_b (Down)     <- -Z_s (Up)
    return out


def _select_yaw_deg(
    yawdeg: Optional[np.ndarray],
    angz_deg: Optional[np.ndarray],
    cfg: ImuTransformConfig,
) -> tuple[np.ndarray, str]:
    """
    根据配置选择 yaw 源字段（YawDeg / AngZ）。

    返回
    ----------
    yaw_deg : np.ndarray
        选定的 yaw（deg）。
    src_name : str
        实际使用的源字段名称："YawDeg" 或 "AngZ"。
    """
    if cfg.yaw_source == "YawDeg" and yawdeg is not None:
        y = np.asarray(yawdeg, dtype=float)
        return y, "YawDeg"

    if cfg.yaw_source == "AngZ" and angz_deg is not None:
        y = np.asarray(angz_deg, dtype=float)
        return y, "AngZ"

    # 回退策略：如果配置源不可用，尝试另一个；若都没有则报错
    if yawdeg is not None:
        return np.asarray(yawdeg, dtype=float), "YawDeg"
    if angz_deg is not None:
        return np.asarray(angz_deg, dtype=float), "AngZ"

    raise ValueError("No valid yaw source: both YawDeg and AngZ are None.")

def _interp_nan_on_time(t_s: np.ndarray, y_deg: np.ndarray, name: str) -> np.ndarray:
    """
    在时间轴 t_s 上对含 NaN 的角度序列（deg）做线性插值补齐，
    两端用最近的有限值延拓。

    参数
    ----------
    t_s : np.ndarray
        时间戳 (N,)，单位 s。
    y_deg : np.ndarray
        原始角度 (N,)，单位 deg，允许包含 NaN。
    name : str
        仅用于 debug 输出（例如 "yaw_deg"）。
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    y = np.asarray(y_deg, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError(f"[transform_raw] {name} shape mismatch: t.shape={t.shape}, y.shape={y.shape}")

    mask = np.isfinite(y)
    n_fin = int(mask.sum())
    n_nan = int((~mask).sum())

    if n_nan == 0:
        return y

    if n_fin == 0:
        raise ValueError(f"[transform_raw] {name} has no finite samples, cannot interpolate.")

    # 仅用有限样本点做线性插值，np.interp 会对两端自动延拓
    y_interp = np.interp(t, t[mask], y[mask])
    print(f"[IMU-TRANS] {name} had {n_nan} NaN samples, interpolated on time axis.")
    return y_interp


def transform_raw(
    t_s: np.ndarray,
    acc_rfu_g: np.ndarray,
    gyro_rfu_dps: np.ndarray,
    angx_deg: np.ndarray,
    angy_deg: np.ndarray,
    yawdeg: Optional[np.ndarray],
    angz_deg: Optional[np.ndarray],
    cfg: ImuTransformConfig,
) -> Dict[str, np.ndarray | float | str]:
    """
    对原始 IMU 数据进行「坐标 / 单位 / 角度」统一整理：

    1) 坐标系：RFU -> FRD（传感器 -> 体坐标）
    2) 单位：
       - 加速度：g -> m/s²
       - 角速度：deg/s -> rad/s
       - 姿态角：deg -> rad
    3) yaw 选择与符号：
       - 按 cfg.yaw_source 选择 YawDeg 或 AngZ
       - 应用 cfg.yaw_sign
       - wrap 到 (-π, π]

    输入
    ----------
    t_s : np.ndarray
        时间戳（秒），形状 (N,)。
    acc_rfu_g : np.ndarray
        原始加速度，RFU 坐标系，单位 g，形状 (N, 3)。
    gyro_rfu_dps : np.ndarray
        原始角速度，RFU 坐标系，单位 deg/s，形状 (N, 3)。
    angx_deg : np.ndarray
        原始 roll 角（deg），通常来自 AngX。
    angy_deg : np.ndarray
        原始 pitch 角（deg），通常来自 AngY。
    yawdeg : np.ndarray | None
        原始 YawDeg 字段（deg），若不存在可为 None。
    angz_deg : np.ndarray | None
        原始 AngZ 字段（deg），可作为 yaw 备选。

    cfg : ImuTransformConfig
        坐标 / 单位 / yaw 选择配置。

    返回
    ----------
    dict
        包含以下键值（均为 np.ndarray 或 float / str）：
          - "t_s"                : (N,) 时间戳
          - "dt_s"               : (N,) 相邻 dt，首个元素与第二个相同
          - "dt_med"             : float，dt 中位数
          - "acc_body_mps2"      : (N,3) FRD 体坐标系加速度 [m/s²]
          - "gyro_body_rad_s"    : (N,3) FRD 体坐标系角速度 [rad/s]
          - "roll_rad"           : (N,) 机体 roll [rad]
          - "pitch_rad"          : (N,) 机体 pitch [rad]
          - "yaw_rad"            : (N,) 机体 yaw [rad]，已应用 yaw_sign 并 wrap 到 (-π, π]
          - "yaw_source_used"    : str，实际使用的 yaw 源字段名称
    """
    t_s = np.asarray(t_s, dtype=float)
    if t_s.ndim != 1:
        raise ValueError("t_s must be a 1D array.")

    # 1) 时间差与 dt 中位数（后续滤波 / 诊断使用）
    if t_s.size < 2:
        raise ValueError("Need at least 2 IMU samples to compute dt.")
    dt_s = np.diff(t_s)
    # 头一个样本的 dt 用第二个样本的 dt 代替，避免长度不一致
    dt_s = np.concatenate([dt_s[:1], dt_s])
    dt_med = float(np.median(dt_s))

    # 2) 加速度：RFU -> FRD，g -> m/s²
    acc_rfu_g = np.asarray(acc_rfu_g, dtype=float)
    if acc_rfu_g.shape[-1] != 3:
        raise ValueError("acc_rfu_g must have shape (N,3) or equivalent.")
    acc_body_g = rfu_to_frd(acc_rfu_g)
    acc_body_mps2 = acc_body_g * float(cfg.g0)

    # 3) 角速度：RFU -> FRD，deg/s -> rad/s
    gyro_rfu_dps = np.asarray(gyro_rfu_dps, dtype=float)
    if gyro_rfu_dps.shape[-1] != 3:
        raise ValueError("gyro_rfu_dps must have shape (N,3) or equivalent.")
    gyro_body_dps = rfu_to_frd(gyro_rfu_dps)
    gyro_body_rad_s = np.deg2rad(gyro_body_dps)

    # 4) 姿态角：deg -> rad，yaw 源选择 + 符号修正 + wrap
    # 4.1 roll / pitch
    roll_deg = np.asarray(angx_deg, dtype=float).reshape(-1)
    pitch_deg = np.asarray(angy_deg, dtype=float).reshape(-1)

    # 可选：如果你怀疑 roll/pitch 也有 NaN，可以一样插值
    if np.isnan(roll_deg).any():
        roll_deg = _interp_nan_on_time(t_s, roll_deg, name="roll_deg")
    if np.isnan(pitch_deg).any():
        pitch_deg = _interp_nan_on_time(t_s, pitch_deg, name="pitch_deg")

    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)

    # 4.2 yaw 源选择
    yaw_deg_raw, yaw_src = _select_yaw_deg(yawdeg, angz_deg, cfg)
    yaw_deg = np.asarray(yaw_deg_raw, dtype=float).reshape(-1)

    # 关键：这里对 yaw 的 NaN 做插值补齐
    if np.isnan(yaw_deg).any():
        yaw_deg = _interp_nan_on_time(t_s, yaw_deg, name=f"yaw_deg({yaw_src})")

    yaw_rad_raw = np.deg2rad(yaw_deg)
    yaw_rad = cfg.yaw_sign * yaw_rad_raw
    yaw_rad = wrap_pm_pi(yaw_rad)
    return {
        "t_s": t_s,
        "dt_s": dt_s,
        "dt_med": dt_med,
        "acc_body_mps2": acc_body_mps2,
        "gyro_body_rad_s": gyro_body_rad_s,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "yaw_source_used": yaw_src,
    }

