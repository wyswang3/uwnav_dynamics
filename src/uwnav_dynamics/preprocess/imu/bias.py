from __future__ import annotations

"""
uwnav_dynamics.preprocess.imu.bias

IMU 零偏估计（bias estimation）与随机游走等噪声指标的工程估计模块。

一、模块目标

在已经完成「坐标统一（FRD）+ 单位统一（m/s², rad/s）+ 重力补偿」之后，
本模块在一段「静止窗」内对 IMU 的输出进行统计，估计出：

  - 加速度零偏：b_a  (3,)
  - 陀螺零偏：  b_g  (3,)
  - 静止噪声标准差：std_a / std_g
  - 随机游走（简单工程估计）：rw_a / rw_g
  - 其他统计量：样本数、时间长度、窗口 dt 中位数等

注意：
  - 本模块仅做统计，不做任何滤波。
  - 真正「去 bias」的操作由 apply_bias(...) 执行。
  - 随机游走估计为「工程近似」，非严格 Allan 方差分析。

二、输入数据假设

输入加速度与角速度为：

  - a_lin_body_mps2 : (N,3)
      体坐标系 FRD 下的「线加速度 + bias」：
        a_lin_b_plus_bias ≈ a_lin_b + b_a
      其中重力分量已经在 gravity.py 中减去。

  - gyro_body_rad_s : (N,3)
      体坐标系 FRD 下的角速度测量，单位 rad/s。

  - t_s : (N,)
      时间戳（秒），按时间递增（单调不减）。

在「静止窗」内假设：
  - a_lin_b ≈ 0
  - gyro ≈ 0

则可以通过平均值估计零偏：
  - b_a ≈ mean(a_lin_body_mps2)
  - b_g ≈ mean(gyro_body_rad_s)

三、随机游走估计（工程简化版）

我们在静止窗内计算噪声标准差 std：

  - std_a : (3,)  加速度噪声（m/s²）
  - std_g : (3,)  角速度噪声（rad/s）

令 dt 为静止窗内样本间隔的中位数，单位秒。

则给出一个简单、可用于滤波器调参的随机游走估计：

  - 加速度随机游走（white-noise 等效）：rw_a ≈ std_a * sqrt(dt)
  - 陀螺随机游走：               rw_g ≈ std_g * sqrt(dt)

注意：
  - 这是工程上的近似，便于把「离散噪声 std」映射为「连续时间强度」；
  - 想要更精确的随机游走参数，应使用 Allan 方差分析，本项目后续可单独实现。

四、模块职责

本模块提供两个主要接口：
  - estimate_bias(...) : 在静止窗内估计 b_a, b_g 以及噪声统计；
  - apply_bias(...)    : 用估计得到的 b_a, b_g 对整个序列去除零偏。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ImuBiasConfig:
    """
    IMU 零偏估计配置。

    属性
    ----------
    bias_window_s : float
        静止窗时间长度（秒）。默认使用 [t0, t0 + bias_window_s] 作为静止窗。
        通常选在实验刚开始、机器人尚未运动的阶段（例如前 20 s）。

    min_samples : int
        静止窗内用于统计的最小样本数。如果窗口内样本数量少于该值，
        可以选择报错 / 回退到使用全局数据等策略（本模块采用回退策略）。
    """
    bias_window_s: float = 20.0
    min_samples: int = 200


def _select_bias_window_mask(
    t_s: np.ndarray,
    cfg: ImuBiasConfig,
) -> Tuple[np.ndarray, float]:
    """
    根据配置从时间序列中选择静止窗掩码（mask）。

    默认策略：
      - 使用 [t0, t0 + bias_window_s] 作为静止窗；
      - 若 bias_window_s <= 0，则使用全局窗口；
      - 若窗口内样本数 < min_samples，则回退为「全局窗口」。

    返回
    ----------
    mask : np.ndarray
        布尔数组，形状 (N,)，True 表示属于静止窗。
    window_len_s : float
        最终选取的静止窗时间长度（秒）。
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("t_s is empty.")

    t0 = float(t[0])
    t1 = float(t[-1])

    # 默认使用 [t0, t0 + bias_window_s]
    if cfg.bias_window_s > 0.0:
        t_end = t0 + float(cfg.bias_window_s)
        # 若 bias_window 超出数据末尾，则截到 t1
        if t_end > t1:
            t_end = t1
        mask = (t >= t0) & (t <= t_end)
        window_len = t_end - t0
    else:
        # bias_window_s <= 0：使用全局窗口
        mask = np.ones_like(t, dtype=bool)
        window_len = t1 - t0

    n_win = int(mask.sum())

    # 如果样本数不足，则回退到「全局窗口」
    if n_win < cfg.min_samples:
        mask = np.ones_like(t, dtype=bool)
        window_len = t1 - t0

    return mask, float(window_len)


def estimate_bias(
    t_s: np.ndarray,
    a_lin_body_mps2: np.ndarray,
    gyro_body_rad_s: np.ndarray,
    cfg: ImuBiasConfig,
) -> Dict[str, np.ndarray | float | int | bool]:
    """
    在静止窗内估计 IMU 的零偏与噪声统计（工程版），并给出随机游走近似。

    输入
    ----------
    t_s : np.ndarray
        时间戳（秒），形状 (N,)，需单调递增。
    a_lin_body_mps2 : np.ndarray
        体坐标系 FRD 下的线加速度 + bias，形状 (N,3)，单位 [m/s²]。
        通常为 gravity.py 输出的 a_lin_body_mps2。
    gyro_body_rad_s : np.ndarray
        体坐标系 FRD 下的角速度测量，形状 (N,3)，单位 [rad/s]。

    cfg : ImuBiasConfig
        静止窗选择与最小样本数配置。

    返回
    ----------
    dict
        主要字段说明：

        - "mask_window"          : (N,) bool，时间静止窗掩码
        - "n_window"             : int，静止窗内「有效样本数」（去掉 NaN 后）
        - "t_window_s"           : float，静止窗时间长度（秒）
        - "dt_med_window"        : float，静止窗内 dt 中位数（秒）
        - "ba_body_mps2"         : (3,) 加速度零偏估计
        - "bg_body_rad_s"        : (3,) 陀螺零偏估计
        - "std_a_body_mps2"      : (3,) 静止窗内加速度噪声标准差
        - "std_g_body_rad_s"     : (3,) 静止窗内陀螺噪声标准差
        - "rw_a_body_mps2_sqrt_s": (3,) 加速度随机游走近似（m/s² * sqrt(s)）
        - "rw_g_body_rad_s_sqrt_s":(3,) 陀螺随机游走近似（rad/s * sqrt(s)）
        - "valid"                : bool，表示估计是否有效
    """
    # ---- 0) 基本检查与数组化 ----
    t = np.asarray(t_s, dtype=float).reshape(-1)
    acc = np.asarray(a_lin_body_mps2, dtype=float)
    gyro = np.asarray(gyro_body_rad_s, dtype=float)

    if acc.shape != gyro.shape:
        raise ValueError("a_lin_body_mps2 and gyro_body_rad_s must have the same shape.")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("a_lin_body_mps2 must have shape (N,3).")

    N = t.size
    if N != acc.shape[0]:
        raise ValueError("t_s length must match a_lin_body_mps2 length.")

    if N < cfg.min_samples:
        raise ValueError(
            f"Not enough IMU samples for bias estimation: N={N} < min_samples={cfg.min_samples}"
        )

    # ---- 1) 按时间选择静止窗（可能包含 NaN）----
    mask_time, window_len_s = _select_bias_window_mask(t, cfg)
    n_win_time = int(mask_time.sum())
    if n_win_time < cfg.min_samples:
        raise ValueError(
            f"Bias window has too few samples: n_window={n_win_time} < min_samples={cfg.min_samples}"
        )

    t_win = t[mask_time]
    acc_win = acc[mask_time, :]
    gyro_win = gyro[mask_time, :]

    # ---- 2) 在静止窗内进一步剔除 NaN / inf ----
    finite_acc = np.isfinite(acc_win).all(axis=1)
    finite_gyro = np.isfinite(gyro_win).all(axis=1)
    mask_finite = finite_acc & finite_gyro

    t_win2 = t_win[mask_finite]
    acc_win2 = acc_win[mask_finite, :]
    gyro_win2 = gyro_win[mask_finite, :]

    n_win_valid = int(mask_finite.sum())
    if n_win_valid < cfg.min_samples:
        raise ValueError(
            f"Bias window valid samples too few after dropping NaNs: "
            f"n_window_valid={n_win_valid} < min_samples={cfg.min_samples}"
        )

    # ---- 3) 静止窗内 dt 中位数 ----
    if t_win2.size >= 2:
        dt_win = np.diff(t_win2)
        dt_win_med = float(np.median(dt_win))
    else:
        dt_win_med = float("nan")

    # ---- 4) 零偏估计（仅用 finite 样本）----
    ba = acc_win2.mean(axis=0)   # (3,)
    bg = gyro_win2.mean(axis=0)  # (3,)

    # ---- 5) 噪声标准差 ----
    std_a = acc_win2.std(axis=0, ddof=1)
    std_g = gyro_win2.std(axis=0, ddof=1)

    # ---- 6) 随机游走近似：std * sqrt(dt_med_window) ----
    if np.isfinite(dt_win_med) and dt_win_med > 0.0:
        sqrt_dt = np.sqrt(dt_win_med)
        rw_a = std_a * sqrt_dt
        rw_g = std_g * sqrt_dt
    else:
        rw_a = np.full(3, np.nan, dtype=float)
        rw_g = np.full(3, np.nan, dtype=float)

    # ---- 7) 返回结果 ----
    return {
        "mask_window": mask_time,                 # 时间静止窗掩码
        "n_window": n_win_valid,                  # 有效样本数（去掉 NaN 后）
        "t_window_s": window_len_s,
        "dt_med_window": dt_win_med,
        "ba_body_mps2": ba,
        "bg_body_rad_s": bg,
        "std_a_body_mps2": std_a,
        "std_g_body_rad_s": std_g,
        "rw_a_body_mps2_sqrt_s": rw_a,
        "rw_g_body_rad_s_sqrt_s": rw_g,
        "valid": True,
    }


def apply_bias(
    a_lin_body_mps2: np.ndarray,
    gyro_body_rad_s: np.ndarray,
    ba_body_mps2: np.ndarray,
    bg_body_rad_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用估计得到的零偏，对整段 IMU 序列进行去 bias 处理。

    输入
    ----------
    a_lin_body_mps2 : np.ndarray
        体坐标系 FRD 下的线加速度 + bias，形状 (N,3)，单位 [m/s²]。
    gyro_body_rad_s : np.ndarray
        体坐标系 FRD 下的角速度 + bias，形状 (N,3)，单位 [rad/s]。
    ba_body_mps2 : np.ndarray
        加速度零偏估计 (3,)。
    bg_body_rad_s : np.ndarray
        陀螺零偏估计 (3,)。

    返回
    ----------
    a_corr : np.ndarray
        去 bias 后的线加速度，形状 (N,3)。
    gyro_corr : np.ndarray
        去 bias 后的角速度，形状 (N,3)。
    """
    acc = np.asarray(a_lin_body_mps2, dtype=float)
    gyro = np.asarray(gyro_body_rad_s, dtype=float)
    ba = np.asarray(ba_body_mps2, dtype=float).reshape(1, 3)
    bg = np.asarray(bg_body_rad_s, dtype=float).reshape(1, 3)

    if acc.shape != gyro.shape:
        raise ValueError("a_lin_body_mps2 and gyro_body_rad_s must have the same shape.")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("a_lin_body_mps2 must have shape (N,3).")

    a_corr = acc - ba   # 广播减法
    gyro_corr = gyro - bg

    return a_corr, gyro_corr
