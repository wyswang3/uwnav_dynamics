from __future__ import annotations

"""
uwnav_dynamics.preprocess.imu.filter

IMU 滤波模块：在完成「坐标统一（FRD）+ 单位统一（m/s², rad/s）+
重力补偿 + 零偏估计」之后，对信号进行去毛刺 + 低通滤波。

一、输入信号的工程语义

本模块假定输入已经经过前面几个步骤：

  1) transform.py:
     - RFU -> FRD（体坐标系）
     - g -> m/s², deg/s -> rad/s
     - yaw -> rad 且 wrap 到 (-π, π]

  2) gravity.py:
     - 从测量比力中减去重力分量，得到 a_lin_body_mps2

  3) bias.py:
     - 在静止窗内估计出零偏 b_a, b_g
     - apply_bias(...) 得到「去 bias 后」的线加速度与角速度

因此本模块的输入具有如下物理含义：

  - a_lin_body_mps2 : (N,3)
        体坐标系 FRD 下的线加速度，单位 [m/s²]，
        已经去重力、去零偏，仍然包含高频噪声和偶发毛刺。

  - gyro_body_rad_s : (N,3)
        体坐标系 FRD 下的角速度，单位 [rad/s]，
        已经去零偏，仍然包含高频噪声和偶发毛刺。

  - yaw_rad : (N,)
        体坐标系姿态的 yaw 角，单位 [rad]，
        已由 transform.py 根据 ENU 语义处理过并 wrap 到 (-π, π]。

本模块只做「信号质量提升」，不改变任何物理模型假设。

二、滤波策略

1) 去毛刺（despike）

  - 工程中常见的单点离群值（spike），通常源自串口抖动、
    数值溢出、偶发解析异常等，不代表真正物理运动。
  - 简化策略：对每个标量序列 x[k]：
      若 |x[k] - x[k-1]| > 阈值 thresh，则认为 x[k] 是毛刺，用 x[k-1] 替代。

    优点：
      - 实现简单、可解释；
      - 能有效去除单点极大跳变。

    阈值需要按物理量分别设置（加速度 / 角速度 / yaw）。

2) 一阶 IIR 低通滤波

  - 对离散信号 x[k]，给定采样周期 dt、中值：
      alpha = 2π fc dt / (1 + 2π fc dt)

      y[k] = y[k-1] + alpha * (x[k] - y[k-1])

    其中 fc 为截止频率，按工程经验配置。
    这个形式对应一阶连续时间低通在离散域的近似。

  - 对于 yaw，需要注意「2π wrap」问题：
      - 我们先对 yaw 做 unwrap（消除跳变），
      - 在 unwrap 后的连续角度上低通，
      - 最后再 wrap 回 (-π, π]。

三、坐标与单位的保持

  - 本模块不进行坐标变换：
      输入 / 输出均在体坐标系 FRD 下。
  - 本模块不改变单位：
      加速度保持 [m/s²]，角速度保持 [rad/s]，角度保持 [rad]。

四、模块职责

  - 提供基础工具：
      - despike_stepwise(x, thresh)
      - iir_lowpass(x, dt, fc)

  - 提供主接口：
      filter_signals(dt_med, a_lin_body_mps2, gyro_body_rad_s, yaw_rad, cfg)

    统一对线加速度 / 角速度 / yaw 进行去毛刺 + 低通，
    并返回滤波结果及毛刺统计。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ImuFilterConfig:
    """
    IMU 滤波配置。

    属性
    ----------
    cutoff_acc_hz : float
        线加速度通道的一阶低通滤波截止频率 [Hz]。
        例如 3~5 Hz 可抑制高频噪声，又保留运动动态。

    cutoff_gyro_hz : float
        角速度通道的一阶低通截止频率 [Hz]。

    cutoff_yaw_hz : float
        yaw 角的一阶低通截止频率 [Hz]。
        通常可选更低的值（例如 1~2 Hz），用来平滑姿态。

    spike_acc_mps2 : float
        加速度单步差分的毛刺阈值 [m/s²]。
        若 |a[k] - a[k-1]| > spike_acc_mps2，即视为毛刺。

    spike_gyro_rad_s : float
        角速度单步差分的毛刺阈值 [rad/s]。

    spike_yaw_rad : float
        yaw 单步差分的毛刺阈值 [rad]（在 unwrap 之前会先进行简单差分检查）。
    """
    cutoff_acc_hz: float = 5.0
    cutoff_gyro_hz: float = 8.0
    cutoff_yaw_hz: float = 2.0

    spike_acc_mps2: float = 3.0
    spike_gyro_rad_s: float = 1.0
    spike_yaw_rad: float = 0.5


def _unwrap_pm_pi(a_rad: np.ndarray) -> np.ndarray:
    """
    对角度序列做 unwrap，去除跨越 ±π 时的 2π 跳变。

    输入 a_rad 通常在 (-π, π]，本函数生成一个连续的角度序列。
    """
    a = np.asarray(a_rad, dtype=float).reshape(-1)
    if a.size == 0:
        return a

    out = a.copy()
    for i in range(1, a.size):
        da = out[i] - out[i - 1]
        # 若差分过大，说明跨越了 ±π 边界，需要补偿 2π
        if da > np.pi:
            out[i:] -= 2.0 * np.pi
        elif da < -np.pi:
            out[i:] += 2.0 * np.pi
    return out


def _wrap_pm_pi(a_rad: np.ndarray) -> np.ndarray:
    """
    将角度 wrap 到 (-π, π]。
    """
    a = np.asarray(a_rad, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def despike_stepwise(x: np.ndarray, thresh: float) -> Tuple[np.ndarray, int]:
    """
    简单单步差分去毛刺。

    对标量序列 x[k]：
      - 若 |x[k] - x[k-1]| > thresh，则视为毛刺：
            x_clean[k] = x_clean[k-1]

    参数
    ----------
    x : np.ndarray
        输入序列，形状 (N,)。
    thresh : float
        单步差分阈值。

    返回
    ----------
    x_clean : np.ndarray
        去毛刺后的序列。
    n_spikes : int
        被判定为毛刺并替换的样本个数。
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    N = arr.size
    if N <= 1 or thresh <= 0.0:
        return arr.copy(), 0

    out = arr.copy()
    n_spikes = 0
    for i in range(1, N):
        if abs(out[i] - out[i - 1]) > thresh:
            out[i] = out[i - 1]
            n_spikes += 1
    return out, n_spikes


def iir_lowpass(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """
    一阶 IIR 低通滤波器（标量序列）。

    连续时间一阶低通系统：
      H(s) = 1 / (1 + s / (2π fc))

    离散化近似（常见工程形式）：
      alpha = 2π fc dt / (1 + 2π fc dt)
      y[k] = y[k-1] + alpha * (x[k] - y[k-1])

    参数
    ----------
    x : np.ndarray
        输入序列，形状 (N,)。
    dt : float
        采样周期（秒）。建议使用中位 dt。
    cutoff_hz : float
        截止频率 fc（Hz）。若 <= 0，则直接返回原序列。

    返回
    ----------
    y : np.ndarray
        滤波后的序列，形状 (N,)。
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    N = arr.size
    if N == 0 or cutoff_hz <= 0.0 or dt <= 0.0:
        return arr.copy()

    omega_c = 2.0 * np.pi * float(cutoff_hz)
    alpha = (omega_c * dt) / (1.0 + omega_c * dt)

    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, N):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


def filter_signals(
    dt_med: float,
    a_lin_body_mps2: np.ndarray,
    gyro_body_rad_s: np.ndarray,
    yaw_rad: np.ndarray,
    cfg: ImuFilterConfig,
) -> Dict[str, np.ndarray | Dict[str, int]]:
    """
    对去重力 / 去 bias 的 IMU 信号进行去毛刺与一阶低通滤波。

    处理步骤：

      1) 对每个通道分别做去毛刺（despike_stepwise）：
           - 加速度三个轴：阈值 spike_acc_mps2
           - 角速度三个轴：阈值 spike_gyro_rad_s
           - yaw 角        ：阈值 spike_yaw_rad （在 unwrap 之前做简单检查）

      2) 对去毛刺后的信号做一阶 IIR 低通：
           - 加速度 cutoff_acc_hz
           - 角速度 cutoff_gyro_hz
           - yaw   cutoff_yaw_hz

      3) yaw 处理时先做 unwrap，再低通，最后 wrap 回 (-π, π]。

    输入
    ----------
    dt_med : float
        采样周期中位数（秒），通常来自 transform_raw 的 dt_med。
    a_lin_body_mps2 : np.ndarray
        FRD 体坐标线加速度（去重力 + 去 bias），形状 (N,3)，单位 [m/s²]。
    gyro_body_rad_s : np.ndarray
        FRD 体坐标角速度（去 bias），形状 (N,3)，单位 [rad/s]。
    yaw_rad : np.ndarray
        yaw 角（rad），形状 (N,)，当前在 (-π, π] 范围。

    cfg : ImuFilterConfig
        滤波与去毛刺参数配置。

    返回
    ----------
    dict
        主要字段：

        - "a_lin_body_mps2_filt"  : (N,3) 滤波后的线加速度
        - "gyro_body_rad_s_filt"  : (N,3) 滤波后的角速度
        - "yaw_rad_filt"          : (N,)  滤波后的 yaw（rad, wrap 到 (-π, π]）
        - "despike_counts"        : dict  各通道的毛刺替换计数
              {
                "acc_x": int,
                "acc_y": int,
                "acc_z": int,
                "gyro_x": int,
                "gyro_y": int,
                "gyro_z": int,
                "yaw": int,
              }
    """
    acc = np.asarray(a_lin_body_mps2, dtype=float)
    gyro = np.asarray(gyro_body_rad_s, dtype=float)
    yaw = np.asarray(yaw_rad, dtype=float).reshape(-1)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("a_lin_body_mps2 must have shape (N,3).")
    if gyro.shape != acc.shape:
        raise ValueError("gyro_body_rad_s must have the same shape as a_lin_body_mps2.")
    if yaw.shape[0] != acc.shape[0]:
        raise ValueError("yaw_rad length must match a_lin_body_mps2 length.")

    N = acc.shape[0]
    if N <= 1:
        # 样本太少，直接返回原值
        return {
            "a_lin_body_mps2_filt": acc.copy(),
            "gyro_body_rad_s_filt": gyro.copy(),
            "yaw_rad_filt": yaw.copy(),
            "despike_counts": {
                "acc_x": 0,
                "acc_y": 0,
                "acc_z": 0,
                "gyro_x": 0,
                "gyro_y": 0,
                "gyro_z": 0,
                "yaw": 0,
            },
        }

    despike_counts = {
        "acc_x": 0,
        "acc_y": 0,
        "acc_z": 0,
        "gyro_x": 0,
        "gyro_y": 0,
        "gyro_z": 0,
        "yaw": 0,
    }

    # 1) 加速度去毛刺
    acc_f = np.empty_like(acc)
    for i, key in enumerate(("acc_x", "acc_y", "acc_z")):
        acc_f[:, i], n_spk = despike_stepwise(acc[:, i], cfg.spike_acc_mps2)
        despike_counts[key] = n_spk

    # 2) 角速度去毛刺
    gyro_f = np.empty_like(gyro)
    for i, key in enumerate(("gyro_x", "gyro_y", "gyro_z")):
        gyro_f[:, i], n_spk = despike_stepwise(gyro[:, i], cfg.spike_gyro_rad_s)
        despike_counts[key] = n_spk

    # 3) yaw 去毛刺（在 wrap 域做简单检查），之后再 unwrap + 低通
    yaw_despike, n_spk_yaw = despike_stepwise(yaw, cfg.spike_yaw_rad)
    despike_counts["yaw"] = n_spk_yaw

    # yaw unwrap -> 低通 -> wrap 回 (-π, π]
    yaw_unwrap = _unwrap_pm_pi(yaw_despike)
    yaw_low = iir_lowpass(yaw_unwrap, dt_med, cfg.cutoff_yaw_hz)
    yaw_filt = _wrap_pm_pi(yaw_low)

    # 4) 一阶低通：对加速度 / 角速度三个轴分别处理
    for i in range(3):
        acc_f[:, i] = iir_lowpass(acc_f[:, i], dt_med, cfg.cutoff_acc_hz)
        gyro_f[:, i] = iir_lowpass(gyro_f[:, i], dt_med, cfg.cutoff_gyro_hz)

    return {
        "a_lin_body_mps2_filt": acc_f,
        "gyro_body_rad_s_filt": gyro_f,
        "yaw_rad_filt": yaw_filt,
        "despike_counts": despike_counts,
    }
