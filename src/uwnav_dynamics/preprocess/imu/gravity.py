"""
模块名称：IMU 重力补偿

模块职责：
依据统一的 ENU/FRD 坐标约定，将姿态角转换为旋转矩阵，
并把重力分量从 IMU 比力测量中剥离出来，得到线加速度。

主要功能：
1. 根据 roll/pitch/yaw 生成 body 到 nav 的旋转矩阵序列。
2. 将 ENU 中的重力向量转换到体坐标系。
3. 计算去重力后的线加速度和相关诊断量。

数据流：
`transform` 输出的体坐标加速度与姿态角
    -> 旋转矩阵计算
    -> 体坐标重力分量求解
    -> 去重力线加速度
    -> `preprocess.imu.bias`

依赖模块：
1. `numpy`
2. `dataclasses`

备注：
本模块只处理几何与重力项，不负责零偏估计和滤波。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ImuGravityConfig:
    """
    IMU 重力补偿配置。

    属性
    ----------
    g0 : float
        重力加速度标称值，用于在导航坐标系 ENU 下构造 g_n = [0, 0, -g0]^T。
    """
    g0: float = 9.78


def rpy_to_R_nb(
    roll_rad: np.ndarray,
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
) -> np.ndarray:
    """
    根据欧拉角 (roll, pitch, yaw) 生成 body->nav 的旋转矩阵 R_nb。

    欧拉角与旋转顺序约定：
      - 坐标系：nav = ENU, body = FRD
      - 使用 ZYX 顺序：
          R_nb = R_z(yaw) * R_y(pitch) * R_x(roll)

        其中：
          - R_x(roll)  : 绕 X 轴旋转（roll）
          - R_y(pitch) : 绕 Y 轴旋转（pitch）
          - R_z(yaw)   : 绕 Z 轴旋转（yaw）

      - 方向：采用右手定则，正角度为右手定则正向。

    参数
    ----------
    roll_rad : np.ndarray
        roll 角（rad），形状 (N,)。
    pitch_rad : np.ndarray
        pitch 角（rad），形状 (N,)。
    yaw_rad : np.ndarray
        yaw 角（rad），形状 (N,)。

    返回
    ----------
    R_nb : np.ndarray
        形状 (N, 3, 3) 的旋转矩阵数组，
        对于任意体坐标向量 v_b，有：
          v_n = R_nb[i] @ v_b
    """
    roll = np.asarray(roll_rad, dtype=float).reshape(-1)
    pitch = np.asarray(pitch_rad, dtype=float).reshape(-1)
    yaw = np.asarray(yaw_rad, dtype=float).reshape(-1)

    if not (roll.shape == pitch.shape == yaw.shape):
        raise ValueError("roll_rad, pitch_rad, yaw_rad must have the same shape.")

    N = roll.shape[0]
    R_nb = np.empty((N, 3, 3), dtype=float)

    for i in range(N):
        cr = np.cos(roll[i])
        sr = np.sin(roll[i])
        cp = np.cos(pitch[i])
        sp = np.sin(pitch[i])
        cy = np.cos(yaw[i])
        sy = np.sin(yaw[i])

        # R_x(roll)
        R_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=float,
        )

        # R_y(pitch)
        R_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=float,
        )

        # R_z(yaw)
        R_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # ZYX: R_nb = R_z * R_y * R_x
        R_nb[i] = R_z @ R_y @ R_x

    return R_nb


def compensate_gravity(
    acc_body_mps2: np.ndarray,
    roll_rad: np.ndarray,
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
    cfg: ImuGravityConfig,
) -> Dict[str, np.ndarray]:
    """
    对体坐标系下的加速度测量进行重力补偿，得到线加速度（仍含 bias）。

    输入
    ----------
    acc_body_mps2 : np.ndarray
        体坐标系 FRD 下的加速度测量，形状 (N, 3)，单位 [m/s²]。
        通常为 transform.py 输出的 acc_body_mps2，尚未减去重力。

    roll_rad : np.ndarray
        roll 角（rad），形状 (N,)。
    pitch_rad : np.ndarray
        pitch 角（rad），形状 (N,)。
    yaw_rad : np.ndarray
        yaw 角（rad），形状 (N,)。已根据 ENU 语义与 yaw_sign 处理。

    cfg : ImuGravityConfig
        重力补偿配置，主要提供 g0。

    返回
    ----------
    dict
        包含以下键值：
          - "R_nb"               : (N,3,3)  body->nav 旋转矩阵
          - "g_body_mps2"        : (N,3)    体坐标系下的重力向量 g_b
          - "a_lin_body_mps2"    : (N,3)    去重力后的线加速度（仍含 bias）

        其中：
          g_n = [0, 0, -g0]^T
          R_bn = R_nb^T
          g_b[i] = R_bn[i] @ g_n
          a_lin_body[i] = acc_body_mps2[i] - g_b[i]
    """
    acc_body = np.asarray(acc_body_mps2, dtype=float)
    if acc_body.ndim != 2 or acc_body.shape[1] != 3:
        raise ValueError("acc_body_mps2 must have shape (N,3).")

    N = acc_body.shape[0]

    # 1) 计算 R_nb 序列
    R_nb = rpy_to_R_nb(roll_rad, pitch_rad, yaw_rad)
    if R_nb.shape[0] != N:
        raise ValueError("R_nb length must match acc_body_mps2 length.")

    # 2) 在 ENU 下的重力向量 g_n
    g_n = np.array([0.0, 0.0, -float(cfg.g0)], dtype=float)  # [E, N, U]

    # 3) 计算体坐标系下的重力 g_b
    g_body = np.empty_like(acc_body)
    for i in range(N):
        R_bn = R_nb[i].T  # nav -> body
        g_body[i] = R_bn @ g_n

    # 4) 线加速度 = 测量 - 重力
    a_lin_body = acc_body - g_body

    return {
        "R_nb": R_nb,
        "g_body_mps2": g_body,
        "a_lin_body_mps2": a_lin_body,
    }
