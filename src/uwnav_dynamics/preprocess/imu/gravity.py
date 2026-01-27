from __future__ import annotations

"""
uwnav_dynamics.preprocess.imu.gravity

IMU 重力补偿模块：在统一的坐标与单位约定下，从「比力测量」中减去重力分量，
得到用于动力学建模与控制的线加速度。

一、坐标系与旋转约定（与 transform.py 保持一致）

1) 体坐标系（body frame）
   - 采用 FRD（Forward–Right–Down）右手坐标系：
     - X_b：向前（Forward）
     - Y_b：向右（Right）
     - Z_b：向下（Down）

   IMU 在 transform.py 中已经完成：
     - RFU -> FRD 坐标变换
     - 单位转换：g -> m/s²
   因此本模块的输入加速度 acc_body_mps2 已经在 FRD 下。

2) 导航坐标系（navigation frame）
   - 采用 ENU（East–North–Up）右手坐标系：
     - X_n：East（东）
     - Y_n：North（北）
     - Z_n：Up（上）

   姿态角 roll / pitch / yaw 的定义与 ENU 保持一致：
     - yaw：绕 Z_n（Up）轴旋转，正方向通常为从 East 指向 North 的逆时针旋转；
     - pitch：绕中间坐标的 Y 轴；
     - roll：绕中间坐标的 X 轴；
   采用标准 ZYX 欧拉角顺序：
     R_nb = R_z(yaw) * R_y(pitch) * R_x(roll)

   其中：
     - R_nb：body -> nav，把体坐标向量旋转到 ENU 下
     - R_bn = R_nb^T：nav -> body，把 ENU 向量旋转到体坐标

二、加速度计输出与重力补偿

1) 加速度计输出的物理语义
   绝大多数 IMU 的加速度输出是比力（specific force），在体坐标系下满足：
     a_meas_b = a_lin_b + g_b + b_a + noise

   其中：
     - a_lin_b：线加速度（真正用于动力学的项）
     - g_b    ：重力在体坐标系下的分量
     - b_a    ：加速度计零偏（bias）
     - noise  ：测量噪声

   在本模块中，我们假设 bias 尚未去除（由 bias.py 处理），
   只负责从 a_meas_b 中减去 g_b，得到：
     a_lin_b_plus_bias = a_meas_b - g_b ≈ a_lin_b + b_a

2) 全局重力向量
   在 ENU 坐标系中，我们采用统一约定：
     g_n = [0, 0, -g0]^T

   注意：
     - Z_n 轴向上（Up），因此重力指向 -Z_n；
     - g0 通常取 9.78 或 9.81（可在配置中调整）。

3) 体坐标系下重力分量
   已知：
     - g_n in ENU
     - R_nb：body -> nav
   则 nav -> body 的旋转为：
     R_bn = R_nb^T

   重力在体坐标系下为：
     g_b = R_bn * g_n

   最终线加速度（仍含 bias）为：
     a_lin_b_plus_bias = a_meas_b - g_b

三、模块职责

本模块只做两件事：
  - 提供 rpy_to_R_nb：根据 (roll, pitch, yaw) 生成 R_nb 序列；
  - 提供 compensate_gravity：给定体坐标加速度 + 姿态，计算 g_b 与 a_lin_b_plus_bias。

不做：
  - bias 估计与去除（由 bias.py 处理）
  - 滤波与去毛刺（由 filter.py 处理）
"""

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
