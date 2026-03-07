# src/uwnav_dynamics/preprocess/align/aligner.py
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
模块名称：多传感器主时间轴对齐

模块职责：
负责将 IMU、PWM、DVL、Power 等异步传感器数据对齐到统一主时间轴，
生成训练与评估共用的 `train_base.csv` 基础表。

主要功能：
1. 根据 IMU / PWM 时间范围解析主时间轴。
2. 将 IMU dense 观测、PWM 控制、DVL 稀疏监督与 Power 辅助量映射到主轴。
3. 输出可直接进入 dataset build 的训练基础表。

数据流：
IMU proc CSV / PWM CSV / DVL proc CSV / Power CSV
    ↓
主时间轴解析
    ↓
IMU dense 重采样 + PWM hold-last + DVL sparse attach + Power gating
    ↓
train_base.csv / DataFrame

依赖模块：
- numpy
- pandas
- pathlib
- dataclasses

备注：
- IMU Acc/Gyro 属于 dense 连续信号，必须保持主轴对齐结果全 finite；
- DVL / Power 仍按各自的稀疏监督与辅助量语义处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

from uwnav_dynamics.preprocess.qa import (
    assert_train_base_qa_pass,
    render_train_base_qa,
    run_train_base_qa,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class AlignConfig:
    """
    多频率对齐配置（面向控制的训练用）：

      - 主时间轴：默认以 IMU 时间轴为参考（常见 100 Hz）
      - 主时间区间：IMU 与 PWM 的有效交集（可加 margin）
      - DVL：作为稀疏监督，仅在接近主时间点时写入，其余为 NaN
      - Power：低频辅助量，hold-last + 时间窗 gating 对齐到主时间轴

    通常在 configs/preprocess/alignment.yaml 或
    configs/dataset/<name>.yaml 中 YAML 化。
    """
    # 主时间步长（当与 IMU 采样间隔接近时会直接复用 IMU 时间轴）
    dt_main_s: float = 0.02

    # 裁剪边缘：避免刚启动/关停阶段的奇怪数据
    t_margin_s: float = 0.0

    # 可选：若提供已有主时间轴 CSV，则优先使用该主轴
    # 例如某些实验已经离线生成统一 t_s，可避免二次重建主轴。
    main_axis_csv: Optional[str] = None
    main_time_col: str = "t_s"

    # DVL 对齐的允许时间误差（主时间轴最近邻）
    dvl_max_dt_s: float = 0.10   # 10 Hz → 0.1 s；默认一整个周期

    # Power 对齐的允许时间误差（若需要严格限制）
    power_max_dt_s: float = 0.25

    # 是否硬要求某些源必须存在
    require_pwm: bool = True
    require_dvl: bool = False
    require_power: bool = False


# =============================================================================
# Helpers: main grid & binning
# =============================================================================

def _compute_main_grid(
    t_imu: np.ndarray,
    t_pwm: np.ndarray,
    cfg: AlignConfig,
) -> np.ndarray:
    """
    根据 IMU / PWM 时间范围生成主时间轴：

      1) 先确定主区间 = [max(t_imu0, t_pwm0) + margin, min(t_imu1, t_pwm1) - margin]
      2) 若 cfg.dt_main_s 与 IMU 采样间隔接近（<=20% 相对误差），
         则直接复用 IMU 时间戳（符合“以 IMU 为主轴”）
      3) 否则回退为等间隔网格（保持历史配置兼容）
    """
    if t_imu.size < 2:
        raise ValueError("[ALIGN] IMU 样本太少，无法构建主时间轴。")
    if t_pwm.size < 2:
        raise ValueError("[ALIGN] PWM 样本太少，无法构建主时间轴。")

    t0 = max(float(t_imu[0]), float(t_pwm[0])) + float(cfg.t_margin_s)
    t1 = min(float(t_imu[-1]), float(t_pwm[-1])) - float(cfg.t_margin_s)

    if t1 <= t0:
        raise ValueError(
            f"[ALIGN] 无有效时间交集：t0={t0:.3f}, t1={t1:.3f}. "
            "请检查 IMU / PWM 日志时间范围。"
        )

    imu_slice = t_imu[(t_imu >= t0) & (t_imu <= t1)]
    if imu_slice.size < 2:
        raise ValueError(
            "[ALIGN] 主区间内 IMU 样本不足，无法构建主时间轴。"
        )

    dt = float(cfg.dt_main_s)
    if dt <= 0.0:
        raise ValueError(f"[ALIGN] dt_main_s must be > 0, got {dt}")

    imu_dt = float(np.median(np.diff(imu_slice)))
    tol = max(1e-6, 0.2 * max(abs(imu_dt), 1e-6))
    if abs(dt - imu_dt) <= tol:
        t_main = imu_slice.astype(float, copy=False)
        print(
            f"[ALIGN] main axis from IMU: t=[{t_main[0]:.3f}, {t_main[-1]:.3f}], "
            f"N={t_main.size}, imu_dt~{imu_dt:.4f}s"
        )
        return t_main

    n = int(np.floor((t1 - t0) / dt)) + 1
    t_main = t0 + dt * np.arange(n, dtype=float)

    print(
        f"[ALIGN] main time-grid (uniform fallback): "
        f"t=[{t_main[0]:.3f}, {t_main[-1]:.3f}], N={t_main.size}, "
        f"cfg_dt={dt:.4f}s, imu_dt~{imu_dt:.4f}s"
    )
    return t_main


def _load_main_grid_from_csv(main_axis_csv: str | Path, time_col: str) -> np.ndarray:
    """
    从已有 CSV 读取主时间轴。
    """
    p = Path(main_axis_csv).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"[ALIGN] main_axis_csv not found: {p}")
    df = pd.read_csv(p)
    if time_col not in df.columns:
        raise KeyError(f"[ALIGN] main_axis_csv missing column {time_col!r}: {p}")
    t_main = df[time_col].to_numpy(dtype=float).reshape(-1)
    if t_main.size < 2:
        raise ValueError(f"[ALIGN] main axis too short: {p}")
    if not np.all(np.isfinite(t_main)):
        raise ValueError(f"[ALIGN] main axis contains NaN/Inf: {p}")
    if np.any(np.diff(t_main) <= 0):
        raise ValueError(f"[ALIGN] main axis must be strictly increasing: {p}")
    print(
        f"[ALIGN] use existing main axis: {p} ({time_col}), "
        f"N={t_main.size}, t=[{t_main[0]:.3f},{t_main[-1]:.3f}]"
    )
    return t_main


def _resolve_main_grid(
    t_imu: np.ndarray,
    t_pwm: np.ndarray,
    cfg: AlignConfig,
) -> np.ndarray:
    """
    主轴解析：
      1) 若配置了 main_axis_csv，则直接读取；
      2) 否则按 IMU/PWM 交集重建网格。
    """
    if cfg.main_axis_csv:
        return _load_main_grid_from_csv(cfg.main_axis_csv, cfg.main_time_col)
    return _compute_main_grid(t_imu=t_imu, t_pwm=t_pwm, cfg=cfg)


def _bin_average_multi(
    t_src: np.ndarray,
    values: np.ndarray,
    t0: float,
    dt: float,
    n_bins: int,
) -> np.ndarray:
    """
    对高频信号做「按主时间轴分箱平均」，用于：

      - IMU 100 Hz 

    参数
    ----
    t_src  : (Ns,) 源时间戳（单调递增）
    values : (Ns, C) 源数据
    t0     : 主时间轴起点
    dt     : 主时间步长
    n_bins : 主时间步数

    返回
    ----
    avg    : (n_bins, C)，没有样本的 bin 填 NaN
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    if vals.shape[0] != t_src.size:
        raise ValueError("[ALIGN] _bin_average_multi: time/value 长度不一致")

    C = vals.shape[1]
    bin_idx = np.floor((t_src - float(t0)) / float(dt)).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid]
    vals = vals[valid]

    if bin_idx.size == 0:
        return np.full((n_bins, C), np.nan, dtype=float)

    out = np.full((n_bins, C), np.nan, dtype=float)
    count = np.bincount(bin_idx, minlength=n_bins).astype(float)

    for c in range(C):
        s = np.bincount(bin_idx, weights=vals[:, c], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, c] = s / np.maximum(count, 1.0)
        out[count == 0, c] = np.nan

    return out


def _assert_main_axis_within_src_range(
    t_src: np.ndarray,
    t_main_s: np.ndarray,
    *,
    src_name: str,
) -> None:
    """
    断言主时间轴完全落在源时间轴覆盖范围内。

    该检查用于 dense 插值前的 fail-fast：
      - 不允许对源时间轴之外的主轴做外推；
      - 不对主轴做静默裁剪，避免悄悄改变实验语义。
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    t_main = np.asarray(t_main_s, dtype=float).reshape(-1)

    if t_src.size == 0:
        raise ValueError(f"[ALIGN][IMU] empty source time axis: src={src_name}")
    if t_main.size == 0:
        raise ValueError(f"[ALIGN][IMU] empty main time axis for src={src_name}")
    if not np.all(np.isfinite(t_src)):
        raise ValueError(f"[ALIGN][IMU] source time axis has NaN/Inf: src={src_name}")
    if not np.all(np.isfinite(t_main)):
        raise ValueError(f"[ALIGN][IMU] main time axis has NaN/Inf: src={src_name}")

    axis_eps_s = 1e-9
    src_lo = float(t_src[0])
    src_hi = float(t_src[-1])
    main_lo = float(t_main[0])
    main_hi = float(t_main[-1])

    if (main_lo < (src_lo - axis_eps_s)) or (main_hi > (src_hi + axis_eps_s)):
        msg = (
            f"[ALIGN][IMU] main axis out of source range: src={src_name} "
            f"t_src=[{src_lo:.9f}, {src_hi:.9f}] "
            f"t_main=[{main_lo:.9f}, {main_hi:.9f}]"
        )
        print(msg)
        raise ValueError(msg)


def _interp_dense_to_main(
    t_src: np.ndarray,
    values: np.ndarray,
    t_main_s: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    将 dense 连续信号按真实时间轴线性插值到主时间轴。

    约束：
      - 源时间轴必须严格递增且无重复；
      - 源值必须全 finite；
      - 不做自动排序、去重、聚合或外推。
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    t_main = np.asarray(t_main_s, dtype=float).reshape(-1)

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.shape[0] != t_src.size:
        raise ValueError(f"[ALIGN][IMU] dense_interp name={name} time/value length mismatch")
    if t_src.size == 0:
        raise ValueError(f"[ALIGN][IMU] dense_interp name={name} empty source time axis")
    if t_main.size == 0:
        raise ValueError(f"[ALIGN][IMU] dense_interp name={name} empty main time axis")
    if not np.all(np.isfinite(t_src)):
        raise ValueError(f"[ALIGN][IMU] dense_interp name={name} source time axis has NaN/Inf")
    if not np.all(np.isfinite(t_main)):
        raise ValueError(f"[ALIGN][IMU] dense_interp name={name} main time axis has NaN/Inf")

    bad_steps = np.where(np.diff(t_src) <= 0.0)[0]
    if bad_steps.size > 0:
        i = int(bad_steps[0])
        raise ValueError(
            "[ALIGN][IMU] dense_interp "
            f"name={name} source time axis must be strictly increasing without duplicates: "
            f"bad_idx={i} t[i]={float(t_src[i]):.9f} t[i+1]={float(t_src[i + 1]):.9f}"
        )

    src_nonfinite = int((~np.isfinite(vals)).sum())
    if src_nonfinite > 0:
        msg = (
            f"[ALIGN][IMU] ERROR dense_interp name={name} "
            f"src_nonfinite={src_nonfinite}"
        )
        print(msg)
        raise RuntimeError(msg)

    out = np.empty((t_main.size, vals.shape[1]), dtype=float)
    for c in range(vals.shape[1]):
        out[:, c] = np.interp(
            t_main,
            t_src,
            vals[:, c],
            left=np.nan,
            right=np.nan,
        )

    out_nonfinite = int((~np.isfinite(out)).sum())
    if out_nonfinite > 0:
        msg = (
            f"[ALIGN][IMU] ERROR dense_interp name={name} "
            f"out_nonfinite={out_nonfinite} "
            f"t_src=[{float(t_src[0]):.9f}, {float(t_src[-1]):.9f}] "
            f"t_main=[{float(t_main[0]):.9f}, {float(t_main[-1]):.9f}]"
        )
        print(msg)
        raise RuntimeError(msg)

    src_dt_med = float(np.median(np.diff(t_src))) if t_src.size > 1 else float("nan")
    main_dt_med = float(np.median(np.diff(t_main))) if t_main.size > 1 else float("nan")
    print(
        f"[ALIGN][IMU] dense_interp name={name} "
        f"src_n={t_src.size} main_n={t_main.size} "
        f"src_dt_med={src_dt_med:.6f} main_dt_med={main_dt_med:.6f} "
        f"src_nonfinite={src_nonfinite} out_nonfinite={out_nonfinite}"
    )
    return out


def _sample_last_before(
    t_src: np.ndarray,
    values: np.ndarray,
    t_main: np.ndarray,
) -> np.ndarray:
    """
    在每个主时间点，对「控制输入」型信号（PWM / Power）做 hold-last：

      u[k] = max{ u_i | t_src[i] <= t_main[k] }

    若在 t_main[k] 之前没有历史样本，则填 NaN。
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    t_main = np.asarray(t_main, dtype=float).reshape(-1)

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    if vals.shape[0] != t_src.size:
        raise ValueError("[ALIGN] _sample_last_before: time/value 长度不一致")

    N = t_main.size
    C = vals.shape[1]
    out = np.full((N, C), np.nan, dtype=float)

    if t_src.size == 0:
        return out

    # 双指针线性扫描，O(N+Ns)
    i = 0
    last = np.full(C, np.nan, dtype=float)
    for k in range(N):
        tk = t_main[k]
        while i < t_src.size and t_src[i] <= tk:
            last = vals[i]
            i += 1
        out[k] = last

    return out


def _sample_last_before_with_max_dt(
    t_src: np.ndarray,
    values: np.ndarray,
    t_main: np.ndarray,
    max_dt: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    hold-last + 最大时间差 gating。

    返回：
      aligned: (N,C)
      mask:    (N,) bool，True 表示当前时刻存在“足够近”的历史样本
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    t_main = np.asarray(t_main, dtype=float).reshape(-1)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.shape[0] != t_src.size:
        raise ValueError("[ALIGN] _sample_last_before_with_max_dt: time/value 长度不一致")

    N = t_main.size
    C = vals.shape[1]
    out = np.full((N, C), np.nan, dtype=float)
    mask = np.zeros(N, dtype=bool)
    if t_src.size == 0:
        return out, mask

    # 浮点时间轴通常来自 np.arange / 插值 / 差分等计算，理论上相同的采样点
    # 可能会表现成 0.6 与 0.6000000000000001 这种微小差异。
    # 这里给一个“只吞掉舍入误差”的极小时间容差：
    #   - 目的是修正二进制浮点表示误差；
    #   - 不是为了扩大真实 max_dt 时间窗；
    #   - 因此量级必须远小于任何实际采样周期。
    TIME_CMP_EPS_S = 1e-12

    i = 0
    last = np.full(C, np.nan, dtype=float)
    last_t = np.nan
    for k in range(N):
        tk = float(t_main[k])
        while i < t_src.size and t_src[i] <= (tk + TIME_CMP_EPS_S):
            last = vals[i]
            last_t = float(t_src[i])
            i += 1

        if not np.isfinite(last_t):
            continue
        if max_dt is not None and (tk - last_t) > (float(max_dt) + TIME_CMP_EPS_S):
            continue

        out[k] = last
        mask[k] = True
    return out, mask


def _attach_sparse_to_main(
    t_sparse: np.ndarray,
    values: np.ndarray,
    t_main: np.ndarray,
    max_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将低频 / 稀疏信号（DVL）attach 到主时间轴：

      - 对每一个 t_sparse[i]，找到最近的 t_main[k]
      - 若 |t_sparse[i] - t_main[k]| <= max_dt，则将该值写入第 k 行
      - 一个 bin 若有多条样本，则简单平均

    返回
    ----
    (aligned, mask_has)
      aligned : (N_main, C) NaN + 稀疏填值
      mask_has: (N_main,) bool，对应行是否有至少一个样本
    """
    t_sparse = np.asarray(t_sparse, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    t_main = np.asarray(t_main, dtype=float).reshape(-1)
    max_dt = float(max_dt)

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.shape[0] != t_sparse.size:
        raise ValueError("[ALIGN] _attach_sparse_to_main: time/value 长度不一致")

    N = t_main.size
    C = vals.shape[1]
    out = np.full((N, C), np.nan, dtype=float)
    count = np.zeros(N, dtype=int)

    if t_sparse.size == 0:
        return out, np.zeros(N, dtype=bool)

    for i in range(t_sparse.size):
        ti = t_sparse[i]
        ins = int(np.searchsorted(t_main, ti, side="left"))
        cand: list[int] = []
        if ins < N:
            cand.append(ins)
        if ins - 1 >= 0:
            cand.append(ins - 1)
        if not cand:
            continue
        k = min(cand, key=lambda kk: abs(ti - t_main[kk]))
        if abs(ti - t_main[k]) > max_dt:
            continue
        if count[k] == 0:
            out[k] = vals[i]
            count[k] = 1
        else:
            # 多个样本落在同一 bin：做简单平均
            out[k] = (out[k] * count[k] + vals[i]) / float(count[k] + 1)
            count[k] += 1

    mask_has = count > 0
    return out, mask_has

# src/uwnav_dynamics/preprocess/align/aligner.py

def save_training_table_imu_main(
    imu_proc_csv: str | Path,
    pwm_csv: str | Path,
    dvl_proc_csv: Optional[str | Path],
    power_csv: Optional[str | Path],
    out_csv: str | Path,
    cfg: Optional[AlignConfig] = None,
) -> Path:
    """
    包一层：调用 build_training_table_imu_main 并将结果写成 CSV。

    out_csv 一般建议类似：
      out/train/2026-01-10_pooltest02_train_base.csv
    """
    df = build_training_table_imu_main(
        imu_proc_csv=imu_proc_csv,
        pwm_csv=pwm_csv,
        dvl_proc_csv=dvl_proc_csv,
        power_csv=power_csv,
        cfg=cfg,
    )

    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[ALIGN] saved training base table to: {out_path}")
    print(f"[ALIGN] shape={df.shape}")
    return out_path


# =============================================================================
# Public API: 构造「控制导向」训练表
# =============================================================================

def build_training_table_imu_main(
    imu_proc_csv: str | Path,
    pwm_csv: str | Path,
    dvl_proc_csv: Optional[str | Path] = None,
    power_csv: Optional[str | Path] = None,
    cfg: Optional[AlignConfig] = None,
) -> pd.DataFrame:
    """
    构造「以 IMU 主轴（或外部主轴）+ PWM」的训练表：

      - 主时间轴：优先外部 main_axis_csv；否则由 IMU/PWM 交集确定
        （默认复用 IMU 时间轴；必要时回退等间隔网格）
      - 特征：
          * IMU：a_body, gyro_body（按主轴分箱平均）
          * PWM：ch1_cmd..ch8_cmd（hold-last）
          * DVL：VelBx/VelBy/VelBz/Speed（稀疏 attach，+ dvl_mask 掩码）
          * Power：P0..P7（hold-last + max_dt gating，+ power_mask）

    输出 DataFrame 列的典型顺序：

      ['t_s',
       'AccX_body_mps2', 'AccY_body_mps2', 'AccZ_body_mps2',
       'GyroX_body_rad_s', 'GyroY_body_rad_s', 'GyroZ_body_rad_s',
       'ch1_cmd', ..., 'ch8_cmd',
       'VelBx_body_mps', 'VelBy_body_mps', 'VelBz_body_mps', 'Speed_body_mps',
       'dvl_mask', 'has_dvl',
       'power_mask',
       'P0_W', ..., 'P7_W']
    """
    if cfg is None:
        cfg = AlignConfig()

    imu_path = Path(imu_proc_csv).expanduser().resolve()
    pwm_path = Path(pwm_csv).expanduser().resolve()
    dvl_path = Path(dvl_proc_csv).expanduser().resolve() if dvl_proc_csv else None
    power_path = Path(power_csv).expanduser().resolve() if power_csv else None

    # ---------------- 1) 读 IMU 预处理结果 ----------------
    df_imu = pd.read_csv(imu_path)
    if "t_s" not in df_imu.columns:
        raise ValueError(f"[ALIGN] IMU CSV 缺少列 't_s': {imu_path}")

    t_imu = df_imu["t_s"].to_numpy(dtype=float)

    imu_acc_cols = ["AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2"]
    imu_gyro_cols = ["GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"]
    for c in imu_acc_cols + imu_gyro_cols:
        if c not in df_imu.columns:
            raise KeyError(f"[ALIGN] IMU CSV 缺少列 {c!r}: {imu_path}")

    acc_imu = df_imu[imu_acc_cols].to_numpy(dtype=float)
    gyro_imu = df_imu[imu_gyro_cols].to_numpy(dtype=float)

    # ---------------- 2) 读 PWM 对齐日志 ----------------
    df_pwm = pd.read_csv(pwm_path)
    if "t_s" not in df_pwm.columns:
        raise ValueError(f"[ALIGN] PWM CSV 缺少列 't_s': {pwm_path}")

    t_pwm = df_pwm["t_s"].to_numpy(dtype=float)

    pwm_cols = [c for c in df_pwm.columns if c.startswith("ch") and c.endswith("_cmd")]
    if not pwm_cols:
        raise KeyError(
            f"[ALIGN] PWM CSV 未找到 ch*_cmd 列，请检查: {pwm_path}"
        )
    pwm_vals = df_pwm[pwm_cols].to_numpy(dtype=float)

    # ---------------- 3) 主时间轴 ----------------
    t_main = _resolve_main_grid(t_imu=t_imu, t_pwm=t_pwm, cfg=cfg)
    N = t_main.size

    # ---------------- 4) IMU -> 主时间轴（dense interpolation） ----------------
    _assert_main_axis_within_src_range(t_imu, t_main, src_name="imu")
    acc_main = _interp_dense_to_main(
        t_src=t_imu,
        values=acc_imu,
        t_main_s=t_main,
        name="acc_body",
    )
    gyro_main = _interp_dense_to_main(
        t_src=t_imu,
        values=gyro_imu,
        t_main_s=t_main,
        name="gyro_body",
    )
    if (not np.all(np.isfinite(acc_main))) or (not np.all(np.isfinite(gyro_main))):
        raise RuntimeError(
            "[ALIGN][IMU] aligned dense IMU targets must be finite before writing train_base.csv"
        )

    # ---------------- 5) PWM: hold-last 到主时间轴 ----------------
    pwm_main = _sample_last_before(
        t_src=t_pwm,
        values=pwm_vals,
        t_main=t_main,
    )
    # 检查是否存在大量 NaN（例如主时间轴开头超前于 PWM）
    if np.isnan(pwm_main).all():
        raise RuntimeError("[ALIGN] PWM 对齐结果全为 NaN，请检查时间范围。")

    # ---------------- 6) DVL 稀疏监督（可选） ----------------
    vel_body_main = np.full((N, 3), np.nan, dtype=float)
    speed_main = np.full(N, np.nan, dtype=float)  # 先显式用 1D
    dvl_mask = np.zeros(N, dtype=bool)

    if dvl_path is not None and dvl_path.exists():
        df_dvl = pd.read_csv(dvl_path)
        if "t_s" not in df_dvl.columns:
            raise ValueError(f"[ALIGN] DVL CSV 缺少列 't_s': {dvl_path}")

        # 若存在 kind / used，可以仅使用 BI & used==1 的行
        mask = np.ones(len(df_dvl), dtype=bool)
        if "kind" in df_dvl.columns:
            mask &= df_dvl["kind"].astype(str) == "BI"
        if "used" in df_dvl.columns:
            mask &= df_dvl["used"].astype(int) == 1

        df_dvl_use = df_dvl.loc[mask].reset_index(drop=True)
        t_dvl = df_dvl_use["t_s"].to_numpy(dtype=float)

        vel_cols = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]
        for c in vel_cols:
            if c not in df_dvl_use.columns:
                raise KeyError(f"[ALIGN] DVL CSV 缺少列 {c!r}: {dvl_path}")
        vel_body = df_dvl_use[vel_cols].to_numpy(dtype=float)

        vel_body_main, dvl_mask = _attach_sparse_to_main(
            t_sparse=t_dvl,
            values=vel_body,
            t_main=t_main,
            max_dt=cfg.dvl_max_dt_s,
        )

        # Speed 为 |v_body|，若 CSV 中已有 Speed_body_mps，可直接 attach 一次；
        # 否则在对齐后的 v_body 上现算。
        if "Speed_body_mps" in df_dvl_use.columns:
            speed_sparse = df_dvl_use["Speed_body_mps"].to_numpy(dtype=float)
            speed_aligned, speed_mask = _attach_sparse_to_main(
                t_sparse=t_dvl,
                values=speed_sparse,
                t_main=t_main,
                max_dt=cfg.dvl_max_dt_s,
            )
            # speed_aligned: (N, 1) → 压成 (N,)
            if speed_aligned.ndim == 2 and speed_aligned.shape[1] == 1:
                speed_main = speed_aligned[:, 0]
            else:
                speed_main = np.asarray(speed_aligned, dtype=float).reshape(-1)
            # speed 与速度向量按“有效样本交集”保留
            dvl_mask = dvl_mask & speed_mask
            speed_main[~dvl_mask] = np.nan
        else:
            # 用对齐后的 v_body 算范数（如果 v_body_main 这一行是 NaN，norm 也会给 NaN）
            speed_main = np.linalg.norm(vel_body_main, axis=1)
        vel_body_main[~dvl_mask, :] = np.nan

        print(
            f"[ALIGN] DVL attached: N_sparse={t_dvl.size}, "
            f"N_main_with_dvl={int(dvl_mask.sum())}"
        )
    else:
        vel_body_main[:, :] = np.nan
        speed_main[:] = np.nan
        dvl_mask[:] = False
    # ---------------- 7) Power: hold-last（可选） ----------------
    power_cols: Sequence[str] = []
    power_main = None
    power_mask = np.zeros(N, dtype=bool)

    if power_path is not None and power_path.exists():
        df_pw = pd.read_csv(power_path)
        if "t_s" not in df_pw.columns:
            raise ValueError(f"[ALIGN] Power CSV 缺少列 't_s': {power_path}")

        t_pw = df_pw["t_s"].to_numpy(dtype=float)
        power_cols = [c for c in df_pw.columns if c.startswith("P") and c.endswith("_W")]
        if not power_cols:
            print(f"[ALIGN] WARNING: Power CSV 中未找到 P*_W 列: {power_path}")
        else:
            power_vals = df_pw[power_cols].to_numpy(dtype=float)
            power_main, power_mask = _sample_last_before_with_max_dt(
                t_src=t_pw,
                values=power_vals,
                t_main=t_main,
                max_dt=float(cfg.power_max_dt_s),
            )
            power_main[~power_mask, :] = np.nan
            print(
                f"[ALIGN] Power attached: cols={power_cols}, "
                f"N={power_main.shape[0]}, N_main_with_power={int(power_mask.sum())}"
            )
    else:
        if cfg.require_power:
            raise FileNotFoundError(
                f"[ALIGN] require_power=True, 但未提供 Power CSV: {power_csv}"
            )

    # ---------------- 8) 组装 DataFrame ----------------
    data: Dict[str, Any] = {}

    # 主时间轴
    data["t_s"] = t_main

    # IMU
    data["AccX_body_mps2"] = acc_main[:, 0]
    data["AccY_body_mps2"] = acc_main[:, 1]
    data["AccZ_body_mps2"] = acc_main[:, 2]

    data["GyroX_body_rad_s"] = gyro_main[:, 0]
    data["GyroY_body_rad_s"] = gyro_main[:, 1]
    data["GyroZ_body_rad_s"] = gyro_main[:, 2]

    # PWM（控制输入）
    for i, c in enumerate(pwm_cols):
        data[c] = pwm_main[:, i]

    # DVL（稀疏监督）
    data["VelBx_body_mps"] = vel_body_main[:, 0]
    data["VelBy_body_mps"] = vel_body_main[:, 1]
    data["VelBz_body_mps"] = vel_body_main[:, 2]
    data["Speed_body_mps"] = speed_main
    data["dvl_mask"] = dvl_mask.astype(int)   # 新标准掩码列
    data["has_dvl"] = dvl_mask.astype(int)    # 兼容旧代码

    # Power（可选）
    data["power_mask"] = power_mask.astype(int)
    if power_main is not None and power_cols:
        for i, c in enumerate(power_cols):
            data[c] = power_main[:, i]

    df_out = pd.DataFrame(data)
    qa_report = run_train_base_qa(
        df_out,
        stage="train_base",
        time_col="t_s",
        dense_target_cols=imu_acc_cols + imu_gyro_cols,
        key_stat_cols=imu_acc_cols + imu_gyro_cols,
    )
    print(render_train_base_qa(qa_report))
    assert_train_base_qa_pass(qa_report)
    return df_out
