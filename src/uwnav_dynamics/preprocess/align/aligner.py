# src/uwnav_dynamics/preprocess/align/aligner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass
class AlignConfig:
    """
    多频率对齐配置（面向控制的训练用）：

      - 主时间轴：50 Hz（dt_main_s=0.02）
      - 主时间区间：IMU & PWM 时间轴的交集（可加 margin）
      - DVL：作为稀疏监督，仅在接近主时间点时写入，其余为 NaN
      - Power：低频辅助量，hold-last 对齐到主时间轴

    通常在 configs/preprocess/alignment.yaml 或
    configs/dataset/<name>.yaml 中 YAML 化。
    """
    # 主时间步长：控制频率 50 Hz
    dt_main_s: float = 0.02

    # 裁剪边缘：避免刚启动/关停阶段的奇怪数据
    t_margin_s: float = 0.0

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
    根据 IMU / PWM 时间范围生成主时间轴（100 Hz 等间距）：

      - 主区间 = [max(t_imu0, t_pwm0) + margin, min(t_imu1, t_pwm1) - margin]
      - 步长 = cfg.dt_main_s
    """
    if t_imu.size < 2 or t_pwm.size < 2:
        raise ValueError("[ALIGN] IMU/PWM 样本太少，无法构建主时间轴。")

    t0 = max(float(t_imu[0]), float(t_pwm[0])) + float(cfg.t_margin_s)
    t1 = min(float(t_imu[-1]), float(t_pwm[-1])) - float(cfg.t_margin_s)

    if t1 <= t0:
        raise ValueError(
            f"[ALIGN] 无有效时间交集：t0={t0:.3f}, t1={t1:.3f}. "
            "请检查 IMU / PWM 日志时间范围。"
        )

    dt = float(cfg.dt_main_s)
    n = int(np.floor((t1 - t0) / dt)) + 1
    t_main = t0 + dt * np.arange(n, dtype=float)

    print(
        f"[ALIGN] main time-grid: t=[{t_main[0]:.3f}, {t_main[-1]:.3f}], "
        f"N={t_main.size}, dt={dt:.4f}s"
    )
    return t_main


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

    t0 = float(t_main[0])
    dt = float(t_main[1] - t_main[0]) if N >= 2 else 1.0

    for i in range(t_sparse.size):
        ti = t_sparse[i]
        k = int(np.round((ti - t0) / dt))
        if k < 0 or k >= N:
            continue
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
    构造「以 IMU+PWM 为主」的 50 Hz 训练表：

      - 主时间轴 50 Hz：由 IMU & PWM 时间交集确定
      - 特征：
          * IMU：a_body, gyro_body（bin-average 100→50 Hz）
          * PWM：ch1_cmd..ch8_cmd（hold-last）
          * DVL：VelBx/VelBy/VelBz/Speed（稀疏 attach，+ has_dvl 掩码）
          * Power：P0..P7（hold-last，仅作辅助，可在训练时屏蔽）

    输出 DataFrame 列的典型顺序：

      ['t_s',
       'AccX_body_mps2', 'AccY_body_mps2', 'AccZ_body_mps2',
       'GyroX_body_rad_s', 'GyroY_body_rad_s', 'GyroZ_body_rad_s',
       'ch1_cmd', ..., 'ch8_cmd',
       'VelBx_body_mps', 'VelBy_body_mps', 'VelBz_body_mps', 'Speed_body_mps',
       'has_dvl',
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

    # ---------------- 3) 主时间轴（50 Hz） ----------------
    t_main = _compute_main_grid(t_imu, t_pwm, cfg)
    N = t_main.size

    # ---------------- 4) IMU 100 Hz → 50 Hz (bin-average) ----------------
    acc_main = _bin_average_multi(
        t_src=t_imu,
        values=acc_imu,
        t0=float(t_main[0]),
        dt=float(cfg.dt_main_s),
        n_bins=N,
    )
    gyro_main = _bin_average_multi(
        t_src=t_imu,
        values=gyro_imu,
        t0=float(t_main[0]),
        dt=float(cfg.dt_main_s),
        n_bins=N,
    )

    # 若某些 bin 完全没有 IMU 样本（理论上不应该），直接抛异常提醒
    if np.isnan(acc_main).all():
        raise RuntimeError("[ALIGN] IMU bin-average 结果全为 NaN，请检查时间戳。")

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
    has_dvl = np.zeros(N, dtype=bool)

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

        vel_body_main, has_dvl = _attach_sparse_to_main(
            t_sparse=t_dvl,
            values=vel_body,
            t_main=t_main,
            max_dt=cfg.dvl_max_dt_s,
        )

        # Speed 为 |v_body|，若 CSV 中已有 Speed_body_mps，可直接 attach 一次；
        # 否则在对齐后的 v_body 上现算。
        if "Speed_body_mps" in df_dvl_use.columns:
            speed_sparse = df_dvl_use["Speed_body_mps"].to_numpy(dtype=float)
            speed_aligned, _ = _attach_sparse_to_main(
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
        else:
            # 用对齐后的 v_body 算范数（如果 v_body_main 这一行是 NaN，norm 也会给 NaN）
            speed_main = np.linalg.norm(vel_body_main, axis=1)

        print(
            f"[ALIGN] DVL attached: N_sparse={t_dvl.size}, "
            f"N_main_with_dvl={int(has_dvl.sum())}"
        )
    # ---------------- 7) Power: hold-last（可选） ----------------
    power_cols: Sequence[str] = []
    power_main = None

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
            power_main = _sample_last_before(
                t_src=t_pw,
                values=power_vals,
                t_main=t_main,
            )
            print(
                f"[ALIGN] Power attached: cols={power_cols}, "
                f"N={power_main.shape[0]}"
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
    data["has_dvl"] = has_dvl.astype(int)  # 存成 0/1，便于训练中做 masking

    # Power（可选）
    if power_main is not None and power_cols:
        for i, c in enumerate(power_cols):
            data[c] = power_main[:, i]

    df_out = pd.DataFrame(data)
    return df_out
