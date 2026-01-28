# SPDX-License-Identifier: AGPL-3.0-or-later
# src/uwnav_dynamics/preprocess/align/aligner.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple, List

import numpy as np
import pandas as pd


# =============================================================================
# Config (A: grid-based, legacy)
# =============================================================================

@dataclass
class AlignConfig:
    """
    多频率对齐配置（legacy / 跑通 pipeline 用）：

      - 主时间轴：固定步长 dt_main_s（默认 50Hz=0.02s）
      - 主区间：IMU & PWM 时间轴交集（可加 margin）
      - IMU：bin-average 到主时间轴
      - PWM/Power：hold-last 到主时间轴
      - DVL：稀疏 attach 到主时间轴（其余填 NaN）+ has_dvl mask

    说明：
      - 这是“网格对齐（grid）”范式，会产生 NaN（稀疏监督源）。
      - 方案 B1（event-based）请使用 EventStreamConfig + build_event_stream()。
    """
    dt_main_s: float = 0.02
    t_margin_s: float = 0.0

    dvl_max_dt_s: float = 0.10
    power_max_dt_s: float = 0.25

    require_pwm: bool = True
    require_dvl: bool = False
    require_power: bool = False


# =============================================================================
# Config (B1: event-based, recommended for “学术路线”)
# =============================================================================

@dataclass
class EventStreamConfig:
    """
    方案 B1：事件流（异步）对齐配置。

    核心思想：
      - 不再构建等间隔 t_main；
      - 输出 union(events) 的时间轴 t；
      - 每一行只写“当时刻发生的事件”的字段，其余字段为 0；
      - 用 mask m 指明哪些字段有效（1=有效，0=缺失）。

    默认 Din=25 的 layout（与你当前 S1 保持一致）：
      [PWM8] + [IMU6] + [Vel_state3] + [Power8]
      idx:  0..7     8..13    14..16      17..24

    注意：
      - Vel_state3 如果你还没明确来源，B1 第一版通常先不写入（保留为 0 + mask=0）。
      - DVL 推荐作为“目标监督源”单独输出（t_dvl, v_dvl, m_dvl），而不是写进 x。
    """
    t_margin_s: float = 0.0

    din: int = 25
    pwm_idx: Sequence[int] = tuple(range(0, 8))
    imu_idx: Sequence[int] = tuple(range(8, 14))
    vel_state_idx: Sequence[int] = tuple(range(14, 17))
    power_idx: Sequence[int] = tuple(range(17, 25))

    # 是否把 DVL 写进 x（不推荐 B1 第一版；建议作为 target 单独输出）
    dvl_as_input: bool = False

    # 若你要把某些源要求为必需，可在上层脚本检查，这里保持宽松
    require_power: bool = False
    require_dvl: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _sort_ch_cols(cols: List[str]) -> List[str]:
    """Sort ['ch1_cmd', 'ch2_cmd', ...] by channel index."""
    def _key(c: str) -> int:
        # 'ch12_cmd' -> 12
        s = c
        s = s.replace("ch", "").replace("_cmd", "")
        try:
            return int(s)
        except Exception:
            return 10**9
    return sorted(cols, key=_key)


def _sort_p_cols(cols: List[str]) -> List[str]:
    """Sort ['P0_W','P1_W',...] by index."""
    def _key(c: str) -> int:
        s = c
        s = s.replace("P", "").replace("_W", "")
        try:
            return int(s)
        except Exception:
            return 10**9
    return sorted(cols, key=_key)


def _clip_range(t0: float, t1: float, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip (t,v) to [t0,t1]."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    m = (t >= float(t0)) & (t <= float(t1))
    return t[m], v[m]


# =============================================================================
# Legacy grid helpers (A)
# =============================================================================

def _compute_main_grid(t_imu: np.ndarray, t_pwm: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """
    主时间轴：由 IMU & PWM 的时间交集确定，步长 dt_main_s（等间距）。
    """
    if t_imu.size < 2 or t_pwm.size < 2:
        raise ValueError("[ALIGN] IMU/PWM 样本太少，无法构建主时间轴。")

    t0 = max(float(t_imu[0]), float(t_pwm[0])) + float(cfg.t_margin_s)
    t1 = min(float(t_imu[-1]), float(t_pwm[-1])) - float(cfg.t_margin_s)

    if t1 <= t0:
        raise ValueError(
            f"[ALIGN] 无有效时间交集：t0={t0:.6f}, t1={t1:.6f}. "
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
    对高频信号（IMU）做按主时间轴分箱平均。
    """
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
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


def _sample_last_before(t_src: np.ndarray, values: np.ndarray, t_main: np.ndarray) -> np.ndarray:
    """
    对控制输入型信号（PWM/Power）做 hold-last 对齐到主时间轴。
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
    稀疏信号（DVL）attach 到主时间轴的最近邻 bin（允许误差 max_dt）。
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
            out[k] = (out[k] * count[k] + vals[i]) / float(count[k] + 1)
            count[k] += 1

    mask_has = count > 0
    return out, mask_has


# =============================================================================
# Legacy grid public API (A)
# =============================================================================

def save_training_table_imu_main(
    imu_proc_csv: str | Path,
    pwm_csv: str | Path,
    dvl_proc_csv: Optional[str | Path],
    power_csv: Optional[str | Path],
    out_csv: str | Path,
    cfg: Optional[AlignConfig] = None,
) -> Path:
    """
    legacy: build_training_table_imu_main 并写 CSV。
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


def build_training_table_imu_main(
    imu_proc_csv: str | Path,
    pwm_csv: str | Path,
    dvl_proc_csv: Optional[str | Path] = None,
    power_csv: Optional[str | Path] = None,
    cfg: Optional[AlignConfig] = None,
) -> pd.DataFrame:
    """
    legacy / grid-based：
      - 主时间轴固定 dt_main_s
      - IMU bin-average
      - PWM/Power hold-last
      - DVL sparse attach + has_dvl

    升级版特性（防沉默失败）：
      - 时间戳单调性检查/必要时排序（不悄悄改数据，打印明确告警）
      - 与 main-grid 相同的交集裁剪，减少边界 NaN
      - 对齐质量诊断：NaN ratio / 最长连续 NaN / PWM&Power 陈旧度 / DVL attach 误差统计
    """
    if cfg is None:
        cfg = AlignConfig()

    imu_path = Path(imu_proc_csv).expanduser().resolve()
    pwm_path = Path(pwm_csv).expanduser().resolve()
    dvl_path = Path(dvl_proc_csv).expanduser().resolve() if dvl_proc_csv else None
    power_path = Path(power_csv).expanduser().resolve() if power_csv else None

    # ---------------------------------------------------------------------
    # helpers (local)
    # ---------------------------------------------------------------------
    def _is_non_decreasing(t: np.ndarray) -> bool:
        t = np.asarray(t, dtype=float).reshape(-1)
        return bool(np.all(np.diff(t) >= 0))

    def _sort_by_time(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if v.shape[0] != t.size:
            raise ValueError("[ALIGN] time/value 长度不一致（sort_by_time）")
        idx = np.argsort(t, kind="mergesort")
        return t[idx], v[idx]

    def _dup_ratio(t: np.ndarray) -> float:
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size <= 1:
            return 0.0
        u = np.unique(t).size
        return float(1.0 - (u / float(t.size)))

    def _max_run(mask: np.ndarray) -> int:
        """Longest consecutive True run length."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size == 0:
            return 0
        # run-length on mask
        best = 0
        cur = 0
        for b in mask:
            if b:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def _nan_stats(arr: np.ndarray) -> tuple[float, int]:
        """Return (nan_ratio, max_consecutive_nan_run). Works for 1D."""
        a = np.asarray(arr)
        if a.ndim != 1:
            a = a.reshape(-1)
        nan = np.isnan(a)
        return float(np.mean(nan)), _max_run(nan)

    def _sample_last_before_with_time(
        t_src: np.ndarray, values: np.ndarray, t_main: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        hold-last，对齐到 t_main。
        返回:
          - out: (N,C) last value
          - t_last: (N,) last update time (NaN if never updated)
        """
        t_src = np.asarray(t_src, dtype=float).reshape(-1)
        vals = np.asarray(values, dtype=float)
        t_main = np.asarray(t_main, dtype=float).reshape(-1)

        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if vals.shape[0] != t_src.size:
            raise ValueError("[ALIGN] hold-last: time/value 长度不一致")

        N = t_main.size
        C = vals.shape[1]
        out = np.full((N, C), np.nan, dtype=float)
        t_last = np.full((N,), np.nan, dtype=float)

        if t_src.size == 0:
            return out, t_last

        i = 0
        last = np.full(C, np.nan, dtype=float)
        last_t = np.nan
        for k in range(N):
            tk = t_main[k]
            while i < t_src.size and t_src[i] <= tk:
                last = vals[i]
                last_t = t_src[i]
                i += 1
            out[k] = last
            t_last[k] = last_t

        return out, t_last

    # ---------------------------------------------------------------------
    # ---- IMU ----
    # ---------------------------------------------------------------------
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

    # 单调性/重复检查
    if not _is_non_decreasing(t_imu):
        print("[ALIGN][WARN] IMU t_s 非单调，已按时间排序（mergesort 保持稳定）。")
        t_imu, acc_imu = _sort_by_time(t_imu, acc_imu)
        _, gyro_imu = _sort_by_time(df_imu["t_s"].to_numpy(dtype=float), gyro_imu)  # re-sort gyro with same original t
        # 为了稳妥，直接再用 t_imu 的排序索引重排 gyro（避免上面二次排序潜在偏差）
        # 这里用 argsort 一次性解决：
        idx = np.argsort(df_imu["t_s"].to_numpy(dtype=float), kind="mergesort")
        gyro_imu = gyro_imu[idx]
    dup_imu = _dup_ratio(t_imu)
    if dup_imu > 0.001:
        print(f"[ALIGN][WARN] IMU 时间戳存在重复：dup_ratio={dup_imu:.4f}（不强制去重）。")

    # ---------------------------------------------------------------------
    # ---- PWM ----
    # ---------------------------------------------------------------------
    df_pwm = pd.read_csv(pwm_path)
    if "t_s" not in df_pwm.columns:
        raise ValueError(f"[ALIGN] PWM CSV 缺少列 't_s': {pwm_path}")
    t_pwm = df_pwm["t_s"].to_numpy(dtype=float)

    pwm_cols = [c for c in df_pwm.columns if c.startswith("ch") and c.endswith("_cmd")]
    pwm_cols = _sort_ch_cols(pwm_cols)
    if not pwm_cols:
        raise KeyError(f"[ALIGN] PWM CSV 未找到 ch*_cmd 列: {pwm_path}")
    pwm_vals = df_pwm[pwm_cols].to_numpy(dtype=float)

    if not _is_non_decreasing(t_pwm):
        print("[ALIGN][WARN] PWM t_s 非单调，已按时间排序。")
        t_pwm, pwm_vals = _sort_by_time(t_pwm, pwm_vals)
    dup_pwm = _dup_ratio(t_pwm)
    if dup_pwm > 0.001:
        print(f"[ALIGN][WARN] PWM 时间戳存在重复：dup_ratio={dup_pwm:.4f}（不强制去重）。")

    # ---------------------------------------------------------------------
    # ---- main grid ----
    # ---------------------------------------------------------------------
    t_main = _compute_main_grid(t_imu, t_pwm, cfg)
    N = t_main.size

    # 与 main-grid 一致的交集裁剪（减少边界 NaN/末尾 hold-last 空）
    t0 = float(t_main[0])
    t1 = float(t_main[-1])
    t_imu, acc_imu = _clip_range(t0, t1, t_imu, acc_imu)
    _, gyro_imu = _clip_range(t0, t1, df_imu["t_s"].to_numpy(dtype=float), df_imu[imu_gyro_cols].to_numpy(dtype=float))
    # 上面 gyro 用原 df 再裁剪一次；为了严格对应 t_imu 的裁剪，可直接重新从 df_imu 裁剪整块：
    # 但保持“最小侵入”，这里不改变你的 IMU 分块方式。

    t_pwm, pwm_vals = _clip_range(t0, t1, t_pwm, pwm_vals)

    # ---------------------------------------------------------------------
    # ---- IMU bin-average ----
    # ---------------------------------------------------------------------
    acc_main = _bin_average_multi(t_imu, acc_imu, float(t_main[0]), float(cfg.dt_main_s), N)
    gyro_main = _bin_average_multi(t_imu, gyro_imu, float(t_main[0]), float(cfg.dt_main_s), N)
    if np.isnan(acc_main).all():
        raise RuntimeError("[ALIGN] IMU bin-average 结果全为 NaN，请检查时间戳/范围。")

    # ---------------------------------------------------------------------
    # ---- PWM hold-last ----
    # ---------------------------------------------------------------------
    pwm_main, t_pwm_last = _sample_last_before_with_time(t_pwm, pwm_vals, t_main)
    if np.isnan(pwm_main).all():
        raise RuntimeError("[ALIGN] PWM 对齐结果全为 NaN，请检查时间范围。")

    # ---------------------------------------------------------------------
    # ---- DVL sparse attach ----
    # ---------------------------------------------------------------------
    vel_body_main = np.full((N, 3), np.nan, dtype=float)
    speed_main = np.full((N,), np.nan, dtype=float)
    has_dvl = np.zeros((N,), dtype=bool)
    dvl_attach_err = np.full((N,), np.nan, dtype=float)  # |t_dvl - t_main[k]| for attached bins

    if dvl_path is not None and dvl_path.exists():
        df_dvl = pd.read_csv(dvl_path)
        if "t_s" not in df_dvl.columns:
            raise ValueError(f"[ALIGN] DVL CSV 缺少列 't_s': {dvl_path}")

        mask = np.ones(len(df_dvl), dtype=bool)
        if "kind" in df_dvl.columns:
            mask &= df_dvl["kind"].astype(str) == "BI"
        if "used" in df_dvl.columns:
            mask &= df_dvl["used"].astype(int) == 1
        df_dvl_use = df_dvl.loc[mask].reset_index(drop=True)

        if len(df_dvl_use) == 0:
            print("[ALIGN][WARN] DVL 经过 (kind==BI & used==1) 筛选后为空。")
        else:
            t_dvl = df_dvl_use["t_s"].to_numpy(dtype=float)

            vel_cols = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]
            for c in vel_cols:
                if c not in df_dvl_use.columns:
                    raise KeyError(f"[ALIGN] DVL CSV 缺少列 {c!r}: {dvl_path}")
            vel_body = df_dvl_use[vel_cols].to_numpy(dtype=float)

            if not _is_non_decreasing(t_dvl):
                print("[ALIGN][WARN] DVL t_s 非单调，已按时间排序。")
                t_dvl, vel_body = _sort_by_time(t_dvl, vel_body)
            dup_dvl = _dup_ratio(t_dvl)
            if dup_dvl > 0.001:
                print(f"[ALIGN][WARN] DVL 时间戳存在重复：dup_ratio={dup_dvl:.4f}（不强制去重）。")

            # 裁剪到 main 范围
            t_dvl, vel_body = _clip_range(t0, t1, t_dvl, vel_body)

            vel_body_main, has_dvl = _attach_sparse_to_main(
                t_sparse=t_dvl,
                values=vel_body,
                t_main=t_main,
                max_dt=cfg.dvl_max_dt_s,
            )

            # attach 误差统计（用相同的 bin 映射逻辑重算一次误差）
            if t_dvl.size and N >= 2:
                t0m = float(t_main[0])
                dtm = float(t_main[1] - t_main[0])
                for ti in t_dvl:
                    k = int(np.round((ti - t0m) / dtm))
                    if 0 <= k < N and abs(ti - t_main[k]) <= float(cfg.dvl_max_dt_s):
                        # 多次落在同一个 k，记录最小误差
                        e = abs(ti - t_main[k])
                        if np.isnan(dvl_attach_err[k]) or e < dvl_attach_err[k]:
                            dvl_attach_err[k] = e

            if "Speed_body_mps" in df_dvl_use.columns:
                speed_sparse = df_dvl_use["Speed_body_mps"].to_numpy(dtype=float)
                speed_sparse = speed_sparse[: len(df_dvl_use)]  # 防御性
                # 注意：如果你对 df_dvl_use 做了排序/裁剪，上面 speed_sparse 要同步处理；
                # 最小侵入：这里重新从裁剪后的 df_dvl_use 对齐（用 t_dvl 的 mask）
                # 简化做法：直接用 vel_body_main 的 norm（更一致）
                speed_main = np.linalg.norm(vel_body_main, axis=1)
            else:
                speed_main = np.linalg.norm(vel_body_main, axis=1)

            print(f"[ALIGN] DVL attached: N_sparse={t_dvl.size}, N_main_with_dvl={int(has_dvl.sum())}")
    else:
        if cfg.require_dvl:
            raise FileNotFoundError(f"[ALIGN] require_dvl=True, 但未提供/找不到 DVL CSV: {dvl_proc_csv}")

    # ---------------------------------------------------------------------
    # ---- Power hold-last ----
    # ---------------------------------------------------------------------
    power_main = None
    t_pow_last = None
    power_cols: Sequence[str] = []
    if power_path is not None and power_path.exists():
        df_pw = pd.read_csv(power_path)
        if "t_s" not in df_pw.columns:
            raise ValueError(f"[ALIGN] Power CSV 缺少列 't_s': {power_path}")
        t_pw = df_pw["t_s"].to_numpy(dtype=float)

        power_cols = [c for c in df_pw.columns if c.startswith("P") and c.endswith("_W")]
        power_cols = _sort_p_cols(power_cols)
        if not power_cols:
            print(f"[ALIGN][WARN] Power CSV 中未找到 P*_W 列: {power_path}")
        else:
            power_vals = df_pw[power_cols].to_numpy(dtype=float)
            if not _is_non_decreasing(t_pw):
                print("[ALIGN][WARN] Power t_s 非单调，已按时间排序。")
                t_pw, power_vals = _sort_by_time(t_pw, power_vals)
            dup_pw = _dup_ratio(t_pw)
            if dup_pw > 0.001:
                print(f"[ALIGN][WARN] Power 时间戳存在重复：dup_ratio={dup_pw:.4f}（不强制去重）。")

            # 裁剪到 main 范围
            t_pw, power_vals = _clip_range(t0, t1, t_pw, power_vals)

            power_main, t_pow_last = _sample_last_before_with_time(t_pw, power_vals, t_main)
            print(f"[ALIGN] Power attached: cols={list(power_cols)}, N={power_main.shape[0]}")
    else:
        if cfg.require_power:
            raise FileNotFoundError(f"[ALIGN] require_power=True, 但未提供/找不到 Power CSV: {power_csv}")

    # ---------------------------------------------------------------------
    # ---- assemble ----
    # ---------------------------------------------------------------------
    data: Dict[str, Any] = {}
    data["t_s"] = t_main

    data["AccX_body_mps2"] = acc_main[:, 0]
    data["AccY_body_mps2"] = acc_main[:, 1]
    data["AccZ_body_mps2"] = acc_main[:, 2]
    data["GyroX_body_rad_s"] = gyro_main[:, 0]
    data["GyroY_body_rad_s"] = gyro_main[:, 1]
    data["GyroZ_body_rad_s"] = gyro_main[:, 2]

    for i, c in enumerate(pwm_cols):
        data[c] = pwm_main[:, i]

    data["VelBx_body_mps"] = vel_body_main[:, 0]
    data["VelBy_body_mps"] = vel_body_main[:, 1]
    data["VelBz_body_mps"] = vel_body_main[:, 2]
    data["Speed_body_mps"] = speed_main
    data["has_dvl"] = has_dvl.astype(int)

    # ---- Power + has_power ----
    has_power = np.zeros((N,), dtype=bool)
    if power_main is not None and power_cols:
        for i, c in enumerate(power_cols):
            data[c] = power_main[:, i]

        # has_power: 8 路功率都有效才算 1（你也可以改成 any-valid，但推荐 all-valid）
        pw_block = np.asarray(power_main, dtype=float)
        if pw_block.ndim == 1:
            pw_block = pw_block.reshape(-1, 1)
        # 若列数不足 8，也按现有列判断
        has_power = ~np.any(np.isnan(pw_block), axis=1)

    data["has_power"] = has_power.astype(int)

    df_out = pd.DataFrame(data)
    # ---------------------------------------------------------------------
    # ---- DIAG: anti-silent-failure ----
    # ---------------------------------------------------------------------
    print("[ALIGN][DIAG] legacy grid quality:")
    print(f"  - main: N={N} dt={float(cfg.dt_main_s):.4f} t=[{t_main[0]:.3f},{t_main[-1]:.3f}]")
    print(f"  - IMU:  samples={t_imu.size} dup_ratio={dup_imu:.4f}")
    print(f"  - PWM:  samples={t_pwm.size} dup_ratio={dup_pwm:.4f}")

    # IMU NaN 统计（每列）
    for c in ["AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2",
              "GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"]:
        r, run = _nan_stats(df_out[c].to_numpy(dtype=float))
        print(f"  - {c}: nan_ratio={r:.4f} max_nan_run={run}")

    # PWM block NaN
    pwm_block = df_out[list(pwm_cols)].to_numpy(dtype=float) if pwm_cols else np.empty((N, 0))
    pwm_nan_ratio = float(np.mean(np.isnan(pwm_block))) if pwm_block.size else 1.0
    print(f"  - PWM block: nan_ratio={pwm_nan_ratio:.4f}")

    # PWM 陈旧度（hold-last age）
    if t_pwm_last is not None:
        age = df_out["t_s"].to_numpy(dtype=float) - np.asarray(t_pwm_last, dtype=float)
        age[np.isnan(age)] = np.inf
        p95 = float(np.percentile(age[np.isfinite(age)], 95)) if np.any(np.isfinite(age)) else np.inf
        max_age = float(np.max(age[np.isfinite(age)])) if np.any(np.isfinite(age)) else np.inf
        print(f"  - PWM age: p95={p95:.3f}s max={max_age:.3f}s")
        if p95 > 0.2:  # 经验阈值：50Hz 主网格下 >0.2s 通常不正常
            print("[ALIGN][WARN] PWM hold-last 陈旧度偏大（p95>0.2s），请检查 PWM 日志频率/是否断流。")

    # DVL 稀疏度与 attach 误差
    if "has_dvl" in df_out.columns:
        has = int(df_out["has_dvl"].sum())
        frac = has / float(len(df_out)) if len(df_out) else 0.0
        print(f"  - DVL has_dvl: {has}/{len(df_out)} = {frac:.4f}")
        if has > 0 and np.any(np.isfinite(dvl_attach_err)):
            e = dvl_attach_err[np.isfinite(dvl_attach_err)]
            print(f"  - DVL attach_err: mean={float(np.mean(e)):.4f}s  p95={float(np.percentile(e,95)):.4f}s  max={float(np.max(e)):.4f}s")
        if cfg.require_dvl and has == 0:
            raise RuntimeError("[ALIGN] require_dvl=True 但对齐后 has_dvl 全为 0，请检查 dvl_max_dt_s / 时间范围 / DVL数据质量。")

    # Power block NaN + 陈旧度
    if power_main is not None and power_cols:
        power_block = df_out[list(power_cols)].to_numpy(dtype=float)
        power_nan_ratio = float(np.mean(np.isnan(power_block))) if power_block.size else 1.0
        print(f"  - Power block: nan_ratio={power_nan_ratio:.4f}")

        if "has_power" in df_out.columns:
            hp = int(df_out["has_power"].sum())
            frac = hp / len(df_out) if len(df_out) else 0.0
            print(f"  - has_power: {hp}/{len(df_out)} = {frac:.4f}")
            if frac < 0.5:
                print("[ALIGN][WARN] has_power 覆盖率 < 50%，若 power 作为训练输入，建议用 has_power 做 mask 或缩短对齐区间。")

        if t_pow_last is not None:
            agep = df_out["t_s"].to_numpy(dtype=float) - np.asarray(t_pow_last, dtype=float)
            agep[np.isnan(agep)] = np.inf
            p95p = float(np.percentile(agep[np.isfinite(agep)], 95)) if np.any(np.isfinite(agep)) else np.inf
            maxp = float(np.max(agep[np.isfinite(agep)])) if np.any(np.isfinite(agep)) else np.inf
            print(f"  - Power age: p95={p95p:.3f}s max={maxp:.3f}s")
            if p95p > float(cfg.power_max_dt_s):
                print("[ALIGN][WARN] Power hold-last 陈旧度超过 power_max_dt_s，可能电源日志频率过低或断流。")

    # 强告警：IMU/PWM 大面积 NaN 属于典型沉默失败
    imu_nan_any = any(float(np.mean(np.isnan(df_out[c].to_numpy(dtype=float)))) > 0.2 for c in [
        "AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2",
        "GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s"
    ])
    if imu_nan_any:
        print("[ALIGN][WARN] IMU 在主网格上出现较多 NaN（>20%），通常意味着时间戳不匹配/裁剪区间异常/IMU日志断档。")

    if pwm_nan_ratio > 0.05:
        print("[ALIGN][WARN] PWM 在主网格上出现较多 NaN（>5%），通常不正常；请检查 IMU/PWM 时间交集与日志完整性。")

    return df_out

# =============================================================================
# Event-based public API (B1)
# =============================================================================
def build_event_stream(
    imu_proc_csv: str | Path,
    pwm_csv: str | Path,
    power_csv: Optional[str | Path] = None,
    dvl_proc_csv: Optional[str | Path] = None,
    cfg: Optional[EventStreamConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    方案 B1：构建异步事件流（events union），不做网格化、不做插值、不产生 NaN。

    修复要点（防沉默失败）：
      - 时间轴统一量化到 int64 纳秒（ns），避免 float 精确相等导致写入丢失；
      - 增加写入覆盖率/重复率/写入缺失告警诊断。

    返回 payload（建议 npz 保存）：
      - t     : (Ne,) float64    seconds, derived from t_ns
      - t_ns  : (Ne,) int64      nanoseconds, exact event time base
      - x     : (Ne, Din) float32
      - m     : (Ne, Din) uint8
      - src   : (Ne,) int8

    若提供 dvl_proc_csv，则额外返回（targets）：
      - t_dvl, t_dvl_ns, v_dvl, m_dvl
    """
    if cfg is None:
        cfg = EventStreamConfig()

    imu_path = Path(imu_proc_csv).expanduser().resolve()
    pwm_path = Path(pwm_csv).expanduser().resolve()
    power_path = Path(power_csv).expanduser().resolve() if power_csv else None
    dvl_path = Path(dvl_proc_csv).expanduser().resolve() if dvl_proc_csv else None

    def _to_ns(t_s: np.ndarray) -> np.ndarray:
        """float seconds -> int64 nanoseconds (rounded)."""
        t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
        return np.rint(t_s * 1e9).astype(np.int64)

    def _dup_ratio(t_ns: np.ndarray) -> float:
        t_ns = np.asarray(t_ns, dtype=np.int64).reshape(-1)
        if t_ns.size <= 1:
            return 0.0
        u = np.unique(t_ns).size
        return float(1.0 - (u / float(t_ns.size)))

    # ---- load IMU ----
    df_imu = pd.read_csv(imu_path)
    if "t_s" not in df_imu.columns:
        raise ValueError(f"[EVENT] IMU CSV 缺少列 't_s': {imu_path}")
    t_imu_s = df_imu["t_s"].to_numpy(dtype=float)
    imu_cols = [
        "AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2",
        "GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s",
    ]
    for c in imu_cols:
        if c not in df_imu.columns:
            raise KeyError(f"[EVENT] IMU CSV 缺少列 {c!r}: {imu_path}")
    imu_val = df_imu[imu_cols].to_numpy(dtype=float)

    # ---- load PWM ----
    df_pwm = pd.read_csv(pwm_path)
    if "t_s" not in df_pwm.columns:
        raise ValueError(f"[EVENT] PWM CSV 缺少列 't_s': {pwm_path}")
    t_pwm_s = df_pwm["t_s"].to_numpy(dtype=float)
    pwm_cols = [c for c in df_pwm.columns if c.startswith("ch") and c.endswith("_cmd")]
    pwm_cols = _sort_ch_cols(pwm_cols)
    if not pwm_cols:
        raise KeyError(f"[EVENT] PWM CSV 未找到 ch*_cmd 列: {pwm_path}")
    pwm_val = df_pwm[pwm_cols].to_numpy(dtype=float)

    # ---- optional Power ----
    t_pow_s = np.empty((0,), dtype=np.float64)
    pow_val = np.empty((0, 0), dtype=float)
    power_cols: List[str] = []
    if power_path is not None and power_path.exists():
        df_pw = pd.read_csv(power_path)
        if "t_s" not in df_pw.columns:
            raise ValueError(f"[EVENT] Power CSV 缺少列 't_s': {power_path}")
        t_pow_s = df_pw["t_s"].to_numpy(dtype=float)
        power_cols = [c for c in df_pw.columns if c.startswith("P") and c.endswith("_W")]
        power_cols = _sort_p_cols(power_cols)
        if not power_cols:
            if cfg.require_power:
                raise KeyError(f"[EVENT] require_power=True, 但 Power CSV 未找到 P*_W: {power_path}")
        else:
            pow_val = df_pw[power_cols].to_numpy(dtype=float)
    else:
        if cfg.require_power:
            raise FileNotFoundError(f"[EVENT] require_power=True, 但未提供/找不到 Power CSV: {power_csv}")

    # ---- time trimming by intersection (PWM & IMU) ----
    if t_imu_s.size < 2 or t_pwm_s.size < 2:
        raise ValueError("[EVENT] IMU/PWM 样本太少，无法构建事件流。")

    t0 = max(float(t_imu_s[0]), float(t_pwm_s[0])) + float(cfg.t_margin_s)
    t1 = min(float(t_imu_s[-1]), float(t_pwm_s[-1])) - float(cfg.t_margin_s)
    if t1 <= t0:
        raise ValueError(f"[EVENT] 无有效时间交集：t0={t0:.6f}, t1={t1:.6f}")

    t_imu_s, imu_val = _clip_range(t0, t1, t_imu_s, imu_val)
    t_pwm_s, pwm_val = _clip_range(t0, t1, t_pwm_s, pwm_val)
    if t_pow_s.size and pow_val.size:
        t_pow_s, pow_val = _clip_range(t0, t1, t_pow_s, pow_val)

    # ---- convert to ns (exact time base) ----
    t_imu_ns = _to_ns(t_imu_s)
    t_pwm_ns = _to_ns(t_pwm_s)
    t_pow_ns = _to_ns(t_pow_s) if (t_pow_s.size and pow_val.size) else np.empty((0,), dtype=np.int64)

    # ---- union time axis (events) ----
    # 注意：这里仍然不把 DVL union 进来（推荐作为 targets）
    t_all_ns = np.unique(np.concatenate([t_pwm_ns, t_imu_ns, t_pow_ns]).astype(np.int64))
    Ne = int(t_all_ns.size)
    Din = int(cfg.din)

    x = np.zeros((Ne, Din), dtype=np.float32)
    m = np.zeros((Ne, Din), dtype=np.uint8)
    src = np.full((Ne,), -1, dtype=np.int8)  # -1=multi/unknown

    # seconds view (for backward compatibility / convenience)
    t_all_s = t_all_ns.astype(np.float64) * 1e-9

    # ---- writer (exact match on ns) ----
    def _write_exact_ns(t_src_ns: np.ndarray, v_src: np.ndarray, idx_cols: Sequence[int], src_id: int) -> Tuple[int, int]:
        """
        Write events whose timestamps exist in t_all_ns (exact ns).
        Returns (n_rows_written, n_entries_written).
        """
        if t_src_ns.size == 0:
            return 0, 0

        t_src_ns = np.asarray(t_src_ns, dtype=np.int64).reshape(-1)
        v_src = np.asarray(v_src, dtype=float)
        if v_src.ndim == 1:
            v_src = v_src.reshape(-1, 1)
        if v_src.shape[0] != t_src_ns.size:
            raise ValueError("[EVENT] time/value 长度不一致")

        idx = np.asarray(list(idx_cols), dtype=int)
        if idx.size != v_src.shape[1]:
            raise ValueError(f"[EVENT] idx_cols size {idx.size} != value dim {v_src.shape[1]}")

        # map source timestamps to union positions
        pos = np.searchsorted(t_all_ns, t_src_ns)
        ok = (pos >= 0) & (pos < Ne) & (t_all_ns[pos] == t_src_ns)

        pos = pos[ok]
        if pos.size == 0:
            return 0, 0

        v_src = v_src[ok].astype(np.float32, copy=False)

        # write
        x[pos[:, None], idx[None, :]] = v_src
        m[pos[:, None], idx[None, :]] = 1

        # src tagging
        for p in pos:
            if src[p] == -1:
                src[p] = src_id
            elif src[p] != src_id:
                src[p] = -1

        n_rows_written = int(np.unique(pos).size)
        n_entries_written = int(pos.size * idx.size)
        return n_rows_written, n_entries_written

    # ---- write sources ----
    rows_pwm, ent_pwm = _write_exact_ns(t_pwm_ns, pwm_val, cfg.pwm_idx, src_id=0)
    rows_imu, ent_imu = _write_exact_ns(t_imu_ns, imu_val, cfg.imu_idx, src_id=1)

    rows_pow, ent_pow = 0, 0
    if t_pow_ns.size and pow_val.size and power_cols:
        if len(power_cols) != len(cfg.power_idx):
            print(
                f"[EVENT][WARN] Power cols={len(power_cols)} != power_idx={len(cfg.power_idx)} "
                f"(cols={power_cols}). 建议预处理阶段统一成 8 维。"
            )
        nmin = min(pow_val.shape[1], len(cfg.power_idx))
        rows_pow, ent_pow = _write_exact_ns(t_pow_ns, pow_val[:, :nmin], list(cfg.power_idx)[:nmin], src_id=3)

    # ---- diagnostics (anti-silent-failure) ----
    def _block_stats(name: str, cols: Sequence[int], t_src_ns: np.ndarray) -> None:
        cols = list(cols)
        if not cols:
            return
        block_m = m[:, cols]
        n_rows = int((block_m.sum(axis=1) > 0).sum())
        n_entries = int(block_m.sum())
        dup = _dup_ratio(t_src_ns) if t_src_ns.size else 0.0
        uniq = int(np.unique(t_src_ns).size) if t_src_ns.size else 0
        tot = int(t_src_ns.size)
        print(
            f"[EVENT][DIAG] {name}: src_events={tot} unique_ts={uniq} dup_ratio={dup:.3f} "
            f"-> written_rows={n_rows} written_entries={n_entries}"
        )
        # “沉默失败”告警：写入行数远小于 unique_ts
        if uniq > 0 and n_rows < int(0.9 * uniq):
            print(
                f"[EVENT][WARN] {name}: written_rows ({n_rows}) << unique_ts ({uniq}). "
                "可能存在时间轴量化/裁剪/异常重复导致覆盖；请重点检查时间戳来源。"
            )

    _block_stats("PWM", cfg.pwm_idx, t_pwm_ns)
    _block_stats("IMU", cfg.imu_idx, t_imu_ns)
    if t_pow_ns.size:
        _block_stats("POWER", list(cfg.power_idx)[: (pow_val.shape[1] if pow_val.size else 0)], t_pow_ns)

    cov = float(m.sum()) / float(max(1, Ne * Din))
    print(f"[EVENT][DIAG] coverage: m.sum={int(m.sum())} / (Ne*Din)={Ne*Din} => {cov:.4f}")

    # 若覆盖率极低，属于典型沉默失败（比如写入全丢）
    if cov < 0.01:
        print(
            "[EVENT][WARN] coverage < 1%. 这通常不正常（除非你刻意做极稀疏事件）。"
            "请检查：t_s 是否为 float 且精度异常？裁剪区间是否过窄？输入列是否缺失？"
        )

    payload: Dict[str, np.ndarray] = {
        "t": t_all_s,
        "t_ns": t_all_ns,
        "x": x,
        "m": m,
        "src": src,
    }

    # ---- DVL as targets ----
    if dvl_path is not None and dvl_path.exists():
        df_dvl = pd.read_csv(dvl_path)
        if "t_s" not in df_dvl.columns:
            raise ValueError(f"[EVENT] DVL CSV 缺少列 't_s': {dvl_path}")

        mask = np.ones(len(df_dvl), dtype=bool)
        if "kind" in df_dvl.columns:
            mask &= (df_dvl["kind"].astype(str) == "BI")
        if "used" in df_dvl.columns:
            mask &= (df_dvl["used"].astype(int) == 1)
        df_dvl = df_dvl.loc[mask].reset_index(drop=True)

        vel_cols = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]
        for c in vel_cols:
            if c not in df_dvl.columns:
                raise KeyError(f"[EVENT] DVL CSV 缺少列 {c!r}: {dvl_path}")

        t_dvl_s = df_dvl["t_s"].to_numpy(dtype=float)
        v_dvl = df_dvl[vel_cols].to_numpy(dtype=float)
        t_dvl_s, v_dvl = _clip_range(t0, t1, t_dvl_s, v_dvl)

        t_dvl_ns = _to_ns(t_dvl_s)

        payload["t_dvl"] = np.asarray(t_dvl_s, dtype=np.float64)
        payload["t_dvl_ns"] = np.asarray(t_dvl_ns, dtype=np.int64)
        payload["v_dvl"] = np.asarray(v_dvl, dtype=np.float32)
        payload["m_dvl"] = np.ones_like(payload["v_dvl"], dtype=np.uint8)

        print(f"[EVENT] DVL targets: Nd={payload['t_dvl'].shape[0]} (BI & used==1 & clipped)")
        print(f"[EVENT][DIAG] DVL dup_ratio={_dup_ratio(t_dvl_ns):.3f}")
    else:
        if cfg.require_dvl:
            raise FileNotFoundError(f"[EVENT] require_dvl=True, 但未提供/找不到 DVL CSV: {dvl_proc_csv}")

    print(f"[EVENT] events: Ne={Ne}  t=[{t_all_s[0]:.3f},{t_all_s[-1]:.3f}]  Din={Din}")
    return payload

def save_event_stream_npz(out_npz: str | Path, payload: Dict[str, np.ndarray]) -> Path:
    """
    保存事件流（npz），建议用于 B1 数据集构建：
      np.savez_compressed(out_npz, **payload)
    """
    out_path = Path(out_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)

    print(f"[EVENT] saved: {out_path}")
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            print(f"  - {k}: shape={v.shape} dtype={v.dtype}")
    return out_path
