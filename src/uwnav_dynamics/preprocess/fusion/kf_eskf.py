"""
模块名称：KF / ESKF 状态代理量融合

模块职责：
在现有对齐后的 `train_base.csv` 基础上，
利用 DVL 速度观测抑制 IMU 加速度链路噪声，
并利用高频 IMU 补足速度状态的时间分辨率，
形成统一 100 Hz 主时间轴上的低噪声状态代理量。

主要功能：
1. 把 IMU `roll / pitch / yaw` 对齐到 `train_base.csv` 主时间轴。
2. 实现一个因果 `KF / ESKF` 风格的速度-偏置-姿态融合器。
3. 输出训练可直接使用的 `AccKf / GyroKf / VelKf / RollKf / PitchKf / YawKf` 列。
4. 对 power 等辅助输入执行有限值修复，并在落盘前复用 QA 做 hard fail 审查。

数据流：
aligned train_base.csv + imu_proc.csv
    ↓
attitude interpolation / unwrap
    ↓
causal KF / ESKF fusion
    ↓
finite-column audit
    ↓
train_base_kf_v2.csv

依赖模块：
- numpy
- pandas
- yaml
- uwnav_dynamics.preprocess.qa

备注：
- 当前实现采用工程上可用的“误差状态近似 + 对角协方差”简化形式，
  重点是统一时序、压低噪声、保障训练可用；
- 这里的状态代理量是低噪声观测代理，不是精确物理真值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.preprocess.qa import (
    assert_train_base_qa_pass,
    render_train_base_qa,
    run_train_base_qa,
)


_PWM_COLS: tuple[str, ...] = tuple(f"ch{i}_cmd" for i in range(1, 9))
_POWER_COLS: tuple[str, ...] = tuple(f"P{i}_W" for i in range(8))
_RAW_ACC_COLS: tuple[str, ...] = ("AccX_body_mps2", "AccY_body_mps2", "AccZ_body_mps2")
_RAW_GYRO_COLS: tuple[str, ...] = ("GyroX_body_rad_s", "GyroY_body_rad_s", "GyroZ_body_rad_s")
_RAW_DVL_COLS: tuple[str, ...] = ("VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps")
_ATT_COLS: tuple[str, ...] = ("roll_rad", "pitch_rad", "yaw_rad")
_KF_TARGET_COLS: tuple[str, ...] = (
    "AccKfX_body_mps2",
    "AccKfY_body_mps2",
    "AccKfZ_body_mps2",
    "GyroKfX_body_rad_s",
    "GyroKfY_body_rad_s",
    "GyroKfZ_body_rad_s",
    "VelKfX_body_mps",
    "VelKfY_body_mps",
    "VelKfZ_body_mps",
)
_KF_ATT_CTX_COLS: tuple[str, ...] = (
    "RollKf_rad",
    "PitchKf_rad",
    "YawKf_rad",
    "SinYawKf",
    "CosYawKf",
)


@dataclass(frozen=True)
class KfEskfFusionConfig:
    """KF / ESKF 融合配置。"""
    mode: str = "eskf"  # "kf" | "eskf"
    default_dt_s: float = 0.01
    min_dt_s: float = 1.0e-4
    max_dt_s: float = 0.05

    init_vel_from_first_dvl: bool = True
    fill_power_with_ffill_bfill: bool = True

    accel_lpf_alpha: float = 0.22
    gyro_lpf_alpha: float = 0.18
    attitude_blend: float = 0.10
    vel_accel_blend: float = 0.75

    process_noise_vel: float = 0.20
    process_noise_att: float = 0.05
    meas_noise_dvl: tuple[float, float, float] = (0.03, 0.03, 0.04)
    meas_noise_att: tuple[float, float, float] = (0.02, 0.02, 0.04)

    bias_correction_gain_acc: float = 0.015
    bias_correction_gain_gyro: float = 0.020
    innovation_clip_vel_mps: float = 0.40
    innovation_clip_att_rad: float = 0.25

    max_abs_speed_mps: float = 3.0
    max_abs_bias_acc_mps2: float = 2.5
    max_abs_bias_gyro_rad_s: float = 1.0
    min_var: float = 1.0e-6


def _as_diag3(values: Sequence[Any], *, name: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 values, got {values}")
    out = np.asarray([float(v) for v in values], dtype=float).reshape(3)
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {out}")
    return out


def load_fusion_config(yaml_path: str | Path) -> tuple[KfEskfFusionConfig, Path, Path, Path]:
    """从 fusion YAML 读取配置与关键路径。"""
    path = Path(yaml_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "fusion" not in raw:
        raise KeyError(f"YAML {path} missing top-level 'fusion' key")
    sec = raw["fusion"]
    base_csv = Path(str(sec["base_csv"])).expanduser().resolve()
    imu_proc_csv = Path(str(sec["imu_proc_csv"])).expanduser().resolve()
    out_csv = Path(str(sec["out_csv"])).expanduser().resolve()
    cfg = KfEskfFusionConfig(
        mode=str(sec.get("mode", "eskf")).lower(),
        default_dt_s=float(sec.get("default_dt_s", 0.01)),
        min_dt_s=float(sec.get("min_dt_s", 1.0e-4)),
        max_dt_s=float(sec.get("max_dt_s", 0.05)),
        init_vel_from_first_dvl=bool(sec.get("init_vel_from_first_dvl", True)),
        fill_power_with_ffill_bfill=bool(sec.get("fill_power_with_ffill_bfill", True)),
        accel_lpf_alpha=float(sec.get("accel_lpf_alpha", 0.22)),
        gyro_lpf_alpha=float(sec.get("gyro_lpf_alpha", 0.18)),
        attitude_blend=float(sec.get("attitude_blend", 0.10)),
        vel_accel_blend=float(sec.get("vel_accel_blend", 0.75)),
        process_noise_vel=float(sec.get("process_noise_vel", 0.20)),
        process_noise_att=float(sec.get("process_noise_att", 0.05)),
        meas_noise_dvl=tuple(float(v) for v in sec.get("meas_noise_dvl", (0.03, 0.03, 0.04))),
        meas_noise_att=tuple(float(v) for v in sec.get("meas_noise_att", (0.02, 0.02, 0.04))),
        bias_correction_gain_acc=float(sec.get("bias_correction_gain_acc", 0.015)),
        bias_correction_gain_gyro=float(sec.get("bias_correction_gain_gyro", 0.020)),
        innovation_clip_vel_mps=float(sec.get("innovation_clip_vel_mps", 0.40)),
        innovation_clip_att_rad=float(sec.get("innovation_clip_att_rad", 0.25)),
        max_abs_speed_mps=float(sec.get("max_abs_speed_mps", 3.0)),
        max_abs_bias_acc_mps2=float(sec.get("max_abs_bias_acc_mps2", 2.5)),
        max_abs_bias_gyro_rad_s=float(sec.get("max_abs_bias_gyro_rad_s", 1.0)),
        min_var=float(sec.get("min_var", 1.0e-6)),
    )
    if cfg.mode not in {"kf", "eskf"}:
        raise ValueError(f"Unsupported fusion.mode={cfg.mode!r}; expect 'kf' or 'eskf'")
    return cfg, base_csv, imu_proc_csv, out_csv


def _wrap_pm_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _interp_linear_to_main(t_src: np.ndarray, values: np.ndarray, t_main: np.ndarray, *, name: str) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    t_main = np.asarray(t_main, dtype=float).reshape(-1)
    if vals.shape[0] != t_src.size:
        raise ValueError(f"{name}: time/value length mismatch")
    if t_src.size < 2:
        raise ValueError(f"{name}: source axis too short")
    if np.any(np.diff(t_src) <= 0.0):
        raise ValueError(f"{name}: source time axis must be strictly increasing")
    if (float(t_main[0]) < float(t_src[0])) or (float(t_main[-1]) > float(t_src[-1])):
        raise ValueError(
            f"{name}: main axis out of source range: "
            f"src=[{float(t_src[0]):.6f},{float(t_src[-1]):.6f}] "
            f"main=[{float(t_main[0]):.6f},{float(t_main[-1]):.6f}]"
        )
    out = np.empty((t_main.size, vals.shape[1]), dtype=float)
    for c in range(vals.shape[1]):
        out[:, c] = np.interp(t_main, t_src, vals[:, c])
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"{name}: interpolation produced non-finite values")
    return out


def _interp_angle_to_main(t_src: np.ndarray, angle_src: np.ndarray, t_main: np.ndarray, *, name: str) -> np.ndarray:
    ang = np.asarray(angle_src, dtype=float).reshape(-1)
    if not np.all(np.isfinite(ang)):
        raise ValueError(f"{name}: angle source contains NaN/Inf")
    unwrapped = np.unwrap(ang)
    interp = _interp_linear_to_main(t_src, unwrapped, t_main, name=name).reshape(-1)
    return _wrap_pm_pi(interp)


def _first_valid_row(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    good = np.isfinite(vals).all(axis=1)
    if not good.any():
        return np.zeros(vals.shape[1], dtype=float)
    return vals[int(np.argmax(good))].copy()


def _ensure_numeric_cols(df: pd.DataFrame, cols: Sequence[str], *, fill_value: float) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = float(fill_value)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _fill_power_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.ffill().bfill().fillna(0.0)
        df[c] = s.astype(float)


def _clip_vec(x: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), -float(limit), float(limit))


def _safe_vec(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    fb = np.asarray(fallback, dtype=float).reshape(-1)
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    out = fb.copy()
    out[mask] = arr[mask]
    return out


def _select_initial_velocity(dvl_meas: np.ndarray, dvl_mask: np.ndarray) -> np.ndarray:
    if not np.asarray(dvl_mask, dtype=bool).any():
        return np.zeros(3, dtype=float)
    idx = int(np.argmax(np.asarray(dvl_mask, dtype=bool)))
    z = np.asarray(dvl_meas[idx], dtype=float).reshape(3)
    if not np.all(np.isfinite(z)):
        return np.zeros(3, dtype=float)
    return z


def _fuse_state_proxies(
    *,
    t_s: np.ndarray,
    acc_body: np.ndarray,
    gyro_body: np.ndarray,
    dvl_meas: np.ndarray,
    dvl_mask: np.ndarray,
    attitude_meas: np.ndarray,
    cfg: KfEskfFusionConfig,
) -> dict[str, np.ndarray]:
    t = np.asarray(t_s, dtype=float).reshape(-1)
    acc = np.asarray(acc_body, dtype=float)
    gyro = np.asarray(gyro_body, dtype=float)
    vel_meas = np.asarray(dvl_meas, dtype=float)
    has_dvl = np.asarray(dvl_mask, dtype=bool).reshape(-1)
    att_obs = np.asarray(attitude_meas, dtype=float)

    if acc.shape != (t.size, 3) or gyro.shape != (t.size, 3) or vel_meas.shape != (t.size, 3) or att_obs.shape != (t.size, 3):
        raise ValueError("fusion inputs must all align to shape (N,3)")

    n = t.size
    dt_nom = float(np.median(np.diff(t))) if n > 1 else float(cfg.default_dt_s)
    r_v = np.square(_as_diag3(cfg.meas_noise_dvl, name="meas_noise_dvl"))
    r_att = np.square(_as_diag3(cfg.meas_noise_att, name="meas_noise_att"))

    vel = np.zeros((n, 3), dtype=float)
    acc_f = np.zeros((n, 3), dtype=float)
    gyro_f = np.zeros((n, 3), dtype=float)
    att_f = np.zeros((n, 3), dtype=float)
    vel_var = np.zeros((n, 3), dtype=float)
    acc_bias = np.zeros((n, 3), dtype=float)
    gyro_bias = np.zeros((n, 3), dtype=float)
    dt_since_dvl = np.zeros(n, dtype=float)

    v_prev = _select_initial_velocity(vel_meas, has_dvl) if cfg.init_vel_from_first_dvl else np.zeros(3, dtype=float)
    theta = _first_valid_row(att_obs)
    ba = np.zeros(3, dtype=float)
    bg = np.zeros(3, dtype=float)
    p_v = np.full(3, float(cfg.process_noise_vel), dtype=float)
    p_att = np.full(3, float(cfg.process_noise_att), dtype=float)

    acc_lp = _safe_vec(acc[0], np.zeros(3, dtype=float)) - ba
    gyro_lp = _safe_vec(gyro[0], np.zeros(3, dtype=float)) - bg
    last_dvl_t = t[0] if bool(has_dvl[0]) else float("nan")

    for k in range(n):
        if k == 0:
            dt = max(float(cfg.min_dt_s), min(float(cfg.max_dt_s), float(dt_nom)))
        else:
            dt_raw = float(t[k] - t[k - 1])
            dt = max(float(cfg.min_dt_s), min(float(cfg.max_dt_s), dt_raw))

        acc_raw = _safe_vec(acc[k], acc_lp + ba)
        gyro_raw = _safe_vec(gyro[k], gyro_lp + bg)
        att_meas_k = _safe_vec(att_obs[k], theta)

        theta_pred = _wrap_pm_pi(theta + dt * (gyro_raw - bg))
        att_res = _wrap_pm_pi(att_meas_k - theta_pred)
        att_res = _clip_vec(att_res, cfg.innovation_clip_att_rad)

        if cfg.mode == "eskf":
            p_att = np.maximum(p_att + float(cfg.process_noise_att) * dt, float(cfg.min_var))
            k_att = p_att / np.maximum(p_att + r_att, float(cfg.min_var))
            theta = _wrap_pm_pi(theta_pred + k_att * att_res)
            p_att = np.maximum((1.0 - k_att) * p_att, float(cfg.min_var))
            bg = bg - float(cfg.bias_correction_gain_gyro) * (k_att * att_res) / max(dt, float(cfg.min_dt_s))
        else:
            theta = _wrap_pm_pi(theta_pred + float(cfg.attitude_blend) * att_res)
            bg = bg - float(cfg.bias_correction_gain_gyro) * float(cfg.attitude_blend) * att_res / max(dt, float(cfg.min_dt_s))

        bg = _clip_vec(bg, cfg.max_abs_bias_gyro_rad_s)
        gyro_corr = gyro_raw - bg
        gyro_lp = (1.0 - float(cfg.gyro_lpf_alpha)) * gyro_lp + float(cfg.gyro_lpf_alpha) * gyro_corr

        v_pred = v_prev + dt * (acc_raw - ba - np.cross(gyro_lp, v_prev))
        p_v = np.maximum(p_v + float(cfg.process_noise_vel) * dt, float(cfg.min_var))

        acc_inst = acc_raw - ba
        if bool(has_dvl[k]) and np.all(np.isfinite(vel_meas[k])):
            z_v = vel_meas[k]
            vel_res = _clip_vec(z_v - v_pred, cfg.innovation_clip_vel_mps)
            k_v = p_v / np.maximum(p_v + r_v, float(cfg.min_var))
            v_new = v_pred + k_v * vel_res
            p_v = np.maximum((1.0 - k_v) * p_v, float(cfg.min_var))
            ba = ba - float(cfg.bias_correction_gain_acc) * (k_v * vel_res) / max(dt, float(cfg.min_dt_s))
            ba = _clip_vec(ba, cfg.max_abs_bias_acc_mps2)
            acc_from_vel = (v_new - v_prev) / max(dt, float(cfg.min_dt_s)) + np.cross(gyro_lp, v_prev)
            blend = float(cfg.vel_accel_blend) * float(np.clip(np.mean(k_v), 0.0, 1.0))
            acc_inst = (1.0 - blend) * (acc_raw - ba) + blend * acc_from_vel
            last_dvl_t = float(t[k])
        else:
            v_new = v_pred

        v_new = _clip_vec(v_new, cfg.max_abs_speed_mps)
        acc_lp = (1.0 - float(cfg.accel_lpf_alpha)) * acc_lp + float(cfg.accel_lpf_alpha) * acc_inst

        vel[k] = v_new
        acc_f[k] = acc_lp
        gyro_f[k] = gyro_lp
        att_f[k] = theta
        vel_var[k] = p_v
        acc_bias[k] = ba
        gyro_bias[k] = bg
        dt_since_dvl[k] = 0.0 if np.isfinite(last_dvl_t) and bool(has_dvl[k]) else (float(t[k] - last_dvl_t) if np.isfinite(last_dvl_t) else np.inf)

        v_prev = v_new

    return {
        "acc_fused": acc_f,
        "gyro_fused": gyro_f,
        "vel_fused": vel,
        "att_fused": att_f,
        "vel_var": vel_var,
        "acc_bias": acc_bias,
        "gyro_bias": gyro_bias,
        "dt_since_dvl": dt_since_dvl,
    }


def build_fused_train_base_from_frames(
    base_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    *,
    cfg: KfEskfFusionConfig,
) -> pd.DataFrame:
    """基于对齐后的 base 表和 IMU 预处理表构造融合后的训练基础表。"""
    if "t_s" not in base_df.columns:
        raise KeyError("base_df missing 't_s'")
    if "t_s" not in imu_df.columns:
        raise KeyError("imu_df missing 't_s'")

    for c in _RAW_ACC_COLS + _RAW_GYRO_COLS:
        if c not in base_df.columns:
            raise KeyError(f"base_df missing required column {c!r}")
    for c in _PWM_COLS:
        if c not in base_df.columns:
            raise KeyError(f"base_df missing required PWM column {c!r}")
    for c in _ATT_COLS:
        if c not in imu_df.columns:
            raise KeyError(f"imu_df missing required attitude column {c!r}")

    df = base_df.copy()
    _ensure_numeric_cols(df, _POWER_COLS, fill_value=0.0)
    if cfg.fill_power_with_ffill_bfill:
        _fill_power_cols(df, _POWER_COLS)

    t_main = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(t_main)) or np.any(np.diff(t_main) <= 0.0):
        raise ValueError("base_df.t_s must be finite and strictly increasing")

    t_imu = pd.to_numeric(imu_df["t_s"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(t_imu)) or np.any(np.diff(t_imu) <= 0.0):
        raise ValueError("imu_df.t_s must be finite and strictly increasing")

    roll_main = _interp_linear_to_main(
        t_imu,
        pd.to_numeric(imu_df["roll_rad"], errors="coerce").to_numpy(dtype=float),
        t_main,
        name="roll_interp",
    ).reshape(-1)
    pitch_main = _interp_linear_to_main(
        t_imu,
        pd.to_numeric(imu_df["pitch_rad"], errors="coerce").to_numpy(dtype=float),
        t_main,
        name="pitch_interp",
    ).reshape(-1)
    yaw_main = _interp_angle_to_main(
        t_imu,
        pd.to_numeric(imu_df["yaw_rad"], errors="coerce").to_numpy(dtype=float),
        t_main,
        name="yaw_interp",
    ).reshape(-1)
    att_main = np.stack([roll_main, pitch_main, yaw_main], axis=1)

    dvl_mask_col = "dvl_mask" if "dvl_mask" in df.columns else ("has_dvl" if "has_dvl" in df.columns else None)
    if dvl_mask_col is None:
        dvl_mask = np.zeros(len(df), dtype=bool)
    else:
        dvl_mask = pd.to_numeric(df[dvl_mask_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5

    dvl_meas = np.full((len(df), 3), np.nan, dtype=float)
    if all(c in df.columns for c in _RAW_DVL_COLS):
        dvl_meas = df[list(_RAW_DVL_COLS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    fused = _fuse_state_proxies(
        t_s=t_main,
        acc_body=df[list(_RAW_ACC_COLS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        gyro_body=df[list(_RAW_GYRO_COLS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        dvl_meas=dvl_meas,
        dvl_mask=dvl_mask,
        attitude_meas=att_main,
        cfg=cfg,
    )

    df["RollKf_rad"] = fused["att_fused"][:, 0]
    df["PitchKf_rad"] = fused["att_fused"][:, 1]
    df["YawKf_rad"] = _wrap_pm_pi(fused["att_fused"][:, 2])
    df["SinYawKf"] = np.sin(df["YawKf_rad"].to_numpy(dtype=float))
    df["CosYawKf"] = np.cos(df["YawKf_rad"].to_numpy(dtype=float))

    df["AccKfX_body_mps2"] = fused["acc_fused"][:, 0]
    df["AccKfY_body_mps2"] = fused["acc_fused"][:, 1]
    df["AccKfZ_body_mps2"] = fused["acc_fused"][:, 2]
    df["GyroKfX_body_rad_s"] = fused["gyro_fused"][:, 0]
    df["GyroKfY_body_rad_s"] = fused["gyro_fused"][:, 1]
    df["GyroKfZ_body_rad_s"] = fused["gyro_fused"][:, 2]
    df["VelKfX_body_mps"] = fused["vel_fused"][:, 0]
    df["VelKfY_body_mps"] = fused["vel_fused"][:, 1]
    df["VelKfZ_body_mps"] = fused["vel_fused"][:, 2]

    df["HasDvlUpdate"] = dvl_mask.astype(int)
    df["DtSinceDvl_s"] = fused["dt_since_dvl"]
    df["VelKfVarX_body_mps2"] = fused["vel_var"][:, 0]
    df["VelKfVarY_body_mps2"] = fused["vel_var"][:, 1]
    df["VelKfVarZ_body_mps2"] = fused["vel_var"][:, 2]
    df["AccBiasKfX_body_mps2"] = fused["acc_bias"][:, 0]
    df["AccBiasKfY_body_mps2"] = fused["acc_bias"][:, 1]
    df["AccBiasKfZ_body_mps2"] = fused["acc_bias"][:, 2]
    df["GyroBiasKfX_body_rad_s"] = fused["gyro_bias"][:, 0]
    df["GyroBiasKfY_body_rad_s"] = fused["gyro_bias"][:, 1]
    df["GyroBiasKfZ_body_rad_s"] = fused["gyro_bias"][:, 2]

    required_finite_cols = list(_PWM_COLS) + list(_POWER_COLS) + list(_KF_TARGET_COLS) + list(_KF_ATT_CTX_COLS)
    qa_report = run_train_base_qa(
        df,
        stage="train_base_kf",
        time_col="t_s",
        dense_target_cols=_KF_TARGET_COLS,
        key_stat_cols=_KF_TARGET_COLS,
        required_finite_cols=required_finite_cols,
    )
    print(render_train_base_qa(qa_report))
    assert_train_base_qa_pass(qa_report)
    return df


def save_fused_train_base_from_csv(
    *,
    base_csv: str | Path,
    imu_proc_csv: str | Path,
    out_csv: str | Path,
    cfg: KfEskfFusionConfig,
) -> Path:
    """读取 CSV，执行融合，并把融合后的训练基础表落盘。"""
    base_path = Path(base_csv).expanduser().resolve()
    imu_path = Path(imu_proc_csv).expanduser().resolve()
    out_path = Path(out_csv).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"base_csv not found: {base_path}")
    if not imu_path.exists():
        raise FileNotFoundError(f"imu_proc_csv not found: {imu_path}")

    base_df = pd.read_csv(base_path)
    imu_df = pd.read_csv(imu_path)
    fused_df = build_fused_train_base_from_frames(base_df, imu_df, cfg=cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(out_path, index=False)
    print(f"[FUSION] saved fused train base to: {out_path}")
    print(f"[FUSION] shape={fused_df.shape}, mode={cfg.mode}")
    return out_path
