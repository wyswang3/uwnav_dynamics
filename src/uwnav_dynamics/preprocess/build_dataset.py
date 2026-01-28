# src/uwnav_dynamics/preprocess/build_dataset.py
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

"""
uwnav_dynamics.preprocess.build_dataset

用途：
  - 读取对齐后的训练基础表（*_train_base.csv）
  - 根据 YAML 的滑动窗口配置构造监督学习数据集 (X, Y)
  - 支持“异步监督”：
      * 在数据集阶段生成 mask：X_mask / Y_mask
      * NaN/Inf 填 0，避免进入网络与标准化
      * 标准化后缺失位置保持 0（等价于“均值处”）
      * Y_mask 可按规则 gate（例如：速度状态维度仅在 has_dvl==1 时监督）
  - 将结果写入 data/processed/<dataset_name>/ 下的 npz + meta.yaml

命令行示例：
  PYTHONPATH=src \
  python -m uwnav_dynamics.preprocess.build_dataset \
    -y configs/dataset/pooltest02_s1.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.preprocess.sliding_window import (
    SlidingWindowConfig,
    SlidingWindowResult,
    sliding_config_from_dict,
)

# =============================================================================
# Config dataclasses
# =============================================================================


@dataclass
class DatasetOutputConfig:
    dir: Path
    normalize: str  # "standard" | "none"


@dataclass
class DatasetConfig:
    name: str
    base_csv: Path
    time_col: str
    sliding_cfg: SlidingWindowConfig
    output: DatasetOutputConfig


# =============================================================================
# YAML parsing
# =============================================================================


def load_dataset_config(yaml_path: Path) -> DatasetConfig:
    yaml_path = yaml_path.expanduser().resolve()
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    if "dataset" not in cfg_raw:
        raise KeyError(f"YAML {yaml_path} missing top-level 'dataset' key.")

    dset = cfg_raw["dataset"]
    name = str(dset.get("name", "unnamed_dataset"))

    base_tbl = dset["base_table"]
    base_csv = Path(base_tbl["csv"]).expanduser().resolve()
    time_col = str(base_tbl.get("time_col", "t_s"))

    sw_dict = dset["sliding_window"]
    sliding_cfg = sliding_config_from_dict(sw_dict)

    out_dict = dset["output"]
    out_dir = Path(out_dict["dir"]).expanduser().resolve()
    normalize = str(out_dict.get("normalize", "standard")).lower()
    if normalize not in ("standard", "none"):
        raise ValueError(f"output.normalize must be 'standard' or 'none', got {normalize!r}")

    return DatasetConfig(
        name=name,
        base_csv=base_csv,
        time_col=time_col,
        sliding_cfg=sliding_cfg,
        output=DatasetOutputConfig(dir=out_dir, normalize=normalize),
    )


# =============================================================================
# Numeric helpers (mask / fill / zscore)
# =============================================================================
def _power_input_indices(input_cols: Sequence[str]) -> List[int]:
    """识别 input_cols 中 P*_W 的维度索引。"""
    cols = list(input_cols)
    out: List[int] = []
    for i, c in enumerate(cols):
        if c.startswith("P") and c.endswith("_W"):
            out.append(i)
    return out


def _slice_base_mask_to_X_windows(
    base_mask: np.ndarray,
    idx0: np.ndarray,
    *,
    hist_len: int,
) -> np.ndarray:
    """
    把 base 表的逐行 mask (n_rows,) 切成 X 的窗口 mask：(N_win, L)
    base_row = idx0[w] + k
    越界 -> 0
    """
    n_rows = int(base_mask.shape[0])
    N_win = int(idx0.shape[0])
    L = int(hist_len)

    out = np.zeros((N_win, L), dtype=np.uint8)
    for w in range(N_win):
        i0 = int(idx0[w])
        for k in range(L):
            row = i0 + k
            if 0 <= row < n_rows and bool(base_mask[row]):
                out[w, k] = 1
    return out


def _mask_and_fill_zero(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 3D 数组 arr=(N,T,D) 生成 finite mask，并把非 finite 填 0。

    Returns:
      arr_filled: float32 (N,T,D)
      mask: uint8 (N,T,D)  1=finite, 0=missing/invalid
    """
    if arr.ndim != 3:
        raise ValueError(f"_mask_and_fill_zero expects 3D (N,T,D), got {arr.shape}")

    finite = np.isfinite(arr)
    mask = finite.astype(np.uint8)
    out = arr.astype(np.float32, copy=True)
    out[~finite] = 0.0
    return out, mask


def _compute_zscore_stats_ignore_zeros(
    arr_filled: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 zscore mean/std，忽略缺失位置（mask==0）。

    说明：
      - 我们已经把缺失位置填 0，为了不把缺失当成真实 0，需要用 mask 忽略掉；
      - 统计在 (N,T) 上进行：对每个维度 D 计算 mean/std。

    Returns:
      mean: (D,) float32
      std:  (D,) float32
    """
    if arr_filled.ndim != 3 or mask.ndim != 3:
        raise ValueError("arr_filled/mask must be 3D")
    if arr_filled.shape != mask.shape:
        raise ValueError(f"shape mismatch: arr{arr_filled.shape} mask{mask.shape}")

    N, T, D = arr_filled.shape
    flat = arr_filled.reshape(N * T, D)
    mflat = mask.reshape(N * T, D).astype(bool)

    mean = np.zeros((D,), dtype=np.float32)
    std = np.ones((D,), dtype=np.float32)

    for d in range(D):
        v = flat[mflat[:, d], d]
        if v.size == 0:
            # 整列没有有效值：退化为 mean=0,std=1
            mean[d] = 0.0
            std[d] = 1.0
            continue
        mean[d] = float(np.mean(v))
        sd = float(np.std(v))
        std[d] = sd if sd > 1e-8 else 1e-8

    return mean, std


def _apply_zscore_keep_missing_zero(
    arr_filled: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    对已填零 arr 做 zscore；缺失位置保持 0。

    有效位置： (x-mean)/std
    缺失位置： 0
    """
    if arr_filled.shape != mask.shape:
        raise ValueError(f"shape mismatch: arr{arr_filled.shape} mask{mask.shape}")
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("mean/std must be 1D with same shape")

    out = (arr_filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    out = out.astype(np.float32, copy=False)
    out[mask == 0] = 0.0
    return out


# =============================================================================
# Feature engineering: state velocity (ffill DVL BI)
# =============================================================================


def _add_state_velocity_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 DVL BI 体坐标速度，构造“状态速度”列：
      VelX_state_mps, VelY_state_mps, VelZ_state_mps

    约定：
      - 源列：VelBx_body_mps, VelBy_body_mps, VelBz_body_mps（DVL BI, FRD）
      - 有 DVL 时刻：等于观测
      - 无 DVL 时刻：ffill
      - 开头无观测：填 0.0（静止初始化）
    """
    vel_src = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]
    if not all(c in df.columns for c in vel_src):
        print("[STATE-VEL] WARNING: no VelB*_body_mps columns found, skip.")
        return df

    df_out = df.copy()

    has_dvl_col: Optional[str] = None
    for cand in ("has_dvl", "has_dvl_bi", "has_dvl_bi_mask"):
        if cand in df_out.columns:
            has_dvl_col = cand
            break
    if has_dvl_col is not None:
        has = (df_out[has_dvl_col].to_numpy(dtype=float) > 0.5)
        print(f"[STATE-VEL] using DVL BI velocity as state, has_dvl_col={has_dvl_col!r}, N_has={int(has.sum())}")
    else:
        print("[STATE-VEL] using DVL BI velocity as state, no has_dvl mask column found.")

    dst_cols = ["VelX_state_mps", "VelY_state_mps", "VelZ_state_mps"]
    for src, dst in zip(vel_src, dst_cols):
        s = pd.to_numeric(df_out[src], errors="coerce")
        s_ff = s.ffill().fillna(0.0)
        df_out[dst] = s_ff.to_numpy(dtype=float)

    return df_out


# =============================================================================
# Async supervision: Y_mask rule (gate velocity-state dims by has_dvl)
# =============================================================================


def _vel_state_target_indices(target_cols: Sequence[str]) -> List[int]:
    """
    识别 target_cols 中 VelX/Y/Z_state_mps 的索引。
    """
    vel_names = {"VelX_state_mps", "VelY_state_mps", "VelZ_state_mps"}
    return [i for i, c in enumerate(list(target_cols)) if c in vel_names]


def _build_y_rule_mask_from_idx0(
    df_base: pd.DataFrame,
    idx0: np.ndarray,
    *,
    hist_len: int,
    pred_len: int,
    target_cols: Sequence[str],
    has_dvl_col: str = "has_dvl",
) -> np.ndarray:
    """
    基于 sw_res.idx0 构造规则 mask，形状严格对齐 Y_raw：(N_win,H,D_out)。

    规则：
      - 速度状态维度（Vel*_state_mps）：仅当该预测时刻 base 行 has_dvl==1 才监督，否则 mask=0
      - 其它维：默认 mask=1
      - 尾部 pad（row 越界）：mask=0（整行）
    """
    n_rows = len(df_base)
    N_win = int(idx0.shape[0])
    H = int(pred_len)
    D = int(len(list(target_cols)))

    rule = np.ones((N_win, H, D), dtype=np.uint8)
    vel_idx = _vel_state_target_indices(target_cols)
    if not vel_idx:
        return rule

    if has_dvl_col not in df_base.columns:
        print(f"[BUILD][YMASK][WARN] {has_dvl_col!r} not in base table; skip vel gating by has_dvl.")
        return rule

    has_dvl = (df_base[has_dvl_col].to_numpy(dtype=float) > 0.5)

    for w in range(N_win):
        i0 = int(idx0[w])
        for k in range(H):
            row = i0 + int(hist_len) + k
            if row < 0 or row >= n_rows:
                rule[w, k, :] = 0
                continue
            if not bool(has_dvl[row]):
                rule[w, k, vel_idx] = 0

    # 诊断打印
    ratio = float(rule[:, :, vel_idx].mean()) if vel_idx else 1.0
    vel_cols = [list(target_cols)[i] for i in vel_idx]
    print(f"[BUILD][YMASK] vel gated by has_dvl: vel_cols={vel_cols}, valid_ratio={ratio:.4f}")
    return rule


# =============================================================================
# Main pipeline
# =============================================================================


def build_dataset_from_config(cfg: DatasetConfig) -> None:
    print(f"[BUILD] Dataset name : {cfg.name}")
    print(f"[BUILD] Base CSV     : {cfg.base_csv}")
    print(f"[BUILD] Time column  : {cfg.time_col}")
    print(f"[BUILD] Output dir   : {cfg.output.dir}")
    print(f"[BUILD] Normalize    : {cfg.output.normalize}")

    if not cfg.base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {cfg.base_csv}")

    cfg.output.dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base table
    df = pd.read_csv(cfg.base_csv)
    if df.empty:
        raise RuntimeError(f"Base CSV is empty: {cfg.base_csv}")

    # 2) Feature engineering: state velocity
    df = _add_state_velocity_cols(df)

    # 3) Sliding windows (do NOT filter by mask here; async supervision in Y_mask)
    sw_res: SlidingWindowResult = _build_windows(df, cfg)

    X_raw = sw_res.X  # (N,L,Din)
    Y_raw = sw_res.Y  # (N,H,Dout)
    print(f"[BUILD] X shape = {X_raw.shape}, Y shape = {Y_raw.shape}")

    # 4) Finite masks + fill zeros
    X_filled, X_mask = _mask_and_fill_zero(X_raw)
    Y_filled, Y_mask_finite = _mask_and_fill_zero(Y_raw)
    # ---- gate power inputs by has_power (avoid stale/invalid power treated as valid) ----
    p_idx = _power_input_indices(cfg.sliding_cfg.input_cols)
    if p_idx:
        if "has_power" not in df.columns:
            print("[BUILD][XMASK][WARN] input has P*_W but base table has no 'has_power'. "
                  "Will not gate power inputs; consider adding has_power in align stage.")
        else:
            has_power_base = (df["has_power"].to_numpy(dtype=float) > 0.5)
            has_power_win = _slice_base_mask_to_X_windows(
                has_power_base.astype(np.uint8),
                sw_res.idx0,
                hist_len=int(cfg.sliding_cfg.hist_len),
            )  # (N,L)

            # only gate the power feature dims
            X_mask[:, :, p_idx] = (X_mask[:, :, p_idx] & has_power_win[:, :, None]).astype(np.uint8)

            ratio = float(X_mask[:, :, p_idx].mean())
            print(f"[BUILD][XMASK] power gated by has_power: dims={len(p_idx)}, valid_ratio={ratio:.4f}")

    # 5) Rule-based Y mask (gate velocity-state dims by has_dvl), then AND with finite
    y_rule = _build_y_rule_mask_from_idx0(
        df_base=df,
        idx0=sw_res.idx0,
        hist_len=int(cfg.sliding_cfg.hist_len),
        pred_len=int(cfg.sliding_cfg.pred_len),
        target_cols=list(cfg.sliding_cfg.target_cols),
        has_dvl_col="has_dvl",
    )

    if y_rule.shape != Y_mask_finite.shape:
        raise ValueError(
            f"[BUILD] y_rule shape {y_rule.shape} != Y_mask_finite {Y_mask_finite.shape}. "
            "This indicates mismatch between sliding windows and mask indexing."
        )

    Y_mask = (Y_mask_finite & y_rule).astype(np.uint8)

    print(f"[BUILD][MASK] X finite ratio        = {float(X_mask.mean()):.4f}")
    print(f"[BUILD][MASK] Y finite ratio        = {float(Y_mask_finite.mean()):.4f}")
    print(f"[BUILD][MASK] Y final ratio (AND)   = {float(Y_mask.mean()):.4f}")

    # 6) Normalization (keep missing == 0)
    if cfg.output.normalize == "standard":
        x_mean, x_std = _compute_zscore_stats_ignore_zeros(X_filled, X_mask)
        y_mean, y_std = _compute_zscore_stats_ignore_zeros(Y_filled, Y_mask)

        X = _apply_zscore_keep_missing_zero(X_filled, X_mask, x_mean, x_std)
        Y = _apply_zscore_keep_missing_zero(Y_filled, Y_mask, y_mean, y_std)
    else:
        X = X_filled
        Y = Y_filled
        x_mean = np.zeros((X.shape[-1],), dtype=np.float32)
        x_std = np.ones((X.shape[-1],), dtype=np.float32)
        y_mean = np.zeros((Y.shape[-1],), dtype=np.float32)
        y_std = np.ones((Y.shape[-1],), dtype=np.float32)

    # 7) Save artifacts
    feat_path = cfg.output.dir / "features.npz"
    label_path = cfg.output.dir / "labels.npz"
    meta_path = cfg.output.dir / "meta.yaml"

    np.savez_compressed(
        feat_path,
        X=X,
        X_mask=X_mask,  # (N,L,Din)
        t0=sw_res.t0,
        idx0=sw_res.idx0,
        input_cols=np.array(list(cfg.sliding_cfg.input_cols), dtype=object),
    )
    np.savez_compressed(
        label_path,
        Y=Y,
        Y_mask=Y_mask,  # (N,H,Dout)
        target_cols=np.array(list(cfg.sliding_cfg.target_cols), dtype=object),
    )

    meta: Dict[str, Any] = {
        "name": cfg.name,
        "base_csv": str(cfg.base_csv),
        "time_col": cfg.time_col,
        "hist_len": int(cfg.sliding_cfg.hist_len),
        "pred_len": int(cfg.sliding_cfg.pred_len),
        "stride": int(cfg.sliding_cfg.stride),
        "input_cols": list(cfg.sliding_cfg.input_cols),
        "target_cols": list(cfg.sliding_cfg.target_cols),
        "input_dim": int(X.shape[-1]),
        "target_dim": int(Y.shape[-1]),
        "valid_mask_col": cfg.sliding_cfg.valid_mask_col,
        "min_valid_ratio": float(cfg.sliding_cfg.min_valid_ratio),
        "drop_incomplete": bool(cfg.sliding_cfg.drop_incomplete),
        "normalize": cfg.output.normalize,
        "x_mean": [float(v) for v in x_mean.reshape(-1)],
        "x_std": [float(v) for v in x_std.reshape(-1)],
        "y_mean": [float(v) for v in y_mean.reshape(-1)],
        "y_std": [float(v) for v in y_std.reshape(-1)],
        "n_rows_base": int(len(df)),
        "n_windows": int(X.shape[0]),
        "mask_semantics": {
            "X_mask": "finite mask after windowing (1=finite, 0=NaN/Inf); X missing filled with 0",
            "Y_mask": "finite mask AND rule mask (vel-state dims gated by has_dvl); Y missing filled with 0",
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"[BUILD] Saved features to: {feat_path}")
    print(f"[BUILD] Saved labels   to: {label_path}")
    print(f"[BUILD] Saved meta     to: {meta_path}")
    print("[BUILD] Done.")


# =============================================================================
# Sliding window wrapper
# =============================================================================


def _build_windows(df: pd.DataFrame, cfg: DatasetConfig) -> SlidingWindowResult:
    from uwnav_dynamics.preprocess.sliding_window import make_sliding_windows

    return make_sliding_windows(df, time_col=cfg.time_col, cfg=cfg.sliding_cfg)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build training dataset (sliding window) from aligned base CSV."
    )
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        help="Path to dataset YAML config (e.g., configs/dataset/pooltest02_s1.yaml)",
    )
    args = parser.parse_args()

    cfg = load_dataset_config(Path(args.yaml))
    build_dataset_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
