# src/uwnav_dynamics/preprocess/build_dataset.py
from __future__ import annotations

"""
uwnav_dynamics.preprocess.build_dataset

用途：
  - 读取对齐好的 100 Hz 训练基础表（例如 *_train_base.csv）
  - 根据 YAML 中的滑动窗口配置构造 (X, Y) 数据集
  - 可选对输入/输出做标准化（z-score）
  - 将结果写入 data/processed/<dataset_name>/ 下的 npz + meta.yaml

命令行示例：
  PYTHONPATH=src \\
  python -m uwnav_dynamics.preprocess.build_dataset \\
    -y configs/dataset/pooltest02_s1.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.preprocess.sliding_window import (
    SlidingWindowConfig,
    SlidingWindowResult,
    sliding_config_from_dict,
)


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


# ---------------------------------------------------------------------
# YAML 解析
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 标准化工具
# ---------------------------------------------------------------------
def _compute_zscore_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 3D 数组 arr=(N, T, D) 做 z-score 所需的 (mean, std)，
    统计维度是 (N,T)，对每个特征维度 D 分别算。
    """
    # 展平时间与样本维度
    flat = arr.reshape(-1, arr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # 防止除 0
    std[std < 1e-8] = 1e-8
    return mean, std


def _apply_zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

# build_dataset.py 顶部附近加：
def _add_state_velocity_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 DVL BI 体坐标速度，构造“状态速度”列：

        VelX_state_mps, VelY_state_mps, VelZ_state_mps

    约定：
      - 源列为：
          VelBx_body_mps, VelBy_body_mps, VelBz_body_mps   （DVL BI，FRD）
      - 状态列语义：
          * 在有 DVL 的时刻：等于当时测得的 v_body
          * 在缺失 DVL 的时刻：沿时间前向保持最近一次 DVL 观测（ffill）
          * 序列开头若完全没有 DVL，则填 0.0（相当于静止初始化）

    注意：
      - 只做“状态定义”，**不做监督掩码**。
        真正做 loss 掩码时应该在训练侧用 has_dvl 之类的列。
      - 如果源列不存在，直接返回原 df 不做修改。
    """
    vel_src = ["VelBx_body_mps", "VelBy_body_mps", "VelBz_body_mps"]
    if not all(c in df.columns for c in vel_src):
        # 没有 DVL 体速度列，直接返回原表
        print("[STATE-VEL] WARNING: no VelB*_body_mps columns found, skip state-velocity construction.")
        return df

    df_out = df.copy()

    # 如果有 has_dvl 之类的掩码，只用于诊断，不在这里改值
    has_dvl_col = None
    for cand in ("has_dvl", "has_dvl_bi", "has_dvl_bi_mask"):
        if cand in df_out.columns:
            has_dvl_col = cand
            break
    if has_dvl_col is not None:
        has_dvl = df_out[has_dvl_col].to_numpy(dtype=float) > 0.5
        n_has = int(has_dvl.sum())
        print(f"[STATE-VEL] using DVL BI velocity as state, has_dvl_col={has_dvl_col!r}, N_has={n_has}")
    else:
        print("[STATE-VEL] using DVL BI velocity as state, no has_dvl mask column found.")

    dst_cols = ["VelX_state_mps", "VelY_state_mps", "VelZ_state_mps"]

    for src, dst in zip(vel_src, dst_cols):
        # 转为 float，保留 NaN 表示“当前时刻没有观测”
        s = pd.to_numeric(df_out[src], errors="coerce")

        # 1) 前向填充：每个时刻的状态速度 = 最近一次已知 DVL 观测
        s_ff = s.ffill()

        # 2) 开头仍为 NaN 的位置（从未出现过观测），初始化为 0.0
        #    也可以换成其它策略，比如保持 NaN，在训练时显式 mask。
        s_ff = s_ff.fillna(0.0)

        df_out[dst] = s_ff.to_numpy(dtype=float)

    return df_out

# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def build_dataset_from_config(cfg: DatasetConfig) -> None:
    print(f"[BUILD] Dataset name : {cfg.name}")
    print(f"[BUILD] Base CSV     : {cfg.base_csv}")
    print(f"[BUILD] Time column  : {cfg.time_col}")
    print(f"[BUILD] Output dir   : {cfg.output.dir}")
    print(f"[BUILD] Normalize    : {cfg.output.normalize}")

    if not cfg.base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {cfg.base_csv}")

    cfg.output.dir.mkdir(parents=True, exist_ok=True)

    # 1) 读训练基础表
    df = pd.read_csv(cfg.base_csv)
    if df.empty:
        raise RuntimeError(f"Base CSV is empty: {cfg.base_csv}")
    # 在这里插入一行：构造“状态速度”列
    df = _add_state_velocity_cols(df)

    # 2) 滑动窗口构造
    sw_res: SlidingWindowResult = None  # type: ignore
    sw_res = __build_windows(df, cfg)

    X_raw = sw_res.X  # (N_win, L, D_in)
    Y_raw = sw_res.Y  # (N_win, H, D_out)

    print(f"[BUILD] X shape = {X_raw.shape}, Y shape = {Y_raw.shape}")

    # 3) 可选标准化
    if cfg.output.normalize == "standard":
        x_mean, x_std = _compute_zscore_stats(X_raw)
        y_mean, y_std = _compute_zscore_stats(Y_raw)

        X = _apply_zscore(X_raw, x_mean, x_std)
        Y = _apply_zscore(Y_raw, y_mean, y_std)
    else:
        X = X_raw
        Y = Y_raw
        x_mean = np.zeros(X.shape[-1], dtype=float)
        x_std = np.ones(X.shape[-1], dtype=float)
        y_mean = np.zeros(Y.shape[-1], dtype=float)
        y_std = np.ones(Y.shape[-1], dtype=float)

    # 4) 写出 features / labels / meta
    feat_path = cfg.output.dir / "features.npz"
    label_path = cfg.output.dir / "labels.npz"
    meta_path = cfg.output.dir / "meta.yaml"

    np.savez_compressed(
        feat_path,
        X=X,
        t0=sw_res.t0,
        idx0=sw_res.idx0,
        input_cols=np.array(list(cfg.sliding_cfg.input_cols), dtype=object),
    )
    np.savez_compressed(
        label_path,
        Y=Y,
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
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "n_rows_base": int(len(df)),
        "n_windows": int(X.shape[0]),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"[BUILD] Saved features to: {feat_path}")
    print(f"[BUILD] Saved labels   to: {label_path}")
    print(f"[BUILD] Saved meta     to: {meta_path}")
    print("[BUILD] Done.")


def __build_windows(df: pd.DataFrame, cfg: DatasetConfig) -> SlidingWindowResult:
    """
    单独抽出来，方便未来在这里插入更多 QA / 掩码逻辑。
    当前版本直接用 df + cfg.sliding_cfg 调 sliding_window。
    """
    return __call_sliding(df, cfg.time_col, cfg.sliding_cfg)


def __call_sliding(
    df: pd.DataFrame,
    time_col: str,
    sw_cfg: SlidingWindowConfig,
) -> SlidingWindowResult:
    from uwnav_dynamics.preprocess.sliding_window import make_sliding_windows

    return make_sliding_windows(df, time_col=time_col, cfg=sw_cfg)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
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
