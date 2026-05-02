# SPDX-License-Identifier: AGPL-3.0-or-later

"""
模块名称：训练数据集构建

模块职责：
负责从对齐后的 `train_base.csv` 构造滑动窗口数据集，
并将 features / labels / meta artifact 落盘到 processed 数据目录。

主要功能：
1. 读取基础训练表并构造状态速度列。
2. 按滑窗配置生成 `(X, Y)` 数据集与 mask artifact。
3. 在落盘前对 target 列执行最小 fail-fast 数据质量检查。

数据流：
train_base.csv
    ↓
状态速度列构造 / target finite guard
    ↓
sliding window
    ↓
features.npz / labels.npz / meta.yaml

依赖模块：
- numpy
- pandas
- yaml
- uwnav_dynamics.preprocess.sliding_window

备注：
- 本模块不放宽 dense supervision 契约；
- 若 target 列仍含 NaN/Inf，应先修复上游 align，再重新构建数据集。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from uwnav_dynamics.experiment.paths import infer_repo_root, resolve_config_path, resolve_repo_output_path

from uwnav_dynamics.preprocess.qa import (
    assert_train_base_qa_pass,
    render_train_base_qa,
    run_train_base_qa,
)
from uwnav_dynamics.preprocess.sliding_window import (
    SlidingWindowConfig,
    SlidingWindowResult,
    sliding_config_from_dict,
)


@dataclass
class DatasetOutputConfig:
    """数据集构建输出目录与标准化策略配置。"""
    dir: Path
    normalize: str  # "standard" | "none"


@dataclass
class DatasetConfig:
    """训练数据集构建所需的完整配置。"""
    name: str
    base_csv: Path
    time_col: str
    sliding_cfg: SlidingWindowConfig
    output: DatasetOutputConfig


def _resolve_mask_series(
    df: pd.DataFrame,
    *,
    candidates: tuple[str, ...],
    name: str,
) -> np.ndarray:
    """
    从 DataFrame 中解析 0/1 mask 序列；若找不到列则返回全 0。
    """
    for c in candidates:
        if c in df.columns:
            m = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            m = np.where(np.isfinite(m) & (m > 0.5), 1.0, 0.0)
            print(f"[BUILD][MASK] use {name} from column: {c!r}")
            return m.astype(np.float32, copy=False)

    print(f"[BUILD][MASK] no column found for {name}, fallback to all-zero mask.")
    return np.zeros(len(df), dtype=np.float32)


def _build_mask_windows_from_idx0(
    mask_1d: np.ndarray,
    idx0: np.ndarray,
    *,
    hist_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    基于 sliding_window 输出的 idx0，构建历史窗与预测窗 mask。
    """
    m = np.asarray(mask_1d, dtype=np.float32).reshape(-1)
    i0_arr = np.asarray(idx0, dtype=np.int64).reshape(-1)
    n_win = int(i0_arr.shape[0])

    mh = np.zeros((n_win, int(hist_len), 1), dtype=np.float32)
    mp = np.zeros((n_win, int(pred_len), 1), dtype=np.float32)

    for k, i0 in enumerate(i0_arr):
        ih0 = int(i0)
        ih1 = ih0 + int(hist_len)
        ip0 = ih1
        ip1 = ip0 + int(pred_len)
        mh[k, :, 0] = m[ih0:ih1]
        mp[k, :, 0] = m[ip0:ip1]
    return mh, mp


# ---------------------------------------------------------------------
# YAML 解析
# ---------------------------------------------------------------------
def load_dataset_config(yaml_path: Path) -> DatasetConfig:
    """从 dataset YAML 读取并构造数据集构建配置。"""
    yaml_path = yaml_path.expanduser().resolve()
    repo_root = infer_repo_root(yaml_path)
    config_dir = yaml_path.parent
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    if "dataset" not in cfg_raw:
        raise KeyError(f"YAML {yaml_path} missing top-level 'dataset' key.")

    dset = cfg_raw["dataset"]
    name = str(dset.get("name", "unnamed_dataset"))

    base_tbl = dset["base_table"]
    base_csv = resolve_config_path(
        Path(base_tbl["csv"]),
        repo_root=repo_root,
        config_dir=config_dir,
    )
    time_col = str(base_tbl.get("time_col", "t_s"))

    sw_dict = dset["sliding_window"]
    sliding_cfg = sliding_config_from_dict(sw_dict)

    out_dict = dset["output"]
    out_dir = resolve_repo_output_path(
        Path(out_dict["dir"]),
        repo_root=repo_root,
        config_dir=config_dir,
        field_name="dataset.output.dir",
    )
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
    for cand in ("dvl_mask", "has_dvl", "has_dvl_bi", "has_dvl_bi_mask"):
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
    """按配置从 `train_base.csv` 构建训练所需的滑窗数据集 artifact。"""
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
    df = _add_state_velocity_cols(df)
    # 滑窗前先做 fail-fast QA；一旦 dense target 或时间轴异常，直接阻断数据集构建。
    qa_report = run_train_base_qa(
        df,
        stage="base_csv",
        time_col=cfg.time_col,
        dense_target_cols=cfg.sliding_cfg.target_cols,
        hist_len=int(cfg.sliding_cfg.hist_len),
        pred_len=int(cfg.sliding_cfg.pred_len),
        key_stat_cols=cfg.sliding_cfg.target_cols,
    )
    print(render_train_base_qa(qa_report))
    assert_train_base_qa_pass(qa_report)

    # 2) 滑动窗口构造
    sw_res: SlidingWindowResult = None  # type: ignore
    sw_res = __build_windows(df, cfg)

    X_raw = sw_res.X  # (N_win, L, D_in)
    Y_raw = sw_res.Y  # (N_win, H, D_out)

    print(f"[BUILD] X shape = {X_raw.shape}, Y shape = {Y_raw.shape}")

    # 只写原始窗口，不在这里做全量标准化；
    # 真正的 scaler 必须等 train split 确定后再拟合，避免数据泄漏。
    if cfg.output.normalize != "none":
        print(
            "[BUILD] NOTE: output.normalize is ignored here; "
            "scaler fitting has moved to train split in training/evaluation stage."
        )
    X = X_raw
    Y = Y_raw

    # mask 也必须按滑窗起点 `idx0` 对齐切片，这样 train/eval 才能和 `Y` 同步消费。
    dvl_mask_1d = _resolve_mask_series(
        df,
        candidates=("dvl_mask", "has_dvl", "has_dvl_bi", "has_dvl_bi_mask"),
        name="dvl_mask",
    )
    power_mask_1d = _resolve_mask_series(
        df,
        candidates=("power_mask", "has_power"),
        name="power_mask",
    )
    dvl_mask_hist, dvl_mask_pred = _build_mask_windows_from_idx0(
        dvl_mask_1d,
        sw_res.idx0,
        hist_len=int(cfg.sliding_cfg.hist_len),
        pred_len=int(cfg.sliding_cfg.pred_len),
    )
    if str(cfg.sliding_cfg.label_dvl_mask_mode) == "all_true":
        dvl_mask_pred = np.ones_like(dvl_mask_pred, dtype=np.float32)
    power_mask_hist, power_mask_pred = _build_mask_windows_from_idx0(
        power_mask_1d,
        sw_res.idx0,
        hist_len=int(cfg.sliding_cfg.hist_len),
        pred_len=int(cfg.sliding_cfg.pred_len),
    )

    # features / labels 放数组大件，meta.yaml 只放可读性更强的索引与摘要信息。
    feat_path = cfg.output.dir / "features.npz"
    label_path = cfg.output.dir / "labels.npz"
    meta_path = cfg.output.dir / "meta.yaml"

    np.savez_compressed(
        feat_path,
        X=X,
        t0=sw_res.t0,
        idx0=sw_res.idx0,
        input_cols=np.array(list(cfg.sliding_cfg.input_cols), dtype=object),
        dvl_mask_hist=dvl_mask_hist,
        power_mask_hist=power_mask_hist,
    )
    np.savez_compressed(
        label_path,
        Y=Y,
        target_cols=np.array(list(cfg.sliding_cfg.target_cols), dtype=object),
        dvl_mask=dvl_mask_pred,
        power_mask=power_mask_pred,
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
        "mask_keys": {
            "features": ["dvl_mask_hist", "power_mask_hist"],
            "labels": ["dvl_mask", "power_mask"],
        },
        "dvl_mask_available_ratio": float(dvl_mask_1d.mean()) if dvl_mask_1d.size > 0 else 0.0,
        "power_mask_available_ratio": float(power_mask_1d.mean()) if power_mask_1d.size > 0 else 0.0,
        "valid_mask_col": cfg.sliding_cfg.valid_mask_col,
        "min_valid_ratio": float(cfg.sliding_cfg.min_valid_ratio),
        "label_dvl_mask_mode": str(cfg.sliding_cfg.label_dvl_mask_mode),
        "drop_incomplete": bool(cfg.sliding_cfg.drop_incomplete),
        "normalize": "none (deferred to train split scaler)",
        "normalize_requested": cfg.output.normalize,
        "normalize_applied": "none (deferred to train split scaler)",
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
    """训练数据集构建命令行入口。"""
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
    repo_root = infer_repo_root(Path(args.yaml))
    cfg = replace(
        cfg,
        output=DatasetOutputConfig(
            dir=resolve_repo_output_path(
                cfg.output.dir,
                repo_root=repo_root,
                config_dir=Path(args.yaml).expanduser().resolve().parent,
                field_name="dataset.output.dir",
            ),
            normalize=cfg.output.normalize,
        ),
    )
    build_dataset_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
