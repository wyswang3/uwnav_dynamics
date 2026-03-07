"""
模块名称：训练数据管线输入 NaN 清洗测试

模块职责：
验证训练数据准备阶段会在不修改原始 dataset artifact 的前提下，
把经过 scaler 后仍残留在输入 `X` 中的非有限值清洗为 0.0，
避免稀疏辅助通道的 NaN 直接进入 RNN 并导致训练 loss 变成 NaN；
同时兼容那些仅出现在 `target_mask=False` 位置的目标侧非有限值。

主要功能：
1. 构造带输入侧 NaN 的最小 features/labels 数据集。
2. 运行 `prepare_train_data()`，覆盖 split/scaler 与 DataLoader 主路径。
3. 断言训练 batch 的 `X` 已全部 finite，`Y` 与 `target_mask` 契约保持正常。
4. 验证 masked-out 的 velocity NaN 可被兼容清洗，而活跃监督 NaN 仍会 fail-fast。

数据流：
synthetic features.npz / labels.npz
    ↓
prepare_train_data()
    ↓
split/scaler
    ↓
输入 X 非有限值清洗
    ↓
DataLoader batch assertions

依赖模块：
- numpy
- torch
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.train.data
- uwnav_dynamics.train.data_pipeline
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.train.data import DataConfig
from uwnav_dynamics.train.data_pipeline import prepare_train_data


def test_prepare_train_data_sanitizes_nonfinite_input_features(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n, hist_len, pred_len, din, dout = 12, 4, 3, 25, 9
    rng = np.random.default_rng(0)

    x = rng.normal(size=(n, hist_len, din)).astype(np.float32)
    y = rng.normal(size=(n, pred_len, dout)).astype(np.float32)

    # 模拟稀疏辅助输入：最后一个输入维在整个数据集中都缺测。
    # scaler 会把该维的统计量回退到 mean=0/std=1，但 transform 后仍会保留 NaN；
    # 本测试验证训练数据消费端会把这些 NaN 统一清洗为 0。
    x[:, :, -1] = np.nan
    # 再模拟零散缺测，覆盖“部分缺测”而不只是“整维全缺测”。
    x[0, 0, -2] = np.nan
    x[3, 2, -3] = np.nan

    dvl_mask = np.ones((n, pred_len, 1), dtype=bool)

    np.savez_compressed(data_dir / "features.npz", X=x)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=y,
        dvl_mask=dvl_mask,
        target_cols=np.asarray(
            [
                "AccX_body_mps2",
                "AccY_body_mps2",
                "AccZ_body_mps2",
                "GyroX_body_rad_s",
                "GyroY_body_rad_s",
                "GyroZ_body_rad_s",
                "VelX_state_mps",
                "VelY_state_mps",
                "VelZ_state_mps",
            ],
            dtype=object,
        ),
    )

    cfg = DataConfig(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=0,
    )
    layout = RunLayout(out_dir=tmp_path / "out", variant="nan_sanitize")

    prepared = prepare_train_data(cfg, layout)
    xb, yb, mb = next(iter(prepared.train_loader))

    assert torch.isfinite(xb).all()
    assert torch.isfinite(yb).all()
    assert mb.dtype == torch.bool
    # 整维全缺测的输入列在 z-score 后应被清洗为 0.0（对应 train 均值）。
    assert torch.allclose(xb[..., -1], torch.zeros_like(xb[..., -1]))


def test_prepare_train_data_sanitizes_nonfinite_targets_only_when_masked_out(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n, hist_len, pred_len, din, dout = 12, 4, 3, 25, 9
    rng = np.random.default_rng(1)

    x = rng.normal(size=(n, hist_len, din)).astype(np.float32)
    y = rng.normal(size=(n, pred_len, dout)).astype(np.float32)
    dvl_mask = np.ones((n, pred_len, 1), dtype=bool)

    # train split = 前 6 个窗口。这里模拟一个旧 artifact：
    # dvl_mask 仍然标记为可监督，但 velocity target 实际上是 NaN。
    # 训练侧应先把该 supervision 保守降级为无监督，再做兼容清洗。
    y[0, :, 6:] = np.nan

    np.savez_compressed(data_dir / "features.npz", X=x)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=y,
        dvl_mask=dvl_mask,
        target_cols=np.asarray(
            [
                "AccX_body_mps2",
                "AccY_body_mps2",
                "AccZ_body_mps2",
                "GyroX_body_rad_s",
                "GyroY_body_rad_s",
                "GyroZ_body_rad_s",
                "VelX_state_mps",
                "VelY_state_mps",
                "VelZ_state_mps",
            ],
            dtype=object,
        ),
    )

    cfg = DataConfig(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=0,
    )
    layout = RunLayout(out_dir=tmp_path / "out", variant="masked_target_nan")

    prepared = prepare_train_data(cfg, layout)
    x_train, y_train, m_train = prepared.train_loader.dataset.tensors

    assert torch.isfinite(x_train).all()
    assert torch.isfinite(y_train).all()
    # 旧 artifact 中仅发生在 velocity 稀疏监督位置的 NaN 应先被降级为 masked-out，
    # 然后再兼容清洗为 0。
    masked_vel = ~m_train[..., 6:]
    assert masked_vel.any()
    assert torch.allclose(
        y_train[..., 6:][masked_vel],
        torch.zeros_like(y_train[..., 6:][masked_vel]),
    )


def test_prepare_train_data_rejects_nonfinite_active_targets(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n, hist_len, pred_len, din, dout = 12, 4, 3, 25, 9
    rng = np.random.default_rng(2)

    x = rng.normal(size=(n, hist_len, din)).astype(np.float32)
    y = rng.normal(size=(n, pred_len, dout)).astype(np.float32)
    dvl_mask = np.ones((n, pred_len, 1), dtype=bool)

    # 活跃监督位置仍不允许保留 NaN：这里用 acc 维模拟主监督错误。
    # 与 velocity 稀疏监督不同，acc/gyro 不存在“保守降级为无监督”的兼容路径。
    y[0, 0, 0] = np.nan

    np.savez_compressed(data_dir / "features.npz", X=x)
    np.savez_compressed(
        data_dir / "labels.npz",
        Y=y,
        dvl_mask=dvl_mask,
        target_cols=np.asarray(
            [
                "AccX_body_mps2",
                "AccY_body_mps2",
                "AccZ_body_mps2",
                "GyroX_body_rad_s",
                "GyroY_body_rad_s",
                "GyroZ_body_rad_s",
                "VelX_state_mps",
                "VelY_state_mps",
                "VelZ_state_mps",
            ],
            dtype=object,
        ),
    )

    cfg = DataConfig(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=0,
    )
    layout = RunLayout(out_dir=tmp_path / "out", variant="active_target_nan")

    with pytest.raises(ValueError, match="non-finite values found in active Y supervision"):
        prepare_train_data(cfg, layout)
