"""
模块名称：split / scaler 真源与复用测试

模块职责：
验证 PR2 建立的 split / scaler 单一真源契约，
并确保评估阶段继续复用训练阶段的 run-scoped artifact。

主要功能：
1. 验证 contiguous split 语义与 metadata 落盘。
2. 验证 scaler 只由 train split 拟合。
3. 验证 eval 侧复用训练产生的 split / scaler artifact。

数据流：
模拟数据集 + train yaml
    ↓
train.data_pipeline / build_eval_config / evaluate_once
    ↓
split/scaler/eval artifact
    ↓
契约断言

依赖模块：
- uwnav_dynamics.dataset.split
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.eval.config
- uwnav_dynamics.eval.evaluate

备注：
- 本测试也作为 PR3 的回归保护，确保 eval 配置收缩后仍不破坏 PR2 契约。
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import yaml

from uwnav_dynamics.dataset.normalize import fit_scaler, transform
from uwnav_dynamics.dataset.split import (
    DEFAULT_SPLIT_STRATEGY,
    load_split_indices,
    make_split_indices,
    save_split_indices,
)
from uwnav_dynamics.eval.config import build_eval_config
from uwnav_dynamics.eval.evaluate import evaluate_once
from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data_pipeline import prepare_train_data


def test_split_indices_roundtrip_are_disjoint_and_contiguous(tmp_path):
    n = 257
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        indices = make_split_indices(n=n, seed=42, ratios={"train": 0.7, "val": 0.15, "test": 0.15})

    assert any("seed is ignored" in str(w.message) for w in caught)

    # 断言 1：样本索引集合不重叠，避免同一个窗口同时进入多个 split。
    merged = np.concatenate([indices["train"], indices["val"], indices["test"]], axis=0)
    assert merged.size == n
    assert np.unique(merged).size == n
    assert merged.min() >= 0 and merged.max() < n

    # 断言 2：当前 canonical 语义是时间顺序 contiguous split，而非随机打散。
    assert np.array_equal(indices["train"], np.arange(0, len(indices["train"]), dtype=np.int64))
    assert np.array_equal(
        indices["val"],
        np.arange(len(indices["train"]), len(indices["train"]) + len(indices["val"]), dtype=np.int64),
    )
    assert np.array_equal(
        indices["test"],
        np.arange(
            len(indices["train"]) + len(indices["val"]),
            n,
            dtype=np.int64,
        ),
    )

    split_path = tmp_path / "split_indices.npz"
    save_split_indices(split_path, indices)
    loaded = load_split_indices(split_path)

    assert np.array_equal(indices["train"], loaded["train"])
    assert np.array_equal(indices["val"], loaded["val"])
    assert np.array_equal(indices["test"], loaded["test"])
    with np.load(split_path, allow_pickle=False) as z:
        assert str(np.asarray(z["split_strategy"]).item()) == DEFAULT_SPLIT_STRATEGY


def test_scaler_fit_only_train_subset_and_shared_indices(tmp_path):
    rng = np.random.default_rng(123)

    # Build data with trend to make train/full statistics observably different.
    X = rng.normal(size=(200, 5, 4)).astype(np.float32)
    X += np.linspace(-3.0, 3.0, 200, dtype=np.float32).reshape(200, 1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        indices = make_split_indices(n=X.shape[0], seed=7, ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    split_path = tmp_path / "split_indices.npz"
    save_split_indices(split_path, indices)

    # Simulate train side loading split file.
    train_loaded = load_split_indices(split_path)
    train_idx = train_loaded["train"]
    scaler = fit_scaler(X[train_idx])

    # Prove fit statistics are from train subset only.
    train_flat = X[train_idx].reshape(-1, X.shape[-1])
    full_flat = X.reshape(-1, X.shape[-1])
    train_mean = np.nanmean(train_flat, axis=0)
    full_mean = np.nanmean(full_flat, axis=0)

    assert np.allclose(scaler["mean"], train_mean, atol=1e-6)
    assert not np.allclose(scaler["mean"], full_mean, atol=1e-6)

    # Simulate eval side loading the same indices file: must match train side.
    eval_loaded = load_split_indices(split_path)
    assert np.array_equal(train_loaded["train"], eval_loaded["train"])
    assert np.array_equal(train_loaded["val"], eval_loaded["val"])
    assert np.array_equal(train_loaded["test"], eval_loaded["test"])

    # Transform on val/test should not mutate the fitted scaler.
    mean_before = scaler["mean"].copy()
    std_before = scaler["std"].copy()
    _ = transform(X[eval_loaded["val"]], scaler)
    _ = transform(X[eval_loaded["test"]], scaler)
    assert np.array_equal(mean_before, scaler["mean"])
    assert np.array_equal(std_before, scaler["std"])


def test_eval_reuses_train_split_and_scaler_artifacts(tmp_path):
    train_yaml = tmp_path / "train.yaml"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    X = np.random.default_rng(0).normal(size=(24, 5, 25)).astype(np.float32)
    Y = np.random.default_rng(1).normal(size=(24, 4, 9)).astype(np.float32)
    np.savez_compressed(data_dir / "features.npz", X=X)
    np.savez_compressed(data_dir / "labels.npz", Y=Y)

    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "reuse_demo",
                    "seed": 0,
                    "device": "cpu",
                    "amp": False,
                    "out_dir": str(tmp_path / "out"),
                    "variant": "B0_reuse",
                },
                "data": {
                    "data_dir": str(data_dir),
                    "batch_size": 8,
                    "num_workers": 0,
                    "pin_memory": False,
                    "split": {
                        "train_ratio": 0.7,
                        "val_ratio": 0.15,
                    },
                },
                "model": {
                    "name": "s1_predictor",
                    "din": 25,
                    "dout": 9,
                    "pred_len": 4,
                    "rnn_hidden": 16,
                    "rnn_layers": 1,
                    "dropout": 0.0,
                    "u_in_idx": list(range(0, 8)),
                    "y_in_idx": list(range(8, 17)),
                    "use_thruster_as_replacement": True,
                    "use_hydro_feat": True,
                    "blocks": {
                        "thruster_lag": {
                            "enabled": False,
                            "normalize_input": True,
                            "pwm_center": 7.5,
                            "pwm_half_range": 2.5,
                            "deadzone": 0.05,
                            "sat_gain": 2.0,
                            "learn_tau": True,
                            "tau_init": 0.08,
                            "tau_min": 0.01,
                            "per_channel_tau": True,
                            "dt": 0.01,
                        },
                        "hydro_ssm": {
                            "enabled": False,
                            "hidden_dim": 16,
                            "lambda_init": 0.85,
                            "act": "tanh",
                            "per_dim_lambda": True,
                        },
                        "damping": {
                            "enabled": False,
                            "v_start": 6,
                            "v_dim": 3,
                            "mode": "simple",
                            "d_init": 0.5,
                            "mlp_hidden": 16,
                        },
                        "uncertainty": {
                            "enabled": False,
                            "feat_dim": 32,
                            "hidden": 16,
                            "logvar_min": -10.0,
                            "logvar_max": 6.0,
                        },
                    },
                },
                "rollout": {
                    "y0_source": "x_last_state",
                    "mode": "delta_cumsum",
                },
                "loss": {
                    "type": "nll_diag",
                    "logvar_clip": [-10.0, 6.0],
                },
                "optim": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "grad_clip": 1.0,
                },
                "train": {
                    "epochs": 1,
                    "eval_every": 1,
                    "save_best": True,
                    "save_last": True,
                    "metric": "val_loss",
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    cfg_train = load_train_config(train_yaml)
    layout = RunLayout(out_dir=cfg_train.run.out_dir, variant=cfg_train.run.variant)

    prepare_train_data(cfg_train.data, layout)
    split_mtime_before = layout.split_indices_path.stat().st_mtime_ns
    x_scaler_mtime_before = layout.x_scaler_path.stat().st_mtime_ns
    y_scaler_mtime_before = layout.y_scaler_path.stat().st_mtime_ns

    ckpt_path = layout.run_dir / "best.pth"
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": S1Predictor(cfg_train.model).state_dict()}, ckpt_path)

    cfg_eval, cfg_model = build_eval_config(
        train_yaml=train_yaml,
        ckpt=ckpt_path,
        split="test",
        device="cpu",
        batch_size=4,
        out_dir=tmp_path / "eval_test",
        save_samples=2,
    )
    res = evaluate_once(cfg_eval=cfg_eval, cfg_model=cfg_model)

    assert res["n_eval"] > 0
    assert layout.split_indices_path.stat().st_mtime_ns == split_mtime_before
    assert layout.x_scaler_path.stat().st_mtime_ns == x_scaler_mtime_before
    assert layout.y_scaler_path.stat().st_mtime_ns == y_scaler_mtime_before
