from __future__ import annotations

import numpy as np

from uwnav_dynamics.dataset.normalize import fit_scaler, transform
from uwnav_dynamics.dataset.split import load_split_indices, make_split_indices, save_split_indices


def test_split_indices_roundtrip_and_disjoint(tmp_path):
    n = 257
    indices = make_split_indices(n=n, seed=42, ratios={"train": 0.7, "val": 0.15, "test": 0.15})

    merged = np.concatenate([indices["train"], indices["val"], indices["test"]], axis=0)
    assert merged.size == n
    assert np.unique(merged).size == n
    assert merged.min() >= 0 and merged.max() < n

    split_path = tmp_path / "split_indices.npz"
    save_split_indices(split_path, indices)
    loaded = load_split_indices(split_path)

    assert np.array_equal(indices["train"], loaded["train"])
    assert np.array_equal(indices["val"], loaded["val"])
    assert np.array_equal(indices["test"], loaded["test"])


def test_scaler_fit_only_train_subset_and_shared_indices(tmp_path):
    rng = np.random.default_rng(123)

    # Build data with trend to make train/full statistics observably different.
    X = rng.normal(size=(200, 5, 4)).astype(np.float32)
    X += np.linspace(-3.0, 3.0, 200, dtype=np.float32).reshape(200, 1, 1)

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
