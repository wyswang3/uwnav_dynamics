from __future__ import annotations

from pathlib import Path

import torch
import yaml

from uwnav_dynamics.train.config import build_from_dict
from uwnav_dynamics.train.runtime import (
    TrainCliOverrides,
    apply_train_overrides,
    reconcile_data_config_for_device,
    save_resolved_train_config,
)


def _minimal_train_dict(tmp_path: Path) -> dict:
    return {
        "run": {
            "name": "demo",
            "seed": 0,
            "device": "cuda",
            "amp": False,
            "out_dir": str(tmp_path / "out"),
            "variant": "baseline",
        },
        "data": {
            "data_dir": str(tmp_path / "data"),
            "batch_size": 32,
            "num_workers": 2,
            "pin_memory": True,
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
        },
        "model": {
            "name": "s1_predictor",
            "din": 25,
            "dout": 9,
            "pred_len": 10,
            "rnn_hidden": 64,
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
                    "hidden_dim": 32,
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
                    "feat_dim": 64,
                    "hidden": 32,
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
            "epochs": 3,
            "eval_every": 1,
            "save_best": True,
            "save_last": True,
            "metric": "val_loss",
        },
    }


def test_apply_train_overrides_is_pure_and_keeps_original_cfg(tmp_path):
    cfg = build_from_dict(_minimal_train_dict(tmp_path))

    updated = apply_train_overrides(
        cfg,
        TrainCliOverrides(
            data_dir=tmp_path / "override_data",
            device="cpu",
            epochs=11,
            batch_size=128,
            num_workers=0,
            pin_memory=False,
            out_dir=tmp_path / "override_out",
            variant="B1",
            seed=123,
            amp=True,
        ),
    )

    assert cfg.data.data_dir == Path(tmp_path / "data")
    assert cfg.run.device == "cuda"
    assert cfg.train.epochs == 3
    assert cfg.data.batch_size == 32
    assert cfg.run.variant == "baseline"
    assert cfg.run.seed == 0
    assert cfg.data.seed == 0

    assert updated.data.data_dir == Path(tmp_path / "override_data")
    assert updated.run.device == "cpu"
    assert updated.train.device == "cpu"
    assert updated.train.epochs == 11
    assert updated.data.batch_size == 128
    assert updated.data.num_workers == 0
    assert updated.data.pin_memory is False
    assert updated.run.out_dir == Path(tmp_path / "override_out")
    assert updated.train.out_dir == Path(tmp_path / "override_out")
    assert updated.run.variant == "B1"
    assert updated.run.seed == 123
    assert updated.data.seed == 123
    assert updated.run.amp is True
    assert updated.train.amp is True


def test_runtime_helpers_capture_effective_config(tmp_path):
    cfg = build_from_dict(_minimal_train_dict(tmp_path))
    cpu_cfg, note = reconcile_data_config_for_device(cfg.data, torch.device("cpu"))
    assert cpu_cfg.pin_memory is False
    assert note is not None

    out_path = tmp_path / "resolved_train.yaml"
    overrides = TrainCliOverrides(
        device="cpu",
        batch_size=64,
        out_dir=tmp_path / "override_out",
        variant="B1_eval",
    )
    save_resolved_train_config(
        out_path,
        cfg,
        source_yaml="configs/train/demo.yaml",
        cli_overrides=overrides,
        requested_device="cuda",
        runtime_device="cpu",
        run_dir=tmp_path / "out" / "baseline",
        split_indices_path=tmp_path / "out" / "baseline" / "split_indices.npz",
        split_strategy="contiguous_v1",
        x_scaler_path=tmp_path / "out" / "baseline" / "scalers" / "x_scaler.npz",
        y_scaler_path=tmp_path / "out" / "baseline" / "scalers" / "y_scaler.npz",
    )
    saved = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    assert saved["run"]["name"] == "demo"
    assert saved["run"]["out_dir"] == str(tmp_path / "out")
    assert saved["data"]["data_dir"] == str(tmp_path / "data")
    assert saved["_meta"]["schema_version"] == "train_resolved_v1"
    assert saved["_meta"]["source_yaml"] == "configs/train/demo.yaml"
    assert saved["_meta"]["cli_overrides"]["device"] == "cpu"
    assert saved["_meta"]["cli_overrides"]["batch_size"] == 64
    assert saved["_meta"]["requested_device"] == "cuda"
    assert saved["_meta"]["runtime_device"] == "cpu"
    assert saved["_meta"]["run_dir"] == str(tmp_path / "out" / "baseline")
    assert saved["_meta"]["split_indices_path"] == str(tmp_path / "out" / "baseline" / "split_indices.npz")
    assert saved["_meta"]["split_strategy"] == "contiguous_v1"
    assert saved["_meta"]["x_scaler_path"] == str(tmp_path / "out" / "baseline" / "scalers" / "x_scaler.npz")
    assert saved["_meta"]["y_scaler_path"] == str(tmp_path / "out" / "baseline" / "scalers" / "y_scaler.npz")
