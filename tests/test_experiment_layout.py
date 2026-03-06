from __future__ import annotations

from pathlib import Path

import yaml

from uwnav_dynamics.cli.utils import resolve_run_out_dir
from uwnav_dynamics.experiment.layout import load_yaml_dict, run_layout_from_mapping, run_layout_from_train_yaml


def test_run_layout_matches_existing_cli_contract(tmp_path):
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "demo",
                    "out_dir": "out/ckpts/demo",
                    "variant": "B0_baseline",
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    cfg = load_yaml_dict(train_yaml)
    layout_from_mapping = run_layout_from_mapping(cfg["run"])
    layout_from_yaml = run_layout_from_train_yaml(train_yaml)

    assert layout_from_mapping.out_dir == Path("out/ckpts/demo")
    assert layout_from_mapping.variant == "B0_baseline"
    assert layout_from_mapping.run_dir == Path("out/ckpts/demo") / "B0_baseline"
    assert layout_from_mapping.split_indices_path == layout_from_mapping.run_dir / "split_indices.npz"
    assert layout_from_mapping.x_scaler_path == layout_from_mapping.run_dir / "scalers" / "x_scaler.npz"
    assert layout_from_mapping.y_scaler_path == layout_from_mapping.run_dir / "scalers" / "y_scaler.npz"
    assert layout_from_mapping.eval_dir("test") == layout_from_mapping.run_dir / "eval_test"

    assert layout_from_yaml == layout_from_mapping
    assert resolve_run_out_dir(train_yaml) == (Path("out/ckpts/demo"), "B0_baseline")
