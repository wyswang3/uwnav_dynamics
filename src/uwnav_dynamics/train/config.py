from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Iterable

import yaml

from uwnav_dynamics.train.data import DataConfig
from uwnav_dynamics.train.trainer import TrainConfig
from uwnav_dynamics.models.nets.s1_predictor import (
    S1PredictorConfig,
    BlocksConfig,
    ThrusterLagConfig,
    HydroSSMConfig,
    DampingConfig,
    UncertaintyConfig,
)


# =============================================================================
# YAML -> dataclasses (Style A, strict schema)
# =============================================================================

@dataclass(frozen=True)
class LossConfig:
    type: str = "nll_diag"
    logvar_clip_min: float = -10.0
    logvar_clip_max: float = 6.0


@dataclass(frozen=True)
class RolloutConfig:
    y0_source: str = "x_last_state"   # "x_last_state" only for v0
    mode: str = "delta_cumsum"        # "delta_cumsum" only for v0


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int = 0
    device: str = "cuda"
    amp: bool = False
    out_dir: Path = Path("out/ckpts/s1_baseline")
    variant: str = "default"   # +++ 新增


@dataclass(frozen=True)
class TrainYamlConfig:
    run: RunConfig
    data: DataConfig
    model: S1PredictorConfig
    rollout: RolloutConfig
    loss: LossConfig
    train: TrainConfig


# =============================================================================
# Helpers (strict)
# =============================================================================

def _req(d: Dict[str, Any], key: str, *, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {where}")
    return d[key]


def _as_path(p: Any, *, where: str) -> Path:
    if not isinstance(p, (str, Path)):
        raise TypeError(f"{where} must be a path string, got: {type(p)}")
    return Path(p)


def _as_bool(v: Any, *, where: str) -> bool:
    if isinstance(v, bool):
        return v
    raise TypeError(f"{where} must be bool, got: {type(v)}")


def _as_int(v: Any, *, where: str) -> int:
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return int(v)
    raise TypeError(f"{where} must be int, got: {type(v)}")


def _as_float(v: Any, *, where: str) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    raise TypeError(f"{where} must be float, got: {type(v)}")


def _as_str(v: Any, *, where: str) -> str:
    if isinstance(v, str):
        return v
    raise TypeError(f"{where} must be str, got: {type(v)}")


def _as_int_tuple(v: Any, *, where: str) -> Tuple[int, ...]:
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{where} must be list/tuple[int], got: {type(v)}")
    out = []
    for i, x in enumerate(v):
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"{where}[{i}] must be int, got: {type(x)}")
        out.append(int(x))
    return tuple(out)


def _as_dict(v: Any, *, where: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{where} must be a dict, got: {type(v)}")
    return v


def _check_no_unknown_keys(d: Dict[str, Any], allowed: Iterable[str], *, where: str) -> None:
    allowed_set = set(allowed)
    extra = [k for k in d.keys() if k not in allowed_set]
    if extra:
        raise KeyError(f"Unknown keys in {where}: {extra}. Allowed: {sorted(allowed_set)}")


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise TypeError(f"Top-level YAML must be a dict: {path}")
    return d


# =============================================================================
# Blocks parsing (YAML must write FULL blocks)
# =============================================================================

def _parse_thruster_lag(d: Dict[str, Any], *, where: str) -> ThrusterLagConfig:
    _check_no_unknown_keys(
        d,
        allowed=[
            "enabled",
            "normalize_input",
            "pwm_center",
            "pwm_half_range",
            "deadzone",
            "sat_gain",
            "learn_tau",
            "tau_init",
            "tau_min",
            "per_channel_tau",
            "dt",
        ],
        where=where,
    )
    return ThrusterLagConfig(
        enabled=_as_bool(_req(d, "enabled", where=where), where=f"{where}.enabled"),
        normalize_input=_as_bool(_req(d, "normalize_input", where=where), where=f"{where}.normalize_input"),
        pwm_center=_as_float(_req(d, "pwm_center", where=where), where=f"{where}.pwm_center"),
        pwm_half_range=_as_float(_req(d, "pwm_half_range", where=where), where=f"{where}.pwm_half_range"),
        deadzone=_as_float(_req(d, "deadzone", where=where), where=f"{where}.deadzone"),
        sat_gain=_as_float(_req(d, "sat_gain", where=where), where=f"{where}.sat_gain"),
        learn_tau=_as_bool(_req(d, "learn_tau", where=where), where=f"{where}.learn_tau"),
        tau_init=_as_float(_req(d, "tau_init", where=where), where=f"{where}.tau_init"),
        tau_min=_as_float(_req(d, "tau_min", where=where), where=f"{where}.tau_min"),
        per_channel_tau=_as_bool(_req(d, "per_channel_tau", where=where), where=f"{where}.per_channel_tau"),
        dt=_as_float(_req(d, "dt", where=where), where=f"{where}.dt"),
    )


def _parse_hydro_ssm(d: Dict[str, Any], *, where: str) -> HydroSSMConfig:
    _check_no_unknown_keys(
        d,
        allowed=[
            "enabled",
            "hidden_dim",
            "lambda_init",
            "act",
            "per_dim_lambda",
        ],
        where=where,
    )
    return HydroSSMConfig(
        enabled=_as_bool(_req(d, "enabled", where=where), where=f"{where}.enabled"),
        hidden_dim=_as_int(_req(d, "hidden_dim", where=where), where=f"{where}.hidden_dim"),
        lambda_init=_as_float(_req(d, "lambda_init", where=where), where=f"{where}.lambda_init"),
        act=_as_str(_req(d, "act", where=where), where=f"{where}.act"),
        per_dim_lambda=_as_bool(_req(d, "per_dim_lambda", where=where), where=f"{where}.per_dim_lambda"),
    )


def _parse_damping(d: Dict[str, Any], *, where: str) -> DampingConfig:
    _check_no_unknown_keys(
        d,
        allowed=[
            "enabled",
            "v_start",
            "v_dim",
            "mode",
            "d_init",
            "mlp_hidden",
        ],
        where=where,
    )
    return DampingConfig(
        enabled=_as_bool(_req(d, "enabled", where=where), where=f"{where}.enabled"),
        v_start=_as_int(_req(d, "v_start", where=where), where=f"{where}.v_start"),
        v_dim=_as_int(_req(d, "v_dim", where=where), where=f"{where}.v_dim"),
        mode=_as_str(_req(d, "mode", where=where), where=f"{where}.mode"),
        d_init=_as_float(_req(d, "d_init", where=where), where=f"{where}.d_init"),
        mlp_hidden=_as_int(_req(d, "mlp_hidden", where=where), where=f"{where}.mlp_hidden"),
    )


def _parse_uncertainty(d: Dict[str, Any], *, where: str) -> UncertaintyConfig:
    _check_no_unknown_keys(
        d,
        allowed=[
            "enabled",
            "feat_dim",
            "hidden",
            "logvar_min",
            "logvar_max",
        ],
        where=where,
    )
    return UncertaintyConfig(
        enabled=_as_bool(_req(d, "enabled", where=where), where=f"{where}.enabled"),
        feat_dim=_as_int(_req(d, "feat_dim", where=where), where=f"{where}.feat_dim"),
        hidden=_as_int(_req(d, "hidden", where=where), where=f"{where}.hidden"),
        logvar_min=_as_float(_req(d, "logvar_min", where=where), where=f"{where}.logvar_min"),
        logvar_max=_as_float(_req(d, "logvar_max", where=where), where=f"{where}.logvar_max"),
    )


def _parse_blocks(model_d: Dict[str, Any], *, where: str) -> BlocksConfig:
    """
    强制要求 YAML 写全 blocks（避免 silent bug）
    """
    blocks_d = _as_dict(_req(model_d, "blocks", where=where), where=f"{where}.blocks")
    _check_no_unknown_keys(
        blocks_d,
        allowed=["thruster_lag", "hydro_ssm", "damping", "uncertainty"],
        where=f"{where}.blocks",
    )

    tl_d = _as_dict(_req(blocks_d, "thruster_lag", where=f"{where}.blocks"), where=f"{where}.blocks.thruster_lag")
    hs_d = _as_dict(_req(blocks_d, "hydro_ssm", where=f"{where}.blocks"), where=f"{where}.blocks.hydro_ssm")
    dp_d = _as_dict(_req(blocks_d, "damping", where=f"{where}.blocks"), where=f"{where}.blocks.damping")
    uc_d = _as_dict(_req(blocks_d, "uncertainty", where=f"{where}.blocks"), where=f"{where}.blocks.uncertainty")

    return BlocksConfig(
        thruster_lag=_parse_thruster_lag(tl_d, where=f"{where}.blocks.thruster_lag"),
        hydro_ssm=_parse_hydro_ssm(hs_d, where=f"{where}.blocks.hydro_ssm"),
        damping=_parse_damping(dp_d, where=f"{where}.blocks.damping"),
        uncertainty=_parse_uncertainty(uc_d, where=f"{where}.blocks.uncertainty"),
    )


# =============================================================================
# Main builder
# =============================================================================

def build_from_dict(d: Dict[str, Any]) -> TrainYamlConfig:
    # ---------------- run ----------------
    run_d = _req(d, "run", where="root")
    if not isinstance(run_d, dict):
        raise TypeError("run must be a dict")

    _check_no_unknown_keys(
        run_d,
        allowed=["name", "seed", "device", "amp", "out_dir", "variant"],
        where="run",
    )

    run = RunConfig(
        name=str(_req(run_d, "name", where="run")),
        seed=int(run_d.get("seed", 0)),
        device=str(run_d.get("device", "cuda")),
        amp=bool(run_d.get("amp", False)),
        out_dir=_as_path(run_d.get("out_dir", "out/ckpts/s1_baseline"), where="run.out_dir"),
        variant=str(run_d.get("variant", "default")),
    )

    # ---------------- data ----------------
    data_d = _req(d, "data", where="root")
    if not isinstance(data_d, dict):
        raise TypeError("data must be a dict")

    split_d = data_d.get("split", {})
    if split_d is None:
        split_d = {}
    if not isinstance(split_d, dict):
        raise TypeError("data.split must be a dict")

    _check_no_unknown_keys(
        data_d,
        allowed=["data_dir", "batch_size", "num_workers", "pin_memory", "split"],
        where="data",
    )
    _check_no_unknown_keys(
        split_d,
        allowed=["train_ratio", "val_ratio"],
        where="data.split",
    )

    data = DataConfig(
        data_dir=_as_path(_req(data_d, "data_dir", where="data"), where="data.data_dir"),
        batch_size=int(data_d.get("batch_size", 256)),
        num_workers=int(data_d.get("num_workers", 4)),
        pin_memory=bool(data_d.get("pin_memory", True)),
        train_ratio=float(split_d.get("train_ratio", 0.7)),
        val_ratio=float(split_d.get("val_ratio", 0.15)),
        seed=run.seed,
    )

    # ---------------- model ----------------
    model_d = _req(d, "model", where="root")
    if not isinstance(model_d, dict):
        raise TypeError("model must be a dict")

    model_name = str(model_d.get("name", "s1_predictor"))
    if model_name != "s1_predictor":
        raise ValueError(f"Unsupported model.name={model_name!r} (v0 only supports 's1_predictor')")

    _check_no_unknown_keys(
        model_d,
        allowed=[
            "name",
            "din",
            "dout",
            "pred_len",
            "rnn_hidden",
            "rnn_layers",
            "dropout",
            "u_in_idx",
            "y_in_idx",
            "use_thruster_as_replacement",
            "use_hydro_feat",
            "blocks",
        ],
        where="model",
    )

    # 强制要求 blocks 写全
    blocks = _parse_blocks(model_d, where="model")

    model = S1PredictorConfig(
        din=int(model_d.get("din", 25)),
        dout=int(model_d.get("dout", 9)),
        pred_len=int(model_d.get("pred_len", 10)),
        rnn_hidden=int(model_d.get("rnn_hidden", 256)),
        rnn_layers=int(model_d.get("rnn_layers", 2)),
        dropout=float(model_d.get("dropout", 0.0)),
        u_in_idx=_as_int_tuple(_req(model_d, "u_in_idx", where="model"), where="model.u_in_idx"),
        y_in_idx=_as_int_tuple(_req(model_d, "y_in_idx", where="model"), where="model.y_in_idx"),
        use_thruster_as_replacement=_as_bool(
            model_d.get("use_thruster_as_replacement", True), where="model.use_thruster_as_replacement"
        ),
        use_hydro_feat=_as_bool(model_d.get("use_hydro_feat", True), where="model.use_hydro_feat"),
        blocks=blocks,
    )

    # ---------------- rollout ----------------
    rollout_d = d.get("rollout", {}) or {}
    if not isinstance(rollout_d, dict):
        raise TypeError("rollout must be a dict")

    _check_no_unknown_keys(rollout_d, allowed=["y0_source", "mode"], where="rollout")

    rollout = RolloutConfig(
        y0_source=str(rollout_d.get("y0_source", "x_last_state")),
        mode=str(rollout_d.get("mode", "delta_cumsum")),
    )
    if rollout.y0_source != "x_last_state":
        raise ValueError(f"Unsupported rollout.y0_source={rollout.y0_source!r} (v0 only supports 'x_last_state')")
    if rollout.mode != "delta_cumsum":
        raise ValueError(f"Unsupported rollout.mode={rollout.mode!r} (v0 only supports 'delta_cumsum')")

    # ---------------- loss ----------------
    loss_d = d.get("loss", {}) or {}
    if not isinstance(loss_d, dict):
        raise TypeError("loss must be a dict")

    _check_no_unknown_keys(loss_d, allowed=["type", "logvar_clip"], where="loss")

    loss_type = str(loss_d.get("type", "nll_diag"))
    if loss_type != "nll_diag":
        raise ValueError(f"Unsupported loss.type={loss_type!r} (v0 only supports 'nll_diag')")

    clip = loss_d.get("logvar_clip", [-10.0, 6.0])
    if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
        raise TypeError("loss.logvar_clip must be a list/tuple of [min,max]")

    loss = LossConfig(
        type=loss_type,
        logvar_clip_min=float(clip[0]),
        logvar_clip_max=float(clip[1]),
    )

    # ---------------- optim/train -> TrainConfig ----------------
    optim_d = d.get("optim", {}) or {}
    if not isinstance(optim_d, dict):
        raise TypeError("optim must be a dict")

    train_d = d.get("train", {}) or {}
    if not isinstance(train_d, dict):
        raise TypeError("train must be a dict")

    _check_no_unknown_keys(optim_d, allowed=["name", "lr", "weight_decay", "grad_clip"], where="optim")
    _check_no_unknown_keys(
        train_d,
        allowed=["epochs", "eval_every", "save_best", "save_last", "metric"],
        where="train",
    )

    optim_name = str(optim_d.get("name", "adamw")).lower()
    if optim_name != "adamw":
        raise ValueError(f"Unsupported optim.name={optim_name!r} (v0 only supports 'adamw')")

    train_cfg = TrainConfig(
        epochs=int(train_d.get("epochs", 30)),
        lr=float(optim_d.get("lr", 1e-3)),
        weight_decay=float(optim_d.get("weight_decay", 1e-4)),
        grad_clip=float(optim_d.get("grad_clip", 1.0)),
        device=run.device,
        amp=run.amp,
        out_dir=run.out_dir,
    )

    return TrainYamlConfig(
        run=run,
        data=data,
        model=model,
        rollout=rollout,
        loss=loss,
        train=train_cfg,
    )


def load_train_config(yaml_path: Path) -> TrainYamlConfig:
    d = load_yaml(yaml_path)
    return build_from_dict(d)
