"""
模块名称：训练配置 canonical parser

模块职责：
作为 `configs/train/*.yaml` 的唯一解析真源，
将实验定义严格转换为强类型 dataclass 配置，并为 train / eval 共享。

主要功能：
1. 解析 `run / data / model / rollout / loss / train` 配置段。
2. 解析并校验 P0 研发阶段的最小辅助头配置，确保旧 YAML 继续兼容。
3. 解析 early stopping、learning-rate scheduler 与训练期 monitor 配置，统一进入训练强类型对象。
4. 对模型 blocks、索引布局与 runtime schema 执行严格校验。
5. 保证 train / eval 使用同一份模型结构解释结果，避免配置漂移。

数据流：
train yaml
    ↓
strict schema parsing
    ↓
TrainYamlConfig
    ↓
train.run_train / eval.config

依赖模块：
- yaml
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.models.utils.execution_layout

备注：
- 本模块只做配置解析与静态校验，不承担 rollout 数值执行。
- `cfg_model.y_in_idx` 是 execution layout contract 的唯一执行真源。
- P0.1 的 `model.aux_heads` 目前只在 parser 中收口并校验，
  具体接入模型前向由后续 patch 负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from uwnav_dynamics.models.utils.execution_layout import validate_execution_layout, validate_feature_indices


# =============================================================================
# YAML -> dataclasses (Style A, strict schema)
# =============================================================================

@dataclass(frozen=True)
class LossConfig:
    """训练损失配置，包含主损失类型、状态转移复合项与辅助监督权重。"""
    type: str = "nll_diag"
    logvar_clip_min: float = -10.0
    logvar_clip_max: float = 6.0
    state_huber_weight: float = 0.0
    state_huber_delta: float = 1.0
    delta_huber_weight: float = 0.0
    delta_huber_delta: float = 1.0
    logvar_reg_weight: float = 0.0
    tail_weight_power: float = 0.0
    acc_weight: float = 1.0
    gyro_weight: float = 1.0
    vel_weight: float = 1.0
    # P0.1 保守起点建议 0.1 或 0.2；默认 0.0 表示完全关闭 auxiliary DVL loss。
    dvl_obs_weight: float = 0.0
    dvl_obs_delta: float = 1.0


@dataclass(frozen=True)
class AuxHeadConfig:
    """单个辅助预测头的开关与隐藏维配置。"""
    enabled: bool = False
    hidden: int = 128


@dataclass(frozen=True)
class ModelAuxHeadsConfig:
    """模型全部辅助预测头的聚合配置。"""
    dvl_obs: AuxHeadConfig = field(default_factory=AuxHeadConfig)


@dataclass(frozen=True)
class RolloutConfig:
    """rollout 执行契约配置。"""
    y0_source: str = "x_last_state"   # "x_last_state" only for v0
    mode: str = "delta_cumsum"        # "delta_cumsum" only for v0


@dataclass(frozen=True)
class RunConfig:
    """单次训练运行的设备、目录和变体命名配置。"""
    name: str
    seed: int = 0
    device: str = "cuda"
    amp: bool = False
    out_dir: Path = Path("out/ckpts/s1_baseline")
    variant: str = "default"   # +++ 新增


@dataclass(frozen=True)
class TrainYamlConfig:
    """训练 YAML 的顶层强类型配置对象。"""
    run: RunConfig
    data: DataConfig
    model: S1PredictorConfig
    rollout: RolloutConfig
    loss: LossConfig
    train: TrainConfig
    model_aux_heads: ModelAuxHeadsConfig = field(default_factory=ModelAuxHeadsConfig)


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


def _warn_if_out_dir_repeats_variant(out_dir: Path, variant: str) -> None:
    """
    轻量校验 run.out_dir / run.variant 的拼接契约。

    这里暂时只给 warning，不直接抛错，原因是：
      - 历史实验可能已经把 variant 手工写进 out_dir；
      - PR1 的目标是先把契约写清楚并修正样例，避免一次性打断旧工作流。
    后续如果仓库内配置都已收敛，再考虑升级成 hard error。
    """
    out_dir_name = Path(out_dir).name
    if out_dir_name == str(variant):
        print(
            "[WARN] run.out_dir already ends with run.variant; "
            "the canonical contract is run_dir = out_dir / variant, so writing "
            "variant into out_dir will create duplicated nesting."
        )


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取并校验训练 YAML 的顶层字典结构。"""
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


def _parse_aux_head(d: Dict[str, Any], *, where: str) -> AuxHeadConfig:
    _check_no_unknown_keys(
        d,
        allowed=["enabled", "hidden"],
        where=where,
    )
    hidden = _as_int(d.get("hidden", 128), where=f"{where}.hidden")
    if hidden <= 0:
        raise ValueError(f"{where}.hidden must be > 0, got {hidden}")
    return AuxHeadConfig(
        enabled=_as_bool(d.get("enabled", False), where=f"{where}.enabled"),
        hidden=hidden,
    )


def _parse_model_aux_heads(model_d: Dict[str, Any], *, where: str) -> ModelAuxHeadsConfig:
    aux_d = model_d.get("aux_heads", {}) or {}
    if not isinstance(aux_d, dict):
        raise TypeError(f"{where}.aux_heads must be a dict")
    _check_no_unknown_keys(
        aux_d,
        allowed=["dvl_obs"],
        where=f"{where}.aux_heads",
    )
    dvl_d = aux_d.get("dvl_obs", {}) or {}
    if not isinstance(dvl_d, dict):
        raise TypeError(f"{where}.aux_heads.dvl_obs must be a dict")
    return ModelAuxHeadsConfig(
        dvl_obs=_parse_aux_head(dvl_d, where=f"{where}.aux_heads.dvl_obs"),
    )


# =============================================================================
# Main builder
# =============================================================================

def build_from_dict(d: Dict[str, Any]) -> TrainYamlConfig:
    """
    Parse a raw yaml mapping into the fully-typed training config.

    `TrainYamlConfig` is the canonical config contract of the repository.
    Runtime overrides may replace selected fields later, but downstream code
    should not re-interpret the yaml independently.
    """
    # `run` 段只描述实验运行环境与落盘路径，不承载模型或数据语义。
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
    _warn_if_out_dir_repeats_variant(run.out_dir, run.variant)

    # `data` 段只允许出现训练时真正消费的数据路径和 split 比例。
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

    # `model` 段是最严格的 schema：一旦字段未知，就立即 fail-fast，
    # 避免 train / eval 对同一份 yaml 做出不同解释。
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
            "head_mode",
            "group_head_hidden",
            "aux_heads",
            "blocks",
        ],
        where="model",
    )

    # blocks 必须显式写全，防止“默认值悄悄生效”导致实验不可审计。
    blocks = _parse_blocks(model_d, where="model")
    model_aux_heads = _parse_model_aux_heads(model_d, where="model")

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
        head_mode=_as_str(model_d.get("head_mode", "joint"), where="model.head_mode"),
        group_head_hidden=_as_int(model_d.get("group_head_hidden", 128), where="model.group_head_hidden"),
        blocks=blocks,
    )
    validate_feature_indices(model.u_in_idx, upper_bound=model.din, name="model.u_in_idx")
    validate_execution_layout(model.y_in_idx, din=model.din, dout=model.dout)
    if model.head_mode not in {"joint", "grouped"}:
        raise ValueError(f"Unsupported model.head_mode={model.head_mode!r} (expect 'joint' or 'grouped')")
    if model.group_head_hidden <= 0:
        raise ValueError(f"model.group_head_hidden must be > 0, got {model.group_head_hidden}")

    # rollout 契约目前只支持一条执行路径；
    # parser 在这里提前收口，后面的 train / eval 就不再分叉解释。
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

    # loss 段同时约束主损失和辅助头权重，避免“开了权重但没开 head”的静默配置错误。
    loss_d = d.get("loss", {}) or {}
    if not isinstance(loss_d, dict):
        raise TypeError("loss must be a dict")

    _check_no_unknown_keys(
        loss_d,
        allowed=[
            "type",
            "logvar_clip",
            "state_huber_weight",
            "state_huber_delta",
            "delta_huber_weight",
            "delta_huber_delta",
            "logvar_reg_weight",
            "tail_weight_power",
            "acc_weight",
            "gyro_weight",
            "vel_weight",
            "dvl_obs_weight",
            "dvl_obs_delta",
        ],
        where="loss",
    )

    loss_type = str(loss_d.get("type", "nll_diag"))
    if loss_type not in {"nll_diag", "transition_balance"}:
        raise ValueError(
            f"Unsupported loss.type={loss_type!r} "
            "(current parser supports 'nll_diag' or 'transition_balance')"
        )

    clip = loss_d.get("logvar_clip", [-10.0, 6.0])
    if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
        raise TypeError("loss.logvar_clip must be a list/tuple of [min,max]")

    state_huber_weight = _as_float(loss_d.get("state_huber_weight", 0.0), where="loss.state_huber_weight")
    state_huber_delta = _as_float(loss_d.get("state_huber_delta", 1.0), where="loss.state_huber_delta")
    delta_huber_weight = _as_float(loss_d.get("delta_huber_weight", 0.0), where="loss.delta_huber_weight")
    delta_huber_delta = _as_float(loss_d.get("delta_huber_delta", 1.0), where="loss.delta_huber_delta")
    logvar_reg_weight = _as_float(loss_d.get("logvar_reg_weight", 0.0), where="loss.logvar_reg_weight")
    tail_weight_power = _as_float(loss_d.get("tail_weight_power", 0.0), where="loss.tail_weight_power")
    acc_weight = _as_float(loss_d.get("acc_weight", 1.0), where="loss.acc_weight")
    gyro_weight = _as_float(loss_d.get("gyro_weight", 1.0), where="loss.gyro_weight")
    vel_weight = _as_float(loss_d.get("vel_weight", 1.0), where="loss.vel_weight")
    dvl_obs_weight = _as_float(loss_d.get("dvl_obs_weight", 0.0), where="loss.dvl_obs_weight")
    dvl_obs_delta = _as_float(loss_d.get("dvl_obs_delta", 1.0), where="loss.dvl_obs_delta")
    if state_huber_weight < 0.0:
        raise ValueError(f"loss.state_huber_weight must be >= 0, got {state_huber_weight}")
    if state_huber_delta <= 0.0:
        raise ValueError(f"loss.state_huber_delta must be > 0, got {state_huber_delta}")
    if delta_huber_weight < 0.0:
        raise ValueError(f"loss.delta_huber_weight must be >= 0, got {delta_huber_weight}")
    if delta_huber_delta <= 0.0:
        raise ValueError(f"loss.delta_huber_delta must be > 0, got {delta_huber_delta}")
    if logvar_reg_weight < 0.0:
        raise ValueError(f"loss.logvar_reg_weight must be >= 0, got {logvar_reg_weight}")
    if tail_weight_power < 0.0:
        raise ValueError(f"loss.tail_weight_power must be >= 0, got {tail_weight_power}")
    if acc_weight <= 0.0 or gyro_weight <= 0.0 or vel_weight <= 0.0:
        raise ValueError(
            "loss.acc_weight / loss.gyro_weight / loss.vel_weight must all be > 0 "
            f"(got {acc_weight}, {gyro_weight}, {vel_weight})"
        )
    if dvl_obs_weight < 0.0:
        raise ValueError(f"loss.dvl_obs_weight must be >= 0, got {dvl_obs_weight}")
    if dvl_obs_delta <= 0.0:
        raise ValueError(f"loss.dvl_obs_delta must be > 0, got {dvl_obs_delta}")
    if (not model_aux_heads.dvl_obs.enabled) and dvl_obs_weight > 0.0:
        raise ValueError(
            "loss.dvl_obs_weight > 0 requires model.aux_heads.dvl_obs.enabled=true "
            "to avoid silently enabling an unused auxiliary objective"
        )
    if loss_type == "nll_diag":
        if (
            state_huber_weight != 0.0
            or delta_huber_weight != 0.0
            or logvar_reg_weight != 0.0
            or tail_weight_power != 0.0
            or acc_weight != 1.0
            or gyro_weight != 1.0
            or vel_weight != 1.0
        ):
            raise ValueError(
                "loss.type='nll_diag' must keep transition_balance fields at defaults; "
                "set loss.type='transition_balance' for grouped/tail/delta reweighting"
            )
    else:
        if (
            state_huber_weight == 0.0
            and delta_huber_weight == 0.0
            and logvar_reg_weight == 0.0
            and tail_weight_power == 0.0
            and acc_weight == 1.0
            and gyro_weight == 1.0
            and vel_weight == 1.0
        ):
            raise ValueError(
                "loss.type='transition_balance' requires at least one non-default transition-balancing setting"
            )

    loss = LossConfig(
        type=loss_type,
        logvar_clip_min=float(clip[0]),
        logvar_clip_max=float(clip[1]),
        state_huber_weight=state_huber_weight,
        state_huber_delta=state_huber_delta,
        delta_huber_weight=delta_huber_weight,
        delta_huber_delta=delta_huber_delta,
        logvar_reg_weight=logvar_reg_weight,
        tail_weight_power=tail_weight_power,
        acc_weight=acc_weight,
        gyro_weight=gyro_weight,
        vel_weight=vel_weight,
        dvl_obs_weight=dvl_obs_weight,
        dvl_obs_delta=dvl_obs_delta,
    )

    # `optim` 与 `train` 最终收敛成 `TrainConfig`，
    # 运行时只消费这一份强类型对象，不再回看原始 yaml。
    optim_d = d.get("optim", {}) or {}
    if not isinstance(optim_d, dict):
        raise TypeError("optim must be a dict")

    train_d = d.get("train", {}) or {}
    if not isinstance(train_d, dict):
        raise TypeError("train must be a dict")

    scheduler_d = optim_d.get("scheduler", {}) or {}
    if not isinstance(scheduler_d, dict):
        raise TypeError("optim.scheduler must be a dict")

    early_stopping_d = train_d.get("early_stopping", {}) or {}
    if not isinstance(early_stopping_d, dict):
        raise TypeError("train.early_stopping must be a dict")

    _check_no_unknown_keys(optim_d, allowed=["name", "lr", "weight_decay", "grad_clip", "scheduler"], where="optim")
    _check_no_unknown_keys(
        scheduler_d,
        allowed=["name", "factor", "patience", "min_lr"],
        where="optim.scheduler",
    )
    _check_no_unknown_keys(
        train_d,
        allowed=["epochs", "eval_every", "save_best", "save_last", "metric", "early_stopping"],
        where="train",
    )
    _check_no_unknown_keys(
        early_stopping_d,
        allowed=["patience", "min_delta"],
        where="train.early_stopping",
    )

    optim_name = str(optim_d.get("name", "adamw")).lower()
    if optim_name != "adamw":
        raise ValueError(f"Unsupported optim.name={optim_name!r} (v0 only supports 'adamw')")
    scheduler_name = str(scheduler_d.get("name", "none")).lower()
    if scheduler_name not in {"none", "reduce_on_plateau"}:
        raise ValueError(
            f"Unsupported optim.scheduler.name={scheduler_name!r} "
            "(v0 only supports 'none' or 'reduce_on_plateau')"
        )

    train_cfg = TrainConfig(
        epochs=int(train_d.get("epochs", 30)),
        eval_every=int(train_d.get("eval_every", 1)),
        lr=float(optim_d.get("lr", 1e-3)),
        weight_decay=float(optim_d.get("weight_decay", 1e-4)),
        grad_clip=float(optim_d.get("grad_clip", 1.0)),
        device=run.device,
        amp=run.amp,
        out_dir=run.out_dir,
        save_best=bool(train_d.get("save_best", True)),
        save_last=bool(train_d.get("save_last", True)),
        metric=str(train_d.get("metric", "val_loss")),
        scheduler_name=scheduler_name,
        scheduler_factor=float(scheduler_d.get("factor", 0.5)),
        scheduler_patience=int(scheduler_d.get("patience", 5)),
        scheduler_min_lr=float(scheduler_d.get("min_lr", 1e-6)),
        early_stopping_patience=int(early_stopping_d.get("patience", 0)),
        early_stopping_min_delta=float(early_stopping_d.get("min_delta", 0.0)),
    )

    if train_cfg.eval_every <= 0:
        raise ValueError(f"train.eval_every must be > 0, got {train_cfg.eval_every}")
    if train_cfg.metric not in {"val_loss", "val_transition_score"}:
        raise ValueError(
            "train.metric must be one of {'val_loss', 'val_transition_score'}, "
            f"got {train_cfg.metric!r}"
        )
    if train_cfg.scheduler_factor <= 0.0 or train_cfg.scheduler_factor >= 1.0:
        raise ValueError(
            f"optim.scheduler.factor must be in (0,1), got {train_cfg.scheduler_factor}"
        )
    if train_cfg.scheduler_patience < 0:
        raise ValueError(f"optim.scheduler.patience must be >= 0, got {train_cfg.scheduler_patience}")
    if train_cfg.scheduler_min_lr < 0.0:
        raise ValueError(f"optim.scheduler.min_lr must be >= 0, got {train_cfg.scheduler_min_lr}")
    if train_cfg.early_stopping_patience < 0:
        raise ValueError(
            f"train.early_stopping.patience must be >= 0, got {train_cfg.early_stopping_patience}"
        )
    if train_cfg.early_stopping_min_delta < 0.0:
        raise ValueError(
            f"train.early_stopping.min_delta must be >= 0, got {train_cfg.early_stopping_min_delta}"
        )

    return TrainYamlConfig(
        run=run,
        data=data,
        model=model,
        model_aux_heads=model_aux_heads,
        rollout=rollout,
        loss=loss,
        train=train_cfg,
    )


def load_train_config(yaml_path: Path) -> TrainYamlConfig:
    """
    Load the canonical training config from yaml.

    Eval code should reuse this function instead of maintaining a second,
    partial model builder. That keeps train/eval topology strictly aligned.
    """
    d = load_yaml(yaml_path)
    return build_from_dict(d)
