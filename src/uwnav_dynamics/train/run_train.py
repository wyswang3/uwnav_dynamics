"""
模块名称：训练主入口

模块职责：
负责加载训练配置、准备 run-scoped 数据 artifact、
构建模型与损失函数，并调用 trainer 执行训练。

主要功能：
1. 复用 canonical train parser 与 runtime override 生成最终训练配置。
2. 准备 split / scaler / DataLoader，并写出 `resolved_train.yaml`。
3. 基于 execution layout contract 构建 rollout loss，保证 train / eval 对 `y0` 解释一致。
4. 在 PR5 中支持 batch `target_mask` 驱动的 mask-aware supervision。
5. 在 P0.1 中以最小方式接入 `dvl_obs` 辅助监督，不改变主 state head 路径。
6. 在当前阶段支持面向状态转移的 composite loss，用于更强地约束 `acc / gyro / vel`。
7. 支持训练期 `val_transition_score` monitor，用更偏长期 rollout 的信号选择 best ckpt。
8. 在训练期并行记录 z-space `RMSE / MAE`，补充通用误差基线。
9. 训练结束后自动从 `train_history.csv` 生成训练曲线与 dashboard 图。

数据流：
train yaml + CLI override
    ↓
TrainYamlConfig / resolved_train.yaml
    ↓
prepare_train_data()
    ↓
S1Predictor + rollout loss
    ↓
fit()
    ↓
best.pth / last.pth / train_summary.yaml / train_history.csv / train_plots/*

依赖模块：
- uwnav_dynamics.train.config
- uwnav_dynamics.train.data_pipeline
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout
- uwnav_dynamics.models.utils.semantic_output_layout
- uwnav_dynamics.models.losses.auxiliary
- uwnav_dynamics.models.losses.state_transition
- uwnav_dynamics.viz.train.plot_training_history

备注：
- 本模块只接入 execution layout contract，不负责语义分组解释。
- `target_mask` 若存在，则是训练运行时唯一的监督有效性真源。
- `dvl_obs` 辅助监督只读取 semantic velocity group，不改变主 state head 的训练语义。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

from dataclasses import replace
from typing import Callable
import torch
import yaml

from uwnav_dynamics.experiment.layout import RunLayout
from uwnav_dynamics.experiment.paths import relative_path_str
from uwnav_dynamics.train.config import load_train_config
from uwnav_dynamics.train.data_pipeline import prepare_train_data
from uwnav_dynamics.train.runtime import (
    TrainCliOverrides,
    apply_train_overrides,
    reconcile_data_config_for_device,
    resolve_runtime_device,
    save_resolved_train_config,
    set_global_seed,
)
from uwnav_dynamics.train.trainer import fit
from uwnav_dynamics.viz.train.plot_training_history import plot_training_artifacts
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout
from uwnav_dynamics.models.losses.auxiliary import masked_huber_loss
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag, gaussian_nll_diag_elements, gaussian_nll_diag_masked
from uwnav_dynamics.models.losses.state_transition import (
    build_delta_targets,
    build_group_weight_vector,
    build_horizon_weight_vector,
    masked_weighted_mse_loss,
    masked_weighted_huber_loss,
    positive_logvar_penalty,
    reduce_weighted_mean,
    resolve_late_horizon_start,
)


def _copy_source_train_yaml(source_yaml: Path, run_dir: Path) -> Path:
    """把原始 train yaml 复制到 run 目录中，作为不可变配置快照。"""
    dst = run_dir / "source_train.yaml"
    shutil.copyfile(source_yaml, dst)
    return dst


def _write_train_history_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """写出 epoch 级训练历史，供后续画 loss/lr 曲线。"""
    preferred = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_rmse_global_zspace",
        "val_mae_global_zspace",
        "monitor_name",
        "monitor_value",
        "lr",
        "evaluated",
        "is_best",
    ]
    fieldnames: list[str] = []
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_train_summary_yaml(
    path: Path,
    *,
    fit_result: dict[str, object],
    prepared,
    run_dir: Path,
    source_snapshot_path: Path,
    path_root: Path,
) -> Path:
    """把训练阶段关键摘要统一写成审计友好的 yaml。"""
    payload = {
        "schema_version": "train_summary_v3",
        "best_val": float(fit_result["best_val"]),
        "best_epoch": int(fit_result["best_epoch"]),
        "monitor_name": str(fit_result.get("monitor_name", "val_loss")),
        "best_monitor": float(fit_result.get("best_monitor", fit_result["best_val"])),
        "best_monitor_epoch": int(fit_result.get("best_monitor_epoch", fit_result["best_epoch"])),
        "selected_val_loss": float(fit_result.get("selected_val_loss", fit_result["best_val"])),
        "best_val_loss": float(fit_result.get("best_val_loss", fit_result["best_val"])),
        "best_val_loss_epoch": int(fit_result.get("best_val_loss_epoch", fit_result["best_epoch"])),
        "selected_val_rmse_global_zspace": fit_result.get("selected_val_rmse_global_zspace"),
        "selected_val_mae_global_zspace": fit_result.get("selected_val_mae_global_zspace"),
        "best_val_rmse_global_zspace": fit_result.get("best_val_rmse_global_zspace"),
        "best_val_mae_global_zspace": fit_result.get("best_val_mae_global_zspace"),
        "epochs_ran": int(fit_result["epochs_ran"]),
        "stopped_early": bool(fit_result["stopped_early"]),
        "final_lr": float(fit_result["final_lr"]),
        "scheduler_name": str(fit_result["scheduler_name"]),
        "eval_every": int(fit_result["eval_every"]),
        "train_wall_time_sec": float(fit_result["train_wall_time_sec"]),
        "split_strategy": str(prepared.split_strategy),
        "split_sizes": {str(k): int(v) for k, v in prepared.split_sizes.items()},
        "dropped_window_count": int(prepared.dropped_window_count),
        "artifacts": {
            "run_dir": relative_path_str(run_dir, base_dir=path_root),
            "source_train_yaml": relative_path_str(source_snapshot_path, base_dir=path_root),
            "resolved_train_yaml": relative_path_str(run_dir / "resolved_train.yaml", base_dir=path_root),
            "train_history_csv": relative_path_str(run_dir / "train_history.csv", base_dir=path_root),
            "train_plots_dir": relative_path_str(run_dir / "train_plots", base_dir=path_root),
            "training_dashboard_png": relative_path_str(run_dir / "train_plots" / "training_dashboard.png", base_dir=path_root),
            "training_loss_png": relative_path_str(run_dir / "train_plots" / "training_loss_curve.png", base_dir=path_root),
            "training_monitor_png": relative_path_str(run_dir / "train_plots" / "validation_monitor_curve.png", base_dir=path_root),
            "training_lr_png": relative_path_str(run_dir / "train_plots" / "learning_rate_curve.png", base_dir=path_root),
            "best_ckpt": relative_path_str(Path(str(fit_result["best_path"])), base_dir=path_root),
            "last_ckpt": relative_path_str(Path(str(fit_result["last_path"])), base_dir=path_root),
            "split_indices": relative_path_str(run_dir / "split_indices.npz", base_dir=path_root),
            "x_scaler": relative_path_str(run_dir / "scalers" / "x_scaler.npz", base_dir=path_root),
            "y_scaler": relative_path_str(run_dir / "scalers" / "y_scaler.npz", base_dir=path_root),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path


def _global_rmse_mae_torch(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在训练期 z-space 上计算全局 RMSE / MAE。"""
    err = y_hat - y_true
    if target_mask is None:
        mse = torch.mean(err * err)
        mae = torch.mean(torch.abs(err))
        return torch.sqrt(mse), mae

    if target_mask.shape != y_hat.shape:
        raise ValueError(f"target_mask shape mismatch: expect {tuple(y_hat.shape)}, got {tuple(target_mask.shape)}")
    mask = target_mask.to(device=y_hat.device, dtype=y_hat.dtype)
    valid = mask.sum()
    if float(valid.item()) <= 0.0:
        raise ValueError("target_mask contains zero valid supervision elements during training monitor")
    mse = ((err * err) * mask).sum() / valid
    mae = (torch.abs(err) * mask).sum() / valid
    return torch.sqrt(mse), mae


def build_monitor_fn(
    metric_name: str,
    y_in_idx,
    *,
    dout: int = 9,
    state_huber_delta: float = 1.0,
    delta_huber_delta: float = 1.0,
    tail_weight_power: float = 0.0,
    late_horizon_weight: float = 0.0,
    late_horizon_fraction: float = 0.4,
    acc_weight: float = 1.0,
    gyro_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> Callable | None:
    """
    构造训练期验证监控指标。

    当前支持：
      - `val_loss`
        直接复用训练损失，保持历史行为。
      - `val_transition_score`
        仅基于状态误差与转移误差的监控分数，不读取 `logvar`，
        用于避免模型通过放大不确定度掩盖长期 rollout 误差。

    `val_transition_score` 的组成：
      1. 带语义组加权与尾部加权的 rollout state Huber
      2. 最后一步 state Huber（额外强调 horizon 尾部）
      3. 带尾部加权的 delta Huber
    """
    metric_key = str(metric_name or "val_loss").lower()
    if metric_key in {"", "val_loss"}:
        return None
    if metric_key != "val_transition_score":
        raise ValueError(f"Unsupported train.metric={metric_name!r}")

    def _monitor(model, X, Y, target_mask=None):
        if hasattr(model, "forward_with_aux"):
            dY, _logvar, _aux = model.forward_with_aux(X)
        else:
            dY, _logvar = model(X)
        y0 = extract_y0_from_x_last(X, y_in_idx)
        y_hat = rollout_from_delta(y0, dY)
        component_weight = build_group_weight_vector(
            dout,
            acc_weight=float(acc_weight),
            gyro_weight=float(gyro_weight),
            vel_weight=float(vel_weight),
            device=Y.device,
            dtype=Y.dtype,
        )
        horizon_weight = build_horizon_weight_vector(
            Y.shape[1],
            tail_weight_power=max(float(tail_weight_power), 1.0),
            device=Y.device,
            dtype=Y.dtype,
        )
        state_tail = masked_weighted_huber_loss(
            y_hat,
            Y,
            target_mask=target_mask,
            component_weight=component_weight,
            horizon_weight=horizon_weight,
            delta=float(state_huber_delta),
        )
        final_mask = target_mask[:, -1:, :] if target_mask is not None else None
        final_step = masked_weighted_huber_loss(
            y_hat[:, -1:, :],
            Y[:, -1:, :],
            target_mask=final_mask,
            component_weight=component_weight,
            horizon_weight=None,
            delta=float(state_huber_delta),
        )
        dY_true = build_delta_targets(y0, Y)
        delta_tail = masked_weighted_huber_loss(
            dY,
            dY_true,
            target_mask=target_mask,
            component_weight=component_weight,
            horizon_weight=horizon_weight,
            delta=float(delta_huber_delta),
        )
        total = state_tail + 0.75 * final_step + 0.25 * delta_tail
        if float(late_horizon_weight) > 0.0:
            late_start = resolve_late_horizon_start(
                Y.shape[1],
                late_horizon_fraction=float(late_horizon_fraction),
            )
            late_mask = target_mask[:, late_start:, :] if target_mask is not None else None
            late_state = masked_weighted_huber_loss(
                y_hat[:, late_start:, :],
                Y[:, late_start:, :],
                target_mask=late_mask,
                component_weight=component_weight,
                horizon_weight=None,
                delta=float(state_huber_delta),
            )
            total = total + float(late_horizon_weight) * late_state
        return total

    return _monitor


def build_extra_val_metrics_fn(y_in_idx) -> Callable:
    """
    构造训练期附加验证指标。

    当前固定输出：
      - `val_rmse_global_zspace`
      - `val_mae_global_zspace`
    """

    def _metrics(model, X, Y, target_mask=None):
        if hasattr(model, "forward_with_aux"):
            dY, _logvar, _aux = model.forward_with_aux(X)
        else:
            dY, _logvar = model(X)
        y0 = extract_y0_from_x_last(X, y_in_idx)
        y_hat = rollout_from_delta(y0, dY)
        rmse, mae = _global_rmse_mae_torch(y_hat, Y, target_mask)
        return {
            "val_rmse_global_zspace": rmse,
            "val_mae_global_zspace": mae,
        }

    return _metrics


def build_loss_fn(
    logvar_clip_min: float,
    logvar_clip_max: float,
    y_in_idx,
    *,
    dout: int = 9,
    loss_type: str = "nll_diag",
    state_mse_weight: float = 0.0,
    state_huber_weight: float = 0.0,
    state_huber_delta: float = 1.0,
    state_final_weight: float = 0.0,
    late_horizon_weight: float = 0.0,
    late_horizon_fraction: float = 0.4,
    delta_huber_weight: float = 0.0,
    delta_huber_delta: float = 1.0,
    logvar_reg_weight: float = 0.0,
    tail_weight_power: float = 0.0,
    acc_weight: float = 1.0,
    gyro_weight: float = 1.0,
    vel_weight: float = 1.0,
    dvl_obs_weight: float = 0.0,
    dvl_obs_delta: float = 1.0,
):
    """
    状态转移训练损失：
      1) `nll_diag`：保持旧路径不变
         model -> dY/logvar -> rollout_from_delta -> dense/masked NLL
      2) `transition_balance`：在 NLL 之外加入
         - 语义组加权
         - horizon 尾部加权
         - rollout state MSE
         - rollout state Huber
         - delta transition Huber
         - 正向 logvar 正则
      3) 若启用 dvl_obs 辅助头：
         只在 velocity semantic group 上计算 masked Huber auxiliary loss
    """
    if str(loss_type) not in {"nll_diag", "transition_balance"}:
        raise ValueError(f"Unsupported loss_type={loss_type!r}")
    vel_idx_cpu = torch.as_tensor(
        list(canonical_semantic_output_layout(dout).group_indices["vel"]),
        dtype=torch.long,
    )

    def _loss(model, X, Y, target_mask=None):
        if hasattr(model, "forward_with_aux"):
            dY, logvar, aux = model.forward_with_aux(X)
        else:
            dY, logvar = model(X)
            aux = {"dvl_obs": None}

        y0 = extract_y0_from_x_last(X, y_in_idx)
        y_hat = rollout_from_delta(y0, dY)
        logvar = torch.clamp(logvar, min=logvar_clip_min, max=logvar_clip_max)

        if str(loss_type) == "nll_diag":
            if target_mask is not None:
                state_loss = gaussian_nll_diag_masked(y_hat, Y, logvar, target_mask)
            else:
                state_loss = gaussian_nll_diag(y_hat, Y, logvar)
        else:
            component_weight = build_group_weight_vector(
                dout,
                acc_weight=float(acc_weight),
                gyro_weight=float(gyro_weight),
                vel_weight=float(vel_weight),
                device=Y.device,
                dtype=Y.dtype,
            )
            horizon_weight = build_horizon_weight_vector(
                Y.shape[1],
                tail_weight_power=float(tail_weight_power),
                device=Y.device,
                dtype=Y.dtype,
            )
            total = reduce_weighted_mean(
                gaussian_nll_diag_elements(y_hat, Y, logvar),
                target_mask=target_mask,
                component_weight=component_weight,
                horizon_weight=horizon_weight,
            )
            if float(state_mse_weight) > 0.0:
                total = total + float(state_mse_weight) * masked_weighted_mse_loss(
                    y_hat,
                    Y,
                    target_mask=target_mask,
                    component_weight=component_weight,
                    horizon_weight=horizon_weight,
                )
            if float(state_huber_weight) > 0.0:
                total = total + float(state_huber_weight) * masked_weighted_huber_loss(
                    y_hat,
                    Y,
                    target_mask=target_mask,
                    component_weight=component_weight,
                    horizon_weight=horizon_weight,
                    delta=float(state_huber_delta),
                )
            if float(state_final_weight) > 0.0:
                final_mask = target_mask[:, -1:, :] if target_mask is not None else None
                total = total + float(state_final_weight) * masked_weighted_huber_loss(
                    y_hat[:, -1:, :],
                    Y[:, -1:, :],
                    target_mask=final_mask,
                    component_weight=component_weight,
                    horizon_weight=None,
                    delta=float(state_huber_delta),
                )
            if float(late_horizon_weight) > 0.0:
                late_start = resolve_late_horizon_start(
                    Y.shape[1],
                    late_horizon_fraction=float(late_horizon_fraction),
                )
                late_mask = target_mask[:, late_start:, :] if target_mask is not None else None
                total = total + float(late_horizon_weight) * masked_weighted_huber_loss(
                    y_hat[:, late_start:, :],
                    Y[:, late_start:, :],
                    target_mask=late_mask,
                    component_weight=component_weight,
                    horizon_weight=None,
                    delta=float(state_huber_delta),
                )
            if float(delta_huber_weight) > 0.0:
                dY_true = build_delta_targets(y0, Y)
                total = total + float(delta_huber_weight) * masked_weighted_huber_loss(
                    dY,
                    dY_true,
                    target_mask=target_mask,
                    component_weight=component_weight,
                    horizon_weight=horizon_weight,
                    delta=float(delta_huber_delta),
                )
            if float(logvar_reg_weight) > 0.0:
                total = total + float(logvar_reg_weight) * positive_logvar_penalty(
                    logvar,
                    target_mask=target_mask,
                    component_weight=component_weight,
                    horizon_weight=horizon_weight,
                )
            state_loss = total

        if float(dvl_obs_weight) <= 0.0:
            return state_loss

        dvl_enabled = bool(getattr(getattr(model.cfg, "aux_heads", None), "dvl_obs", None) and model.cfg.aux_heads.dvl_obs.enabled)
        if not dvl_enabled:
            raise ValueError("loss.dvl_obs_weight > 0 but model.cfg.aux_heads.dvl_obs.enabled is False")

        if "dvl_obs" not in aux:
            raise ValueError("model.forward_with_aux() must return stable aux key 'dvl_obs'")
        dvl_obs = aux["dvl_obs"]
        if dvl_obs is None:
            raise ValueError("model.cfg.aux_heads.dvl_obs.enabled is True but aux['dvl_obs'] is None")
        if target_mask is None:
            raise ValueError("dvl auxiliary loss requires runtime target_mask; got None")

        vel_idx = vel_idx_cpu.to(device=Y.device)
        y_vel = Y.index_select(dim=-1, index=vel_idx)
        mask_vel = target_mask.index_select(dim=-1, index=vel_idx)
        dvl_aux_loss = masked_huber_loss(
            dvl_obs,
            y_vel,
            mask_vel,
            delta=dvl_obs_delta,
        )
        return state_loss + float(dvl_obs_weight) * dvl_aux_loss

    return _loss


def main() -> int:
    """命令行训练入口，解析配置、执行训练并落盘产物。"""
    ap = argparse.ArgumentParser("uwnav_dynamics.train.run_train")
    ap.add_argument("-y", "--yaml", type=str, required=True, help="configs/train/*.yaml")

    # ---- CLI override：让 pipeline 生效 ----
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)          # cpu/cuda/cuda:0
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--pin_memory", type=str, default=None, choices=["true", "false"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    args = ap.parse_args()

    cfg = load_train_config(Path(args.yaml))

    if args.amp and args.no_amp:
        raise ValueError("Cannot set both --amp and --no_amp")

    amp_override = None
    if args.amp:
        amp_override = True
    if args.no_amp:
        amp_override = False

    # 配置处理顺序固定为：
    #   1) 原始 train yaml（canonical parser）
    #   2) CLI override（纯函数替换，不改原对象）
    #   3) runtime reconcile（例如 CPU 下关闭 pin_memory）
    #   4) resolved_train.yaml（记录最终实际执行配置）
    cli_overrides = TrainCliOverrides(
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory.lower() == "true") if args.pin_memory is not None else None,
        out_dir=args.out_dir,
        variant=args.variant,
        seed=args.seed,
        amp=amp_override,
    )

    cfg = apply_train_overrides(
        cfg,
        cli_overrides,
    )

    # ------------------------------
    # Resolve run_dir = out_dir/variant
    # ------------------------------
    run_layout = RunLayout(out_dir=Path(cfg.run.out_dir), variant=str(getattr(cfg.run, "variant", "default")))
    variant = run_layout.variant
    run_dir = run_layout.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] name={cfg.run.name}")
    print(f"[RUN] variant={variant}")
    print(f"[RUN] run_dir={run_dir}")
    print(f"[RUN] data_dir={cfg.data.data_dir}")
    source_snapshot_path = _copy_source_train_yaml(Path(args.yaml), run_dir)

    # ------------------------------
    # Seed / device
    # ------------------------------
    set_global_seed(cfg.run.seed)
    device = resolve_runtime_device(cfg.run.device)
    data_cfg, pin_memory_note = reconcile_data_config_for_device(cfg.data, device)
    if pin_memory_note is not None:
        print(pin_memory_note)
        cfg = replace(cfg, data=data_cfg)

    # ------------------------------
    # Data / split / scaler (no leakage)
    # ------------------------------
    prepared = prepare_train_data(cfg.data, run_layout)
    print(f"[RUN] resolved_config={run_dir / 'resolved_train.yaml'}")
    save_resolved_train_config(
        run_dir / "resolved_train.yaml",
        cfg,
        source_yaml=args.yaml,
        cli_overrides=cli_overrides,
        requested_device=str(cfg.run.device),
        runtime_device=str(device),
        run_dir=run_dir,
        split_indices_path=run_layout.split_indices_path,
        split_strategy=prepared.split_strategy,
        x_scaler_path=run_layout.x_scaler_path,
        y_scaler_path=run_layout.y_scaler_path,
        source_snapshot_path=source_snapshot_path,
        split_sizes=prepared.split_sizes,
        dropped_window_count=prepared.dropped_window_count,
        path_root=Path.cwd(),
    )

    model = S1Predictor(cfg)

    loss_fn = build_loss_fn(
        cfg.loss.logvar_clip_min,
        cfg.loss.logvar_clip_max,
        cfg.model.y_in_idx,
        dout=cfg.model.dout,
        loss_type=cfg.loss.type,
        state_mse_weight=cfg.loss.state_mse_weight,
        state_huber_weight=cfg.loss.state_huber_weight,
        state_huber_delta=cfg.loss.state_huber_delta,
        state_final_weight=cfg.loss.state_final_weight,
        late_horizon_weight=cfg.loss.late_horizon_weight,
        late_horizon_fraction=cfg.loss.late_horizon_fraction,
        delta_huber_weight=cfg.loss.delta_huber_weight,
        delta_huber_delta=cfg.loss.delta_huber_delta,
        logvar_reg_weight=cfg.loss.logvar_reg_weight,
        tail_weight_power=cfg.loss.tail_weight_power,
        acc_weight=cfg.loss.acc_weight,
        gyro_weight=cfg.loss.gyro_weight,
        vel_weight=cfg.loss.vel_weight,
        dvl_obs_weight=cfg.loss.dvl_obs_weight,
        dvl_obs_delta=cfg.loss.dvl_obs_delta,
    )
    monitor_fn = build_monitor_fn(
        cfg.train.metric,
        cfg.model.y_in_idx,
        dout=cfg.model.dout,
        state_huber_delta=cfg.loss.state_huber_delta,
        delta_huber_delta=cfg.loss.delta_huber_delta,
        tail_weight_power=cfg.loss.tail_weight_power,
        late_horizon_weight=cfg.loss.late_horizon_weight,
        late_horizon_fraction=cfg.loss.late_horizon_fraction,
        acc_weight=cfg.loss.acc_weight,
        gyro_weight=cfg.loss.gyro_weight,
        vel_weight=cfg.loss.vel_weight,
    )
    extra_val_metrics_fn = build_extra_val_metrics_fn(cfg.model.y_in_idx)

    # ------------------------------
    # Fit (pipeline-style): pass device/run_dir/amp
    # ------------------------------
    fit_result = fit(
        model,
        prepared.train_loader,
        prepared.val_loader,
        cfg.train,
        loss_fn,
        device=device,
        run_dir=run_dir,
        amp=bool(cfg.run.amp),
        monitor_fn=monitor_fn,
        monitor_name=cfg.train.metric,
        extra_val_metrics_fn=extra_val_metrics_fn,
    )
    _write_train_history_csv(run_dir / "train_history.csv", fit_result["history"])
    _write_train_summary_yaml(
        run_dir / "train_summary.yaml",
        fit_result=fit_result,
        prepared=prepared,
        run_dir=run_dir,
        source_snapshot_path=source_snapshot_path,
        path_root=Path.cwd(),
    )
    plot_training_artifacts(
        history_csv=run_dir / "train_history.csv",
        out_dir=run_dir / "train_plots",
        summary_yaml=run_dir / "train_summary.yaml",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
