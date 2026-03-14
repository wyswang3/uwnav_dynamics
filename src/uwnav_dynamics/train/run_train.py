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
best.pth / last.pth

依赖模块：
- uwnav_dynamics.train.config
- uwnav_dynamics.train.data_pipeline
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout
- uwnav_dynamics.models.utils.semantic_output_layout
- uwnav_dynamics.models.losses.auxiliary

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
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor
from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.models.utils.semantic_output_layout import canonical_semantic_output_layout
from uwnav_dynamics.models.losses.auxiliary import masked_huber_loss
from uwnav_dynamics.models.losses.nll import gaussian_nll_diag, gaussian_nll_diag_masked


def _copy_source_train_yaml(source_yaml: Path, run_dir: Path) -> Path:
    """把原始 train yaml 复制到 run 目录中，作为不可变配置快照。"""
    dst = run_dir / "source_train.yaml"
    shutil.copyfile(source_yaml, dst)
    return dst


def _write_train_history_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """写出 epoch 级训练历史，供后续画 loss/lr 曲线。"""
    fieldnames = ["epoch", "train_loss", "val_loss", "lr", "evaluated", "is_best"]
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
        "schema_version": "train_summary_v1",
        "best_val": float(fit_result["best_val"]),
        "best_epoch": int(fit_result["best_epoch"]),
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


def build_loss_fn(
    logvar_clip_min: float,
    logvar_clip_max: float,
    y_in_idx,
    *,
    dout: int = 9,
    dvl_obs_weight: float = 0.0,
    dvl_obs_delta: float = 1.0,
):
    """
    P0.1 复合损失：
      1) 主 state 路径保持原逻辑：
         model -> dY/logvar -> rollout_from_delta -> dense/masked NLL
      2) 若启用 dvl_obs 辅助头：
         只在 velocity semantic group 上计算 masked Huber auxiliary loss
    """
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

        if target_mask is not None:
            state_loss = gaussian_nll_diag_masked(y_hat, Y, logvar, target_mask)
        else:
            state_loss = gaussian_nll_diag(y_hat, Y, logvar)

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
        dvl_obs_weight=cfg.loss.dvl_obs_weight,
        dvl_obs_delta=cfg.loss.dvl_obs_delta,
    )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
