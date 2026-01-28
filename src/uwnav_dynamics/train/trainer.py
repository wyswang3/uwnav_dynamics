from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, Tuple, Union
import torch
from torch.utils.data import DataLoader

from uwnav_dynamics.models.losses.metrics import masked_regression_metrics

# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    device: str = "cuda"
    amp: bool = True
    out_dir: Path = Path("out/ckpts/s1_baseline")

    # checkpoint policy
    save_best: bool = True
    save_last: bool = True
    metric: str = "val_loss"  # 预留: 未来支持 val_rmse 等


# =============================================================================
# Batch utilities (dict batch for async supervision)
# =============================================================================

BatchDict = Mapping[str, Any]
BatchType = Union[
    Tuple[torch.Tensor, torch.Tensor],
    BatchDict,
]
def _metrics_from_aux_or_compute(
    aux: Dict[str, Any],
    *,
    fallback_yhat: Optional[torch.Tensor] = None,
    fallback_ytrue: Optional[torch.Tensor] = None,
    fallback_mask: Optional[torch.Tensor] = None,
) -> tuple[float, float, float, float, bool]:
    """
    Returns: (rmse, mae, valid_ratio, valid_count, skipped)
    Priority:
      1) aux['rmse'], aux['mae'], aux['valid_ratio'], aux['valid_count'], aux['skipped']
      2) if aux provides y_hat/y_true/y_mask -> compute masked_regression_metrics
      3) fallback passed in -> compute
      4) else NaN
    """
    skipped = bool(aux.get("skipped", False))

    rmse = _pick_float(aux, "rmse")
    mae = _pick_float(aux, "mae")
    valid_ratio = _pick_float(aux, "valid_ratio")
    valid_count = _pick_float(aux, "valid_count")

    if rmse is not None and mae is not None:
        # 如果 aux 已给 rmse/mae，就直接用（valid_* 若缺失则后面再补）
        if valid_ratio is None:
            valid_ratio = float("nan")
        if valid_count is None:
            valid_count = float("nan")
        return rmse, mae, valid_ratio, valid_count, skipped

    # 尝试从 aux 的张量里算
    y_hat = aux.get("y_hat", None)
    y_true = aux.get("y_true", None)
    y_mask = aux.get("y_mask", None)

    if (y_hat is None) and (fallback_yhat is not None):
        y_hat = fallback_yhat
    if (y_true is None) and (fallback_ytrue is not None):
        y_true = fallback_ytrue
    if (y_mask is None) and (fallback_mask is not None):
        y_mask = fallback_mask

    if torch.is_tensor(y_hat) and torch.is_tensor(y_true) and torch.is_tensor(y_mask) and (not skipped):
        met = masked_regression_metrics(y_hat, y_true, y_mask)
        rmse = float(met["rmse"])
        mae = float(met["mae"])

        # valid_count/ratio：尽量补齐
        if valid_count is None:
            valid_count = float(y_mask.to(torch.float32).sum().item())
        if valid_ratio is None:
            total = float(y_mask.numel())
            valid_ratio = float(valid_count / max(total, 1.0))
        return rmse, mae, valid_ratio, valid_count, skipped

    # 什么都没有 -> NaN
    return float("nan"), float("nan"), float("nan"), float("nan"), skipped

def _to_device_batch(batch: BatchDict, device: torch.device) -> Dict[str, Any]:
    nb = (device.type == "cuda")
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=nb)
        else:
            out[k] = v
    return out


def _require_keys(batch: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise KeyError(f"Batch missing keys: {missing}. Got keys={list(batch.keys())}")

def _mask_stats(m: torch.Tensor) -> Dict[str, float]:
    m_f = m.to(dtype=torch.float32)
    valid = float(m_f.sum().item())
    total = float(m_f.numel())
    ratio = valid / max(total, 1.0)
    return {"valid": valid, "total": total, "ratio": ratio}

def _pick_float(aux: Dict[str, Any], key: str) -> Optional[float]:
    v = aux.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None
# =============================================================================
# Loss wrapper (supports loss_fn returning loss or (loss, aux))
# =============================================================================

def _call_loss(loss_fn, model, batch: Mapping[str, Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    loss_fn is expected to return:
      - loss (Tensor)
      - aux (dict)  [optional; if not provided, we'll create empty aux]
    """
    out = loss_fn(model, batch)
    if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]) and isinstance(out[1], dict):
        return out[0], out[1]
    if torch.is_tensor(out):
        return out, {}
    raise TypeError(f"loss_fn must return loss or (loss, aux_dict). Got: {type(out)}")

# =============================================================================
# Train / Eval (returns metrics dict)
# =============================================================================
# =============================================================================
# Train / Eval (returns metrics dict)
# =============================================================================

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    amp_enabled: bool,
    *,
    log_mask_every: int = 0,
) -> Dict[str, float]:
    model.train()

    sum_loss_w = 0.0
    sum_loss = 0.0

    sum_rmse_w = 0.0
    sum_rmse = 0.0
    sum_mae = 0.0

    sum_validratio_w = 0.0
    sum_validratio = 0.0

    sum_skipped_w = 0.0
    sum_skipped = 0.0

    for step, batch in enumerate(loader, start=1):
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Async supervision requires dict batch. "
                "Your Dataset should return {'X','Y','XM','YM'}."
            )

        batch = _to_device_batch(batch, device)
        _require_keys(batch, ("X", "Y", "YM"))

        X = batch["X"]
        YM = batch["YM"]
        bs = int(X.shape[0])

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and amp_enabled and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                loss, aux = _call_loss(loss_fn, model, batch)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, aux = _call_loss(loss_fn, model, batch)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # -------- loss aggregation (bs-weighted) --------
        lv = float(loss.detach().item())
        sum_loss += lv * bs
        sum_loss_w += bs

        # -------- valid_ratio (prefer aux; fallback to YM stats) --------
        st = _mask_stats(YM)
        vr = _pick_float(aux, "valid_ratio")
        if vr is None or not (vr == vr):  # NaN check
            vr = float(st["ratio"])
        sum_validratio += vr * bs
        sum_validratio_w += bs

        # -------- rmse/mae + skipped (prefer aux; else compute if provided tensors) --------
        with torch.no_grad():
            rmse, mae, _vr2, vcnt, skipped = _metrics_from_aux_or_compute(
                aux,
                fallback_mask=YM,
            )

        # 指标加权：优先按 valid_count（更符合“监督元素数”），否则按 bs
        w = float(vcnt) if (vcnt == vcnt and vcnt > 0) else float(bs)

        if rmse == rmse:  # not NaN
            sum_rmse += rmse * w
            sum_mae += mae * w
            sum_rmse_w += w

        sum_skipped += (1.0 if skipped else 0.0) * bs
        sum_skipped_w += bs

        if log_mask_every and (step % log_mask_every == 0):
            print(f"[MASK][TRAIN] step={step} YM_ratio={st['ratio']:.4f} (valid={st['valid']:.0f}/{st['total']:.0f})")

    loss_avg = sum_loss / max(sum_loss_w, 1.0)
    valid_avg = sum_validratio / max(sum_validratio_w, 1.0)
    skipped_ratio = sum_skipped / max(sum_skipped_w, 1.0)

    rmse_avg = sum_rmse / sum_rmse_w if sum_rmse_w > 0 else float("nan")
    mae_avg = sum_mae / sum_rmse_w if sum_rmse_w > 0 else float("nan")

    return {
        "loss": float(loss_avg),
        "rmse": float(rmse_avg),
        "mae": float(mae_avg),
        "valid_ratio": float(valid_avg),
        "skipped_ratio": float(skipped_ratio),
    }


@torch.no_grad()
def eval_one_epoch(
    model,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    *,
    log_mask_every: int = 0,
) -> Dict[str, float]:
    model.eval()

    sum_loss_w = 0.0
    sum_loss = 0.0

    sum_rmse_w = 0.0
    sum_rmse = 0.0
    sum_mae = 0.0

    sum_validratio_w = 0.0
    sum_validratio = 0.0

    sum_skipped_w = 0.0
    sum_skipped = 0.0

    for step, batch in enumerate(loader, start=1):
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Async supervision requires dict batch. "
                "Your Dataset should return {'X','Y','XM','YM'}."
            )

        batch = _to_device_batch(batch, device)
        _require_keys(batch, ("X", "Y", "YM"))

        X = batch["X"]
        YM = batch["YM"]
        bs = int(X.shape[0])

        loss, aux = _call_loss(loss_fn, model, batch)

        # -------- loss aggregation (bs-weighted) --------
        lv = float(loss.detach().item())
        sum_loss += lv * bs
        sum_loss_w += bs

        # -------- valid_ratio --------
        st = _mask_stats(YM)
        vr = _pick_float(aux, "valid_ratio")
        if vr is None or not (vr == vr):
            vr = float(st["ratio"])
        sum_validratio += vr * bs
        sum_validratio_w += bs

        # -------- rmse/mae --------
        rmse, mae, _vr2, vcnt, skipped = _metrics_from_aux_or_compute(
            aux,
            fallback_mask=YM,
        )
        w = float(vcnt) if (vcnt == vcnt and vcnt > 0) else float(bs)

        if rmse == rmse:
            sum_rmse += rmse * w
            sum_mae += mae * w
            sum_rmse_w += w

        sum_skipped += (1.0 if skipped else 0.0) * bs
        sum_skipped_w += bs

        if log_mask_every and (step % log_mask_every == 0):
            print(f"[MASK][VAL ] step={step} YM_ratio={st['ratio']:.4f} (valid={st['valid']:.0f}/{st['total']:.0f})")

    loss_avg = sum_loss / max(sum_loss_w, 1.0)
    valid_avg = sum_validratio / max(sum_validratio_w, 1.0)
    skipped_ratio = sum_skipped / max(sum_skipped_w, 1.0)

    rmse_avg = sum_rmse / sum_rmse_w if sum_rmse_w > 0 else float("nan")
    mae_avg = sum_mae / sum_rmse_w if sum_rmse_w > 0 else float("nan")

    return {
        "loss": float(loss_avg),
        "rmse": float(rmse_avg),
        "mae": float(mae_avg),
        "valid_ratio": float(valid_avg),
        "skipped_ratio": float(skipped_ratio),
    }

# =============================================================================
# Resolve helpers
# =============================================================================

def _resolve_device(device: Optional[torch.device], cfg_device: str) -> torch.device:
    if device is not None:
        return device
    dev = str(cfg_device).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available -> fallback to CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _resolve_out_dir(run_dir: Optional[Path], cfg_out_dir: Path) -> Path:
    return Path(run_dir) if run_dir is not None else Path(cfg_out_dir)

# =============================================================================
# Fit
# =============================================================================
def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    loss_fn,
    *,
    device: Optional[torch.device] = None,
    run_dir: Optional[Path] = None,
    amp: Optional[bool] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    log_mask_every: int = 0,
) -> Dict[str, Any]:
    """
    Trainer (async supervision ready).

    train_one_epoch/eval_one_epoch must return metrics dict, at least:
      {"loss": float, "valid_ratio": float, ...}

    Save policy:
      - last.pth: every epoch (if save_last)
      - best.pth: best by cfg.metric (default "val_loss"; fallback to "val.loss")
    """
    out_dir = _resolve_out_dir(run_dir, cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, cfg.device)

    # AMP decision: external > cfg, only enable on CUDA
    amp_enabled = bool(cfg.amp) if amp is None else bool(amp)
    if device.type != "cuda":
        amp_enabled = False

    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if amp_enabled and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    # metric selection
    # cfg.metric examples:
    #   "val_loss" (default)
    #   "val_rmse" / "val_mae"
    #   "val.loss" / "val.rmse"
    metric_key = getattr(cfg, "metric", "val_loss") or "val_loss"

    def _get_score(train_m: Dict[str, float], val_m: Dict[str, float]) -> float:
        mk = str(metric_key).strip()

        # normalize aliases
        mk = mk.replace(".", "_")
        if mk.startswith("val_"):
            k = mk[len("val_") :]
            return float(val_m.get(k, float("inf")))
        if mk.startswith("train_"):
            k = mk[len("train_") :]
            return float(train_m.get(k, float("inf")))

        # fallback: treat as val.<mk>
        return float(val_m.get(mk, val_m.get("loss", float("inf"))))

    def _fmt_metrics(tag: str, m: Dict[str, float]) -> str:
        # keep logs stable even if rmse/mae are NaN
        loss = float(m.get("loss", float("nan")))
        vr = float(m.get("valid_ratio", float("nan")))
        rmse = float(m.get("rmse", float("nan")))
        mae = float(m.get("mae", float("nan")))
        sk = float(m.get("skipped_ratio", float("nan")))
        return (
            f"{tag}_loss={loss:.6f} "
            f"{tag}_rmse={rmse:.6f} "
            f"{tag}_mae={mae:.6f} "
            f"{tag}_valid={vr:.3f} "
            f"{tag}_skipped={sk:.3f}"
        )

    best_score = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for ep in range(1, int(cfg.epochs) + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            scaler=scaler,
            grad_clip=float(cfg.grad_clip),
            amp_enabled=amp_enabled,
            log_mask_every=int(log_mask_every),
        )
        va = eval_one_epoch(
            model,
            val_loader,
            loss_fn,
            device=device,
            log_mask_every=int(log_mask_every),
        )

        score = _get_score(tr, va)

        print(
            f"[EPOCH {ep:03d}] "
            + _fmt_metrics("train", tr)
            + "  "
            + _fmt_metrics("val", va)
            + f"  score({metric_key})={score:.6f}"
        )

        payload = {
            "epoch": ep,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "train": tr,
            "val": va,
            "score": float(score),
            "score_name": str(metric_key),
            "device": str(device),
            "amp": bool(amp_enabled),
        }

        # optional: keep a tiny history for debugging/plotting later
        history.append({"epoch": ep, "train": tr, "val": va, "score": float(score)})

        if getattr(cfg, "save_last", True):
            torch.save(payload, last_path)

        if getattr(cfg, "save_best", True) and float(score) < float(best_score):
            best_score = float(score)
            best_epoch = int(ep)
            torch.save(payload, best_path)
            print(f"[CKPT] best -> {best_path} ({metric_key}={best_score:.6f} @epoch={best_epoch})")

    print(f"[DONE] best_score({metric_key})={best_score:.6f} @epoch={best_epoch}  ckpt={best_path}")

    return {
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_path": str(best_path),
        "last_path": str(last_path),
        "metric": str(metric_key),
        "device": str(device),
        "amp": bool(amp_enabled),
        "out_dir": str(out_dir),
        # "history": history,  # 如果你不想返回太大，可以删掉这一项
    }