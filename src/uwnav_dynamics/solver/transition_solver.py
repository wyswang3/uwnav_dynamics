"""
模块名称：经验型状态转移求解器

模块职责：
把训练后的 `S1Predictor` 封装为可递推的经验型状态求解器，
为单步状态预测、长序列 autoregressive replay 与后续控制接口提供统一入口。

主要功能：
1. 从 `train yaml + ckpt + run-scoped split/scaler artifact` 加载训练好的模型。
2. 提供 `predict_next_state()`，把一段历史窗映射成下一时刻主状态预测。
3. 提供 `rollout_with_feature_templates()`，在给定未来控制/上下文模板时做 autoregressive 递推。
4. 对 `pred_len>1` 的旧模型提供保守兼容：默认只消费第一步预测作为单步求解结果。

数据流：
train yaml + ckpt + x/y scaler
    ↓
history window (physical feature space)
    ↓
model forward in z-space
    ↓
next-state prediction (physical state space)
    ↓
autoregressive replay / future controller integration

依赖模块：
- numpy
- torch
- uwnav_dynamics.cli.utils
- uwnav_dynamics.dataset.normalize
- uwnav_dynamics.experiment.layout
- uwnav_dynamics.models.nets.s1_predictor
- uwnav_dynamics.models.utils.execution_layout
- uwnav_dynamics.models.utils.rollout
- uwnav_dynamics.train.config

备注：
- 当前求解器只负责递推主状态维，即 `cfg.model.y_in_idx` 对应的状态槽位。
- 若模型 `pred_len != 1`，本模块会保守退化为“使用 block rollout 的第一步作为 step() 输出”。
- 更严格的一步状态求解器语义仍建议优先使用 `quality_step_v1` 训练分支。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from uwnav_dynamics.cli.utils import pick_ckpt
from uwnav_dynamics.dataset.normalize import inverse_transform, load_scaler, transform
from uwnav_dynamics.experiment.layout import RunLayout, run_layout_from_train_yaml
from uwnav_dynamics.models.nets.s1_predictor import S1Predictor, S1PredictorConfig
from uwnav_dynamics.models.utils.execution_layout import extract_y0_from_x_last
from uwnav_dynamics.models.utils.rollout import rollout_from_delta
from uwnav_dynamics.train.config import TrainYamlConfig, load_train_config


@dataclass(frozen=True)
class TransitionSolverLoadResult:
    """训练后求解器加载结果。"""
    cfg_train: TrainYamlConfig
    run_layout: RunLayout
    ckpt_path: Path
    solver: "TrainedTransitionSolver"


@dataclass(frozen=True)
class TransitionSolverRolloutResult:
    """单次 autoregressive replay 的递推结果。"""
    predicted_states: np.ndarray
    replay_rows: np.ndarray
    nonfinite_trigger_count: int


def _sanitize_scaled_inputs(x_scaled: np.ndarray) -> np.ndarray:
    """
    与 eval 主流程保持一致：对经过 scaler 后仍残留的非有限值做最小清洗。
    """
    bad = ~np.isfinite(x_scaled)
    if int(np.count_nonzero(bad)) <= 0:
        return np.asarray(x_scaled, dtype=np.float32, copy=False)
    x_safe = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    return np.asarray(x_safe, dtype=np.float32, copy=False)


class TrainedTransitionSolver:
    """
    将训练后的 `S1Predictor` 封装为经验型状态求解器。

    当前接口假设调用方提供：
    - `history_window_physical`：形状 `(hist_len, Din)` 的物理量纲输入历史窗
    - `future_feature_templates_physical`：未来每一步的输入模板行

    其中输入模板应包含：
    - 已知的控制输入与上下文特征
    - 由求解器递推填回的主状态槽位（`y_in_idx`）
    """

    def __init__(
        self,
        *,
        model: S1Predictor,
        cfg_model: S1PredictorConfig,
        x_scaler: dict[str, Any],
        y_scaler: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg_model = cfg_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = torch.device(device)
        self.y_in_idx = tuple(int(v) for v in cfg_model.y_in_idx)
        self.hist_len: int | None = None
        self.step_semantics = "single_step" if int(cfg_model.pred_len) == 1 else "first_step_of_block_rollout"

    def predict_next_state(self, history_window_physical: np.ndarray) -> np.ndarray:
        """
        基于当前历史窗预测下一时刻主状态。

        Parameters
        ----------
        history_window_physical:
            `(L, Din)`，物理量纲输入历史窗。
        """
        history = np.asarray(history_window_physical, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != int(self.cfg_model.din):
            raise ValueError(
                "history_window_physical must be (L, Din), got "
                f"{tuple(history.shape)} with Din={self.cfg_model.din}"
            )
        if history.shape[0] <= 0:
            raise ValueError("history_window_physical must contain at least one step")

        self.hist_len = int(history.shape[0])
        x_scaled = transform(history[None, :, :], self.x_scaler).astype(np.float32, copy=False)
        x_scaled = _sanitize_scaled_inputs(x_scaled)
        xb = torch.from_numpy(x_scaled).to(device=self.device, dtype=torch.float32)

        with torch.inference_mode():
            dY, _logvar = self.model(xb)
            y0 = extract_y0_from_x_last(xb, self.cfg_model.y_in_idx)
            y_hat = rollout_from_delta(y0, dY)

        next_state_z = y_hat[:, 0, :].detach().cpu().numpy()
        next_state_phys = inverse_transform(next_state_z, self.y_scaler).astype(np.float32, copy=False)
        next_state = np.asarray(next_state_phys[0], dtype=np.float32)
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError("predicted next state contains non-finite values")
        return next_state

    def rollout_with_feature_templates(
        self,
        *,
        initial_history_physical: np.ndarray,
        future_feature_templates_physical: np.ndarray,
    ) -> TransitionSolverRolloutResult:
        """
        在给定未来输入模板时执行 autoregressive 递推。

        `future_feature_templates_physical[t]` 必须已经包含该时刻已知的控制输入和上下文，
        本方法只会把主状态槽位 `y_in_idx` 替换为求解器预测的状态。
        """
        history = np.asarray(initial_history_physical, dtype=np.float32)
        templates = np.asarray(future_feature_templates_physical, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != int(self.cfg_model.din):
            raise ValueError(
                "initial_history_physical must be (L, Din), got "
                f"{tuple(history.shape)} with Din={self.cfg_model.din}"
            )
        if templates.ndim != 2 or templates.shape[1] != int(self.cfg_model.din):
            raise ValueError(
                "future_feature_templates_physical must be (T, Din), got "
                f"{tuple(templates.shape)} with Din={self.cfg_model.din}"
            )
        if history.shape[0] <= 0:
            raise ValueError("initial_history_physical must contain at least one step")

        hist = np.array(history, copy=True, dtype=np.float32)
        pred_rows: list[np.ndarray] = []
        pred_states: list[np.ndarray] = []
        nonfinite_trigger_count = 0

        for template in templates:
            next_state = self.predict_next_state(hist)
            next_row = np.array(template, copy=True, dtype=np.float32)
            next_row[list(self.y_in_idx)] = next_state

            if not np.all(np.isfinite(next_row)):
                nonfinite_trigger_count += 1
                raise FloatingPointError("autoregressive replay produced non-finite feature row")

            pred_rows.append(next_row)
            pred_states.append(next_state)
            hist = np.concatenate([hist[1:], next_row[None, :]], axis=0)

        if pred_states:
            pred_state_arr = np.stack(pred_states, axis=0).astype(np.float32, copy=False)
            pred_row_arr = np.stack(pred_rows, axis=0).astype(np.float32, copy=False)
        else:
            pred_state_arr = np.zeros((0, int(self.cfg_model.dout)), dtype=np.float32)
            pred_row_arr = np.zeros((0, int(self.cfg_model.din)), dtype=np.float32)

        return TransitionSolverRolloutResult(
            predicted_states=pred_state_arr,
            replay_rows=pred_row_arr,
            nonfinite_trigger_count=int(nonfinite_trigger_count),
        )


def load_trained_transition_solver(
    *,
    train_yaml: str | Path,
    ckpt: str | Path | None = None,
    device: str | None = None,
) -> TransitionSolverLoadResult:
    """
    从训练配置和 checkpoint 加载经验型状态求解器。
    """
    train_yaml_path = Path(train_yaml)
    cfg_train = load_train_config(train_yaml_path)
    run_layout = run_layout_from_train_yaml(train_yaml_path)
    ckpt_path = pick_ckpt(Path(ckpt)) if ckpt is not None else pick_ckpt(run_layout.run_dir)

    runtime_device = torch.device(device if device is not None else str(cfg_train.run.device))
    if runtime_device.type.startswith("cuda") and not torch.cuda.is_available():
        runtime_device = torch.device("cpu")

    model = S1Predictor(cfg_train.model).to(runtime_device)
    ckpt_obj = torch.load(ckpt_path, map_location=runtime_device)
    state_dict = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x_scaler = load_scaler(run_layout.x_scaler_path)
    y_scaler = load_scaler(run_layout.y_scaler_path)

    solver = TrainedTransitionSolver(
        model=model,
        cfg_model=cfg_train.model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=runtime_device,
    )
    return TransitionSolverLoadResult(
        cfg_train=cfg_train,
        run_layout=run_layout,
        ckpt_path=ckpt_path,
        solver=solver,
    )
