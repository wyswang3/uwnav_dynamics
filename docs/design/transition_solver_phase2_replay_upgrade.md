# 状态求解器升级 Phase 2：一步求解与长序列 Replay

更新时间：2026-04-20

## 1. 阶段目标

把当前仓库从“固定 horizon 的轨迹块预测器”进一步推进到：

**可递推的经验型系统状态求解器**

当前阶段不追求：

- 闭环控制最终证明
- 高保真物理仿真真值
- 完整 MPC / RL 环境接入

当前阶段只追求两件事：

1. 训练后模型能够提供稳定的一步状态转移接口。
2. 能够基于现有实验数据做长序列 autoregressive replay，验证长期递推误差是否可控。

## 2. 当前升级路线

推荐按以下顺序推进：

### Phase 2.1 已落地：一步状态求解器接口

最小实现：

- 新增 `src/uwnav_dynamics/solver/transition_solver.py`
- 将训练后的 `S1Predictor + scaler` 封装成 `TrainedTransitionSolver`
- 提供：
  - `predict_next_state(history_window)`
  - `step_with_feature_template(history_window, feature_template)`
  - `rollout_with_feature_templates(initial_history, future_templates)`

当前语义：

- 若模型 `pred_len=1`，直接作为一步状态求解器
- 若模型 `pred_len>1`，保守退化为“取 block rollout 的第一步作为 step 输出”
- `step_with_feature_template()` 是 controller / RL wrapper 的当前最小边界：
  调用方给出下一时刻已知控制量与上下文模板，求解器只填回主状态槽位。

### Phase 2.2 已落地：长序列 autoregressive replay 评估

最小实现：

- 新增 `src/uwnav_dynamics/solver/replay.py`
- 新增 `src/uwnav_dynamics/cli/transition_replay.py`

验证思路：

1. 从 `processed dataset` 的 `meta.yaml + base_csv + idx0` 恢复连续时间轴
2. 从 split 对应的窗口起点构造连续 segment
3. 保留未来控制/上下文模板
4. 只递推主状态槽位 `y_in_idx`
5. 统计长序列 replay 的：
   - `rmse_global / mae_global`
   - `final_step`
   - `rollout_growth`
   - `tail_error`
   - `worst_abs_bias`
   - `segment_count / total_steps / nonfinite_trigger_count`

### Phase 2.3 已落地：多方案统一评估与排行

当前已补充：

- `src/uwnav_dynamics/solver/reporting.py`
- `src/uwnav_dynamics/cli/transition_replay_matrix.py`
- `configs/launch/replay_matrix_example.yaml`

当前统一产物约定：

1. `summary.csv`
2. `ranking.csv`
3. `manifest.yaml`
4. `runs/<candidate>/metrics.yaml`

当前推荐排行协议：

1. 先要求 `segment_count > 0`
2. 要求 `nonfinite_trigger_count == 0`
3. 再按以下主指标做加权排行：
   - `rmse_global`
   - `mae_global`
   - `final_step.rmse_global_mean`
   - `rollout_growth.rmse_last_over_first_p95`
   - `tail_error.abs_p95_global`
   - `tail_error.abs_p99_global`
   - `bias.worst_abs_bias`

建议下一步就按这套协议对：

1. `quality_v3` 与 `quality_step_v1`
2. 不同 blocks 组合
3. 多组实验数据集

做统一 replay 对照，而不是只看单次 run 的局部结果。

### Phase 2.4 后续阶段：最小 controller replay / simulator loop

在 replay 结果稳定之前，不要直接扩成完整控制系统。

后续最小方向应是：

- `solver.step(u_t, context_t) -> x_{t+1}`
- 进程内 replay loop
- 记录推理延迟、循环周期与运行频率

## 3. 当前推荐命令

### 3.1 一步状态转移训练

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

### 3.2 长序列 replay 验证

```bash
python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_seconds 50 \
  --max_seconds_per_segment 50 \
  --dt 0.01
```

说明：100 Hz 数据中 `50 steps` 只有 `0.5s`。
若目标是 50 秒长时长自由递推验证，必须使用 `--min_seconds 50`
或 replay matrix 配置中的 `min_seconds: 50`。

默认产物目录：

```text
run.out_dir/run.variant/replay_test/
```

核心产物：

- `metrics.yaml`
- `segment_metrics.csv`
- `component_metrics.csv`
- `step_metrics.csv`
- `pred_samples.npz`
- `pred_context.npz`
- `resolved_replay.yaml`

路径契约：

- `metrics.yaml["cfg"]` 与 `resolved_replay.yaml["cfg"]` 中的路径必须以相对路径快照保存；
- 运行时可以使用绝对路径加载模型或数据，但落盘 artifact 不应绑定某台机器的绝对目录。

### 3.3 replay matrix 统一排行

```bash
python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/replay_matrix_example.yaml
```

核心产物：

- `summary.csv`
- `ranking.csv`
- `manifest.yaml`
- `runs/<candidate>/metrics.yaml`

## 4. 验证标准

当前阶段至少应满足：

1. 一步状态求解器可从训练 run 正常加载
2. replay 可以在 test split 上跑完整个 segment
3. 不出现非有限值传播
4. 能输出 segment 级与全局级误差摘要
5. 能为后续多组实验保留统一 artifact 契约
6. 默认图包保持单图 3 到 4 个子窗，标题与图例位于数据区域之外或不遮挡曲线
7. 长时长 replay 配置必须记录 `dt_s / min_seconds / max_seconds_per_segment`，
   以便复核实际验证时长

## 4.1 当前推荐可视化

离线评估主图包：

- `prediction_trace_acc_axes.*`
- `prediction_trace_gyro_axes.*`
- `prediction_trace_vel_axes.*`

这三张图分别分析加速度、角速度和速度，每张图只放 X/Y/Z 三个共享 x 轴子窗。
默认窗口建议使用 50s，短 0.1s 样例图只作为排查补充，不作为主图包。

replay model selection 主图包：

- `replay_model_compare.*`
  - 当前版式为 2x2 四窗，集中比较 `rmse_global / final_step / growth / threshold failure`
- `replay_long_horizon_curves.*`
  - 当前版式为 2x2 四窗，比较 step-wise 误差与生存率

## 5. 风险点

- 当前 replay 仍是假设未来控制/上下文模板已知，不是闭环控制
- 若继续使用 `pred_len>1` 模型做 step()，其语义仍弱于真正的单步模型
- 现有 `masked metrics` 在 KF dense target 主线上不能再解释成“仅 DVL 命中监督表现”
- 长序列 replay 只能证明“经验型求解器可重放”，不能单独证明闭环可用
