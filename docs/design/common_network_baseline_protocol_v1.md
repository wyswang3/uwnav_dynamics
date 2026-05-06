# Common Network Baseline Protocol v1

更新时间：2026-05-06

本文档定义后续用于论文与技术汇报的“常见网络方案对照实验”协议。

目标不是重新打开无约束的大规模搜索，而是在当前状态转移求解器主线已经收口的基础上，
用可复现、可审计的对照组回答两个问题：

```text
1. 当前网络架构是否优于常见时序建模方案？
2. 当前训练思路是否优于普通监督训练目标？
```

当前文档是实验设计与后续落地依据，不表示以下所有 baseline 已经实现或完成训练。

当前代码底座状态：

```text
已新增常见网络 baseline：
src/uwnav_dynamics/models/nets/common_baselines.py

已接入模型工厂：
baseline_mlp
baseline_gru
baseline_tcn
baseline_transformer

已新增普通监督目标：
loss.type = state_mse
loss.type = state_huber

尚未新增正式 launch 配置：
configs/launch/pooltest02_common_network_baselines_8gpu_v1.yaml
configs/launch/pooltest02_training_objective_ablation_8gpu_v1.yaml
```

---

## 1. 当前事实边界

当前推荐的主线状态转移模型不是纯裸 LSTM，也不是已经胜出的 stable core。

当前主线可以描述为：

```text
KF / ESKF 代理状态数据
    ↓
S1Predictor
    ↓
LSTM encoder
    ↓
grouped state head
    ↓
dY / logvar
    ↓
y_hat = y0 + cumsum(dY)
    ↓
transition_balance loss
    ↓
50s autoregressive replay 筛选
```

当前主线配置证据：

```text
configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

其中：

```yaml
model:
  name: s1_predictor
  pred_len: 1
  rnn_hidden: 320
  rnn_layers: 2
  head_mode: grouped
  blocks:
    thruster_lag:
      enabled: false
    hydro_ssm:
      enabled: false
    damping:
      enabled: false
    uncertainty:
      enabled: false
```

因此论文或汇报中不应写成：

```text
我们只是使用了普通 LSTM。
```

更准确的表述是：

```text
我们采用基于 LSTM 历史编码器的分组状态转移网络，
结合 KF / ESKF 代理状态、状态增量 rollout 契约和 transition_balance 训练目标，
构建可用于短时状态转移与长序列 replay 的经验型求解器。
```

stable core 当前定位为探索性分支：

```text
src/uwnav_dynamics/models/nets/stable_transition_core.py
configs/launch/pooltest02_stable_transition_core_8gpu_v1.yaml
```

截至 2026-05-06 的第一轮离线 eval，5 个 STC 候选均已能完成训练与 eval，
但相对 StepBase anchor 的 masked RMSE / MAE / tail 指标明显更差。
因此 STC v1 不应作为当前主线优越性结论，只能作为物理内嵌求解器的后续研究方向。

---

## 2. 对照实验的核心原则

所有 baseline 必须固定以下条件，避免把数据差异、split 差异或训练预算差异误判为网络结构优势。

固定项：

```text
数据路线：quality_step_v1
数据目录：data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1
状态来源：因果 KF / ESKF 代理状态
输入：PWM + KF state/context + power
输出：AccKf(3) + GyroKf(3) + VelKf(3)
pred_len：1
dt：0.01 s
split：沿用当前 train/val/test 与 purge 契约
scaler：run-scoped train split scaler
rollout：y_hat = y0 + dY
eval split：test
replay：50 s autoregressive replay
```

训练预算原则：

```text
epochs = 120
optimizer = AdamW
scheduler = ReduceLROnPlateau
batch_size = 1024  # 单步任务默认
amp = true
seed 第一轮固定，第二轮 Top-K 多 seed
```

如果某个 baseline 因显存限制必须降低 batch size 或 hidden size，必须在文档和配置中明确记录，
并在结果表中标注。

---

## 3. Baseline 分层

对照组按用途分为三层。

### 3.1 非深度 baseline

用途：

证明网络模型相对简单经验模型确实有必要。

候选：

| 名称 | 形式 | 说明 |
| --- | --- | --- |
| `persistence` | `y_next = y0` | 零阶保持，状态转移任务最低基线 |
| `linear_arx_ridge` | `[history, control] -> dY` | 线性 ARX / ridge，对照浅层可解释模型 |
| `mlp_narx` | `flatten(history, control) -> dY` | 无循环结构的非线性 NARX |

说明：

- `persistence` 和 `linear_arx_ridge` 可以优先作为离线 baseline 实现，不必占用 GPU。
- `mlp_narx` 可作为轻量深度 baseline 进入 GPU 矩阵。

### 3.2 常见时序网络 baseline

用途：

证明当前 LSTM grouped 方案不是因为“所有时序网络都差不多”，而是在本任务下更合适。

候选：

| 名称 | 结构 | 对照目的 |
| --- | --- | --- |
| `gru_joint` | GRU + joint head | 与 LSTM 对比门控结构差异 |
| `lstm_joint` | LSTM + joint head | 分离 grouped head 的贡献 |
| `lstm_grouped` | LSTM + grouped head | 当前主线架构 |
| `tcn_small` | causal 1D CNN / TCN | 对照卷积时序建模 |
| `transformer_small` | Transformer encoder | 对照注意力时序建模 |

第一版应避免把 Transformer 规模做得过大。
目标是比较常见网络思路，不是让参数规模压倒公平性。

### 3.3 当前项目分支

用途：

证明当前选择不是旧模块堆叠，也不是 stable core v1。

候选：

| 名称 | 结构 | 当前定位 |
| --- | --- | --- |
| `stepbase_current` | S1Predictor + LSTM + grouped head + transition_balance | 当前主线 |
| `b4u1_legacy` | thruster/hydro/uncertainty blocks | 历史阶段对照，不作为当前默认 |
| `stable_transition_core_v1` | dissipative core variants | 探索性物理内嵌分支 |

注意：

`b4u1_legacy` 若纳入论文对照，需要明确它属于旧数据/旧阶段或历史消融结论，
不要与当前 `quality_step_v1` 主线混淆。

---

## 4. 两条证据链

### 4.1 架构优越性实验

固定训练目标：

```text
loss = transition_balance
```

比较对象：

```text
mlp_narx
gru_joint
lstm_joint
lstm_grouped
tcn_small
transformer_small
stable_transition_core_v1
stepbase_current
```

回答问题：

```text
在同一数据、同一 loss、同一 rollout 契约下，
LSTM grouped 是否比常见结构更适合作为状态转移模型？
```

第一轮建议 8GPU 分配：

| GPU | 候选 | 角色 |
| --- | --- | --- |
| 0 | `mlp_narx_seed10` | baseline |
| 1 | `gru_joint_seed10` | baseline |
| 2 | `lstm_joint_seed10` | baseline |
| 3 | `lstm_grouped_seed10` | primary |
| 4 | `tcn_small_seed10` | baseline |
| 5 | `transformer_small_seed10` | baseline |
| 6 | `stable_transition_core_best_seed10` | exploratory |
| 7 | `stepbase_anchor_seed11` | anchor |

若非深度 baseline 已离线跑完，也可以把 GPU0 用于另一个 seed 的 `lstm_grouped` anchor。

### 4.2 训练思路优越性实验

固定架构：

```text
architecture = S1Predictor + LSTM + grouped head
```

比较训练目标：

| 名称 | loss 形式 | 对照目的 |
| --- | --- | --- |
| `mse` | state MSE | 普通回归基线 |
| `huber` | state Huber | 抗离群基线 |
| `nll_diag` | Gaussian NLL | 异方差输出基线 |
| `nll_final` | NLL + final step | 检查末步约束贡献 |
| `transition_balance` | 当前组合目标 | 当前主线 |
| `transition_balance_masked` | 当前 masked 评估口径 | 最终筛查口径 |

回答问题：

```text
在同一网络结构下，
transition_balance 是否改善 final step、tail、growth 与 bias，
而不仅仅改善短窗口 RMSE？
```

第一轮建议 8GPU 分配：

| GPU | 候选 | 角色 |
| --- | --- | --- |
| 0 | `loss_mse_seed10` | baseline |
| 1 | `loss_huber_seed10` | baseline |
| 2 | `loss_nll_seed10` | baseline |
| 3 | `loss_nll_final_seed10` | ablation |
| 4 | `loss_tb_seed10` | primary |
| 5 | `loss_tb_seed11` | primary anchor |
| 6 | `loss_tb_seed12` | primary anchor |
| 7 | `loss_tb_masked_seed10` | final-screening variant |

---

## 5. 指标与判定

### 5.1 离线 eval 指标

主表不应只放 `RMSE(global)`。

推荐列：

```text
status
best_monitor
rmse_global_masked
mae_global_masked
final_step_rmse_global_masked
final_step_mae_global_masked
tail_abs_p95_masked
tail_abs_p99_masked
rmse_growth_masked
mae_growth_masked
worst_abs_bias_masked
epochs_ran
train_wall_time_sec
```

解释口径：

| 指标 | 说明 |
| --- | --- |
| `rmse_global_masked` | 总体二范数误差，说明拟合能力 |
| `mae_global_masked` | 总体绝对误差，降低少数大误差支配 |
| `final_step_*` | rollout 末端误差，贴近状态转移用途 |
| `tail_abs_p95/p99` | 尾部风险，贴近控制前安全筛查 |
| `rmse_growth / mae_growth` | 误差增长倾向，筛查长期发散风险 |
| `worst_abs_bias` | 系统性偏差，筛查长期漂移来源 |

### 5.2 50s replay 指标

进入最终结论的候选必须跑 50s autoregressive replay。

推荐列：

```text
nonfinite_trigger_count
rmse_global
mae_global
final_step_rmse_global_mean
rmse_growth_p95
tail_abs_p95_global
tail_abs_p99_global
worst_abs_bias
threshold_failure_count
```

硬失败条件：

```text
replay 进程失败
nonfinite_trigger_count > 0
rmse_global 为 NaN/Inf
tail_abs_p99_global 为 NaN/Inf
```

### 5.3 最终判定规则

不能只按单一 RMSE 选型。

推荐判定：

```text
可进入 Top-K 多 seed：
  status = ok
  nonfinite_trigger_count = 0
  rmse_global / mae_global 不明显劣于当前 StepBase
  final_step / tail / growth / bias 至少两类指标优于常见 baseline

可写成优越性结论：
  多 seed 平均更好
  多 seed 方差更低或相当
  replay 中不发散
  final/tail/growth/bias 指标能解释其控制前优势
```

如果某个 baseline 的 `rmse_global` 略低，但 `rmse_growth_p95` 或 `tail_abs_p99_global`
明显更差，应表述为：

```text
该模型短窗口拟合能力较强，但不适合作为状态转移求解器主线。
```

---

## 6. 计划新增配置

本节列出后续应新增的配置文件。当前尚未落地，不要把它们当作已存在文件引用。

建议新增：

```text
configs/launch/pooltest02_common_network_baselines_8gpu_v1.yaml
configs/launch/pooltest02_training_objective_ablation_8gpu_v1.yaml
configs/launch/pooltest02_common_network_baselines_8gpu_v1_replay_only.yaml
configs/launch/pooltest02_training_objective_ablation_8gpu_v1_replay_only.yaml
```

如果需要新增模型实现，建议位置：

```text
src/uwnav_dynamics/models/nets/common_baselines.py
```

这些模型应通过现有工厂接入：

```text
src/uwnav_dynamics/models/nets/factory.py
```

并保持输出契约：

```python
dY, logvar = model(X)
```

或：

```python
dY, logvar, aux = model.forward_with_aux(X)
```

当前已接入的 `model.name`：

```text
baseline_mlp
baseline_gru
baseline_tcn
baseline_transformer
```

---

## 7. 推荐执行顺序

### 阶段 A：非深度 baseline

阶段目标：

建立 persistence / linear ARX 的最低基线。

最小实现：

```text
离线 baseline runner 或 eval helper
```

运行输出：

```text
out/baselines/pooltest02_quality_step_v1/
```

验证标准：

```text
能写出与 train_matrix summary.csv 可合并的指标表
```

风险点：

不要把非深度 baseline 的输入历史或数据 split 做得和深度模型不同。

### 阶段 B：常见网络架构矩阵

阶段目标：

在同一 loss 下比较 MLP/GRU/LSTM/TCN/Transformer/STC。

最小实现：

```text
新增必要模型类
新增 model.name dispatch
新增 8GPU launch 配置
```

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_common_network_baselines_8gpu_v1.yaml
```

验证标准：

```text
out/train_matrix/pooltest02_common_network_baselines_8gpu_v1/summary.csv
```

### 阶段 C：训练目标矩阵

阶段目标：

固定 LSTM grouped 架构，验证 transition_balance 的贡献。

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_training_objective_ablation_8gpu_v1.yaml
```

验证标准：

```text
同一架构下 transition_balance 在 final/tail/growth/bias 上有稳定优势
```

### 阶段 D：Top-K replay 复核

阶段目标：

把架构矩阵和训练目标矩阵的 Top-K 候选送入 50s replay。

运行命令示例：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_common_network_baselines_8gpu_v1_replay_only.yaml
```

验证标准：

```text
out/replay_matrix/<variant>/ranking.csv
out/server_pipeline/<variant>/final_selection.csv
```

---

## 8. 论文表述建议

推荐结论表述：

```text
我们的方法并不是简单使用 LSTM。
在相同 KF/ESKF 代理状态、相同输入输出契约和相同训练预算下，
LSTM grouped transition model 结合 transition_balance 目标，
在 50s autoregressive replay 中表现出更小的末端误差、
更低的尾部风险和更弱的误差增长，因此更适合作为控制器内部短时状态转移求解器。
```

避免表述：

```text
LSTM 天然优于 Transformer。
神经网络替代了精确水动力学模型。
stable core 已经证明更优。
```

更稳妥的表述：

```text
在本文数据规模、采样频率和状态转移任务约束下，
当前 LSTM grouped + transition_balance 路线比若干常见网络 baseline
更符合短时状态转移求解器的工程指标。
```

---

## 9. 下次恢复工作入口

下次继续时，建议按以下顺序：

```text
1. 阅读本文档，确认 baseline 协议。
2. 阅读 docs/design/current_transition_solver_selection.md，确认当前主线 StepBase。
3. 阅读 docs/design/stable_transition_core_8gpu_matrix_v1.md，确认 STC 当前定位。
4. 先落地 common network baseline 的最小模型工厂接入。
5. 再新增 8GPU launch 配置与测试。
```

不要一开始就重新跑大规模训练。先保证：

```text
配置可解析
模型 forward 契约一致
小样本 smoke 能跑通
summary.csv 字段可比较
```
