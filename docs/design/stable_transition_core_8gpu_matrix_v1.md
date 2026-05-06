# Stable Transition Core 8GPU Matrix v1

更新时间：2026-05-06

本文档定义下一阶段“稳定状态转移内核”的 8GPU 结构筛选方案。

目标不是继续堆叠旧 optional blocks，而是新增一类独立的、物理规律内嵌的状态转移求解器，
并通过 8GPU 并行实验筛选出更适合长期 rollout / 控制闭环接入的内核结构。

当前实现状态：

```text
已新增模型文件：
src/uwnav_dynamics/models/nets/stable_transition_core.py

已新增模型工厂：
src/uwnav_dynamics/models/nets/factory.py

已接入主运行链路：
src/uwnav_dynamics/train/run_train.py
src/uwnav_dynamics/eval/evaluate.py
src/uwnav_dynamics/solver/transition_solver.py

已新增测试：
tests/test_stable_transition_core.py

已新增训练配置：
configs/train/pooltest02_stable_transition_core_step_v1.yaml

已新增 8GPU 矩阵配置：
configs/launch/pooltest02_stable_transition_core_8gpu_v1.yaml
```

第一版已支持：

```text
model.name: stable_transition_core
model.core_type:
  - stable_diag_damp
  - implicit_euler
  - control_affine
  - residual_budget
  - energy_budget
```

尚未实现：

```text
stable_cross_damp
mode_gated_stable_core
jacobian_contraction_loss
显式 residual/core ratio artifact 写盘
显式 low-thrust energy artifact 写盘
```

---

## 1. 背景与问题定义

当前默认状态转移模型仍是：

```text
S1Predictor
  -> LSTM encoder
  -> grouped head
  -> dY / logvar
  -> y_hat = y0 + cumsum(dY)
```

代码证据：

```text
src/uwnav_dynamics/models/nets/s1_predictor.py
src/uwnav_dynamics/models/utils/rollout.py
src/uwnav_dynamics/train/run_train.py
```

当前主线已经通过 `transition_balance` 增强了训练目标，包括：

- state Huber / MSE；
- final step loss；
- late horizon loss；
- delta transition loss；
- logvar 正则；
- acc / gyro / vel 语义组加权。

但这些约束主要仍是损失层面的误差约束，并没有把以下物理规律写进状态转移方程本身：

- 无输入或低输入时状态应趋于耗散；
- 速度和角速度不应长期无界增长；
- 推进器输入应经过有界映射；
- 网络残差不应长期接管物理项；
- autoregressive replay 中局部误差传播应被抑制。

因此下一阶段重点是：

```text
从“自由 dY 预测器”
升级为
“带耗散、有界控制、有界残差的稳定状态转移内核”
```

---

## 2. 不依赖旧模块的原则

本阶段不依赖现有 optional blocks：

```text
thruster_lag
hydro_ssm
damping
uncertainty
```

原因：

1. 当前最终方案 `StepBase s11` 的证据链并不是这些模块胜出。
2. 旧 blocks 仍是围绕 `S1Predictor -> dY` 的外围增强，不是新的状态转移内核。
3. 下一阶段要比较的是“内核方程形式”，不是继续做旧模块组合消融。

新增模型应独立放置，例如：

```text
src/uwnav_dynamics/models/nets/stable_transition_core.py
```

它可以复用已有公共工具：

```text
execution_layout.py
semantic_output_layout.py
rollout.py
state_transition.py
```

但不 import 或依赖：

```text
src/uwnav_dynamics/models/blocks/*
```

当前代码已按该边界落地：

```text
stable_transition_core.py 只依赖 execution_layout / semantic_output_layout，
不 import models.blocks.*。
```

---

## 3. 固定实验契约

为了保证和当前主线可直接对照，第一阶段所有候选固定以下契约。

### 3.1 数据与 split

```text
数据路线：quality_step_v1
数据目录：data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1
状态来源：因果 KF / ESKF 代理状态
输入：PWM + KF state/context + power
输出：AccKf(3) + GyroKf(3) + VelKf(3)
pred_len：1
dt：0.01 s
split：contiguous_purged_v2
scaler：run-scoped train split scaler
```

### 3.2 输出接口

新增模型必须保持训练、评估、replay 现有接口：

```python
dY, logvar = model(X)
```

或：

```python
dY, logvar, aux = model.forward_with_aux(X)
```

主输出仍是：

```text
dY:     (B, H, 9)
logvar: (B, H, 9)
```

训练和评估仍使用：

```text
y0 = X[:, -1, y_in_idx]
y_hat = y0 + cumsum(dY)
```

这样可以最小化对 train / eval / replay 链路的改动。

当前实现通过 `build_state_predictor()` 统一构造模型：

```python
from uwnav_dynamics.models.nets.factory import build_state_predictor

model = build_state_predictor(cfg)
```

旧配置继续构造 `S1Predictor`：

```yaml
model:
  name: s1_predictor
```

新配置构造 `StableTransitionCore`：

```yaml
model:
  name: stable_transition_core
  core_type: stable_diag_damp
```

### 3.3 不改变的内容

第一阶段不改变：

- dataset build；
- split/scaler artifact 契约；
- eval artifact 契约；
- replay matrix 汇总逻辑；
- 当前 `S1Predictor` 默认候选；
- 现有历史结果文件。

---

## 4. 新内核总形式

统一记状态：

```text
x = [a, omega, v]
```

其中：

```text
a     = AccKfX/Y/Z
omega = GyroKfX/Y/Z
v     = VelKfX/Y/Z
```

建议新模型内部采用：

```text
history encoder:
    X_{k-L+1:k} -> c_k

bounded actuator map:
    u_k, c_k -> q_k

dissipative transition core:
    omega_k, v_k, q_k, c_k -> omega_{k+1}, v_{k+1}

algebraic acceleration head:
    q_k, v_k, c_k -> a_{k+1}

bounded residual:
    small correction, not free full-state takeover
```

最终仍输出：

```text
y_next = [a_{k+1}, omega_{k+1}, v_{k+1}]
dY = y_next - y0
```

---

## 5. 第一轮 8GPU 结构筛选矩阵

第一轮目标是结构发现，不做多 seed 统计。

固定：

```text
第一轮新内核 seed = 10
epochs = 120
metric = val_transition_score
eval_split = test
replay = 50 s
```

当前已落地的第一版配置：

```text
基础训练配置：
configs/train/pooltest02_stable_transition_core_step_v1.yaml

8GPU launch：
configs/launch/pooltest02_stable_transition_core_8gpu_v1.yaml
```

第一轮 8 张 GPU 的实际分配为：

| GPU | 候选名 | 方案类别 | 核心假设 |
| --- | --- | --- | --- |
| 0 | `stc_diag_seed10` | 对角耗散内核 | `vel / gyro` 分量独立指数衰减，残差受限 |
| 1 | `stc_implicit_seed10` | 半隐式稳定更新 | 用类似 implicit Euler 的分母结构抑制高速发散 |
| 2 | `stc_control_affine_seed10` | 控制仿射内核 | `x_next = f_diss(x) + B(c) u_eff + r` |
| 3 | `stc_residual_budget_seed10` | 残差预算内核 | 强约束 residual/core ratio，避免神经残差接管物理项 |
| 4 | `stc_energy_budget_seed10` | 能量预算内核 | 更强耗散、更小残差，优先压制低输入发散 |
| 5 | `step_base_seed10` | 当前主线锚点 | 同批次环境噪声锚点 |
| 6 | `step_base_seed11` | 当前主线锚点 | 当前最终推荐方案对应 seed |
| 7 | `step_base_seed12` | 当前主线锚点 | 基线 seed 方差锚点 |

最初规划中的完整 8 结构候选如下；其中 `stable_cross_damp` 和
`mode_gated_stable_core` 尚未实现，不进入第一版 launch：

| GPU | 候选名 | 方案类别 | 核心假设 |
| --- | --- | --- | --- |
| 0 | `StepBase_ref_s11` | 当前默认锚点 | 用当前 `StepBase s11` 作为 replay 与误差基线 |
| 1 | `StableDiagDamp_s11` | 对角耗散内核 | `vel / gyro` 分量独立指数衰减，残差受限 |
| 2 | `StableCrossDamp_s11` | 低秩交叉耗散 | 在对角耗散上加入低秩耦合，表达轴间水动力耦合 |
| 3 | `ControlAffineCore_s11` | 控制仿射内核 | `x_next = f_diss(x) + B(c) u_eff + r` |
| 4 | `ImplicitEulerCore_s11` | 半隐式稳定更新 | 用类似 implicit Euler 的分母结构抑制高速发散 |
| 5 | `EnergyBudgetCore_s11` | 能量预算内核 | 显式约束低输入时速度/角速度能量不增长 |
| 6 | `ResidualBudgetCore_s11` | 残差预算内核 | 强约束 residual/core ratio，避免神经残差接管物理项 |
| 7 | `ModeGatedStableCore_s11` | 模态门控稳定内核 | 区分静止、推进、转向等工况，切换稳定参数 |

---

## 6. 各候选结构定义

### 6.1 `StepBase_ref_s11`

目的：

作为当前默认方案锚点，不参与新模型实现。

来源：

```text
out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B0_grouped_tb_seed11
```

用途：

- 第一轮结构候选必须超过或接近该方案；
- 若新方案误差更低但 replay 增长更高，不应直接替换默认方案。

### 6.2 `StableDiagDamp_s11`

核心形式：

```text
lambda_v     = exp(-dt * softplus(d_v))
lambda_omega = exp(-dt * softplus(d_omega))

v_next     = lambda_v * v + dt * (F_u + r_v)
omega_next = lambda_omega * omega + dt * (T_u + r_omega)
```

约束：

```text
0 < lambda_v, lambda_omega <= 1
|r_v|, |r_omega| <= r_max
```

优点：

- 最小物理稳定结构；
- 参数可解释；
- 实现风险最低。

风险：

- 对复杂耦合表达能力可能不足。

### 6.3 `StableCrossDamp_s11`

核心形式：

```text
D = diag(softplus(d)) + U U^T
v_next = v + dt * (-D v + F_u + r_v)
```

稳定处理：

```text
D positive semi-definite
dt * ||D|| controlled
```

优点：

- 能表达轴间耦合；
- 比完全自由矩阵更稳定。

风险：

- 如果低秩耦合过强，训练早期可能不稳定；
- 需要对 `U` 或谱范数做限制。

当前状态：

```text
未实现。第一版先用 stable_diag_damp / implicit_euler / control_affine 等低风险结构建立基线。
```

### 6.4 `ControlAffineCore_s11`

核心形式：

```text
u_eff = bounded_mlp(u, context)
B = bounded_matrix(context)

z_next = dissipative_base(z) + B u_eff + r
```

其中：

```text
z = [omega, v]
```

优点：

- 明确区分自然耗散和控制输入；
- 更适合作为控制器内部模型。

风险：

- `B(context)` 如果太自由，可能退化为普通黑箱；
- 需要 residual budget 和 matrix norm 限制。

### 6.5 `ImplicitEulerCore_s11`

核心形式：

```text
v_next = (v + dt * (F_u + r_v)) / (1 + dt * softplus(d_v))
```

`omega` 同理。

优点：

- 数值稳定性强；
- 对高速或大阻尼情况下更不容易爆炸。

风险：

- 表达能力可能偏保守；
- 若阻尼过大，可能造成响应过慢。

### 6.6 `EnergyBudgetCore_s11`

核心形式与 `StableDiagDamp` 类似，但额外输出能量预算项：

```text
E_z = ||v||^2 + alpha ||omega||^2
```

训练时增加：

```text
low_thrust_mask -> relu(E_next - E_now - eps)
```

优点：

- 直接针对长期发散；
- 指标容易解释。

风险：

- 如果 low-thrust mask 定义不准，可能压制真实运动。

### 6.7 `ResidualBudgetCore_s11`

核心思想：

仍使用稳定物理 core，但显式限制残差贡献：

```text
ratio = ||r|| / (||core_update|| + eps)
loss += relu(ratio - ratio_max)
```

优点：

- 防止网络把物理 core 当摆设；
- 便于诊断“模型到底靠物理项还是残差项”。

风险：

- ratio 阈值需要保守设置；
- 早期训练可能收敛慢。

### 6.8 `ModeGatedStableCore_s11`

核心形式：

```text
pi = softmax(gate(context))
theta = sum_i pi_i theta_i
```

每个 mode 都是稳定内核参数：

```text
mode 0: low thrust / drift
mode 1: forward thrust
mode 2: turn / lateral
mode 3: mixed maneuver
```

优点：

- 能表达不同工况；
- 仍保持每个 mode 内部稳定。

风险：

- gate 可能塌缩到单一 mode；
- 第一轮只建议 3-4 个 mode，不要过大。

当前状态：

```text
未实现。待第一轮低风险稳定内核跑通后再加入。
```

---

## 7. 第一轮排序规则

第一轮不要只按 `rmse_global` 排序。

硬性失败条件：

```text
nonfinite_trigger_count > 0
replay 进程失败
eval artifact 不完整
rmse_global 为 NaN/Inf
tail_abs_p99_global 为 NaN/Inf
```

主排序指标：

| 优先级 | 指标 | 方向 | 说明 |
| --- | --- | --- | --- |
| 1 | `nonfinite_trigger_count` | 必须为 0 | 稳定求解器底线 |
| 2 | `rmse_growth_p95` | 越低越好 | 长期发散风险 |
| 3 | `tail_abs_p99_global` | 越低越好 | 极端尾部误差 |
| 4 | `tail_abs_p95_global` | 越低越好 | 常见尾部误差 |
| 5 | `final_step_rmse_global_mean` | 越低越好 | replay 末端状态误差 |
| 6 | `rmse_global` | 越低越好 | 全局误差 |
| 7 | `mae_global` | 越低越好 | 全局绝对误差 |
| 8 | `worst_abs_bias` | 越低越好 | 系统性偏差 |

新增诊断指标：

```text
residual_core_ratio_mean
residual_core_ratio_p95
low_thrust_energy_growth_mean
low_thrust_energy_growth_p95
stable_lambda_min / stable_lambda_max
```

这些指标第一版尚未落盘，不进入当前 launch 的硬判定；后续实现 artifact 后，
应写入候选 `metrics.yaml` 或单独 CSV，用于解释结构是否真的靠稳定 core 工作。

### 7.1 第一版有效性标准线

标准线分三层，避免只靠单个 RMSE 判断。

#### A. 硬门槛

任何候选若触发以下情况，直接判为不可用：

```text
训练失败或评估失败
checkpoint 无法被 eval / solver 加载
metrics.yaml 缺少主指标
nonfinite_trigger_count > 0
rmse_global / rmse_global_masked / tail_abs_p99_global 出现 NaN 或 Inf
```

#### B. 同批次离线 eval 门槛

第一轮 launch 中保留了三个 `StepBase` anchor，用于估计同批次训练噪声。

新内核至少需要满足：

```text
rmse_global_masked 不高于 StepBase anchor 均值的 1.15 倍
mae_global_masked 不高于 StepBase anchor 均值的 1.15 倍
final_step_rmse_global_masked 不高于 StepBase anchor 均值的 1.20 倍
tail_abs_p95_masked 不高于 StepBase anchor 均值的 1.20 倍
worst_abs_bias_masked 不高于 StepBase anchor 均值的 1.20 倍
```

若某个 stable core 在误差略高的情况下显著降低 tail / growth，
可以进入 replay 复核，但不能直接替代主线。

#### C. 50 秒 replay 有效线

当前最终求解器 `StepBase s11` 的历史 replay 锚点为：

| 指标 | `StepBase s11` replay 数值 |
| --- | ---: |
| `rmse_global` | `0.2519594653787598` |
| `mae_global` | `0.1161450209257922` |
| `final_step_rmse_global_mean` | `0.18731854021707864` |
| `rmse_growth_p95` | `131.0431527221474` |
| `tail_abs_p95_global` | `0.49592751264572055` |
| `tail_abs_p99_global` | `1.1673504388332405` |
| `worst_abs_bias` | `0.08816736124534703` |
| `nonfinite_trigger_count` | `0` |

第一版判定建议：

```text
最小可用：
  nonfinite_trigger_count = 0
  rmse_global <= 0.3024        # 约为 StepBase s11 的 1.20 倍
  final_step_rmse_global_mean <= 0.2248
  tail_abs_p99_global <= 1.4008

有明确稳定收益：
  rmse_growth_p95 <= 117.94    # 至少比 StepBase s11 低 10%
  或 tail_abs_p99_global <= 1.0506
  同时 rmse_global 不超过 StepBase s11 的 1.15 倍

可考虑替换主线：
  nonfinite_trigger_count = 0
  rmse_growth_p95 / tail_abs_p99_global / final_step_rmse_global_mean 至少两项优于 StepBase s11
  rmse_global 与 mae_global 均不超过 StepBase s11 的 1.10 倍
  Top-K 多 seed 后仍成立
```

这些数值是第一版工程筛选线，不是论文最终结论。最终结论必须来自同一
`summary.csv / ranking.csv / final_selection.csv` 的可复核证据。

---

## 8. 第二轮 Top-K 多 seed 确认

第一轮结束后，选择 Top-2 或 Top-3。

推荐分配：

### 方案 A：Top-2 强确认

```text
Top1: seed 8 / 9 / 10 / 11
Top2: seed 8 / 9 / 10 / 11
```

适用场景：

- 第一轮中前两名明显领先；
- 第三名指标差距较大或存在风险。

### 方案 B：Top-3 广确认

```text
Top1: seed 8 / 9 / 10
Top2: seed 8 / 9 / 10
Top3: seed 8 / 9
```

适用场景：

- 第一轮前 3 名各有优势；
- 需要判断误差、稳定性、尾部风险之间的 tradeoff。

第二轮排序看 family 均值与方差：

```text
mean_rmse_growth_p95
std_rmse_growth_p95
mean_tail_abs_p99_global
std_tail_abs_p99_global
mean_final_step_rmse_global_mean
success_count
```

最终选择不应只取单 seed 最优，而应取：

```text
稳定性均值好
seed 方差低
失败率低
尾部风险可解释
```

---

## 9. 最小实现阶段划分

### 阶段 1：设计与配置落地

阶段目标：

固定新模型接口、8GPU 矩阵、排序指标和失败判据。

最小实现：

- 本文档；
- 后续新增 launch YAML 草案；
- 不改训练代码。

运行命令：

```bash
git status --short
```

验证标准：

- 文档位于 `docs/design/`；
- 方案不依赖旧 blocks；
- 8GPU 矩阵定义清楚；
- 指标和失败判据明确。

风险点：

- 如果直接跳到代码实现，容易把结构筛选与工程重构混在一起。

### 阶段 2：新增稳定内核模型

阶段目标：

新增 `StableTransitionCore`，保持 `(dY, logvar)` 输出契约。

最小实现：

```text
src/uwnav_dynamics/models/nets/stable_transition_core.py
```

并在模型 factory 或训练入口中支持：

```yaml
model:
  name: stable_transition_core
```

当前状态：

```text
已完成第一版。
```

运行命令：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_execution_layout_contract.py \
  tests/test_transition_balance_design.py \
  tests/test_stable_transition_core.py
```

验证标准：

- 输入 `(B,L,Din)`；
- 输出 `(B,H,9)` 的 `dY/logvar`；
- `y0 + dY` 后 shape 与现有模型一致；
- CPU 上能跑最小 forward 测试。

风险点：

- 不要在第一版引入太多 config 字段；
- 不要破坏 `S1Predictor` 现有默认路径。

### 阶段 3：新增物理正则

阶段目标：

让训练能记录并约束稳定内核的 residual / energy 诊断。

最小实现：

- `residual_budget_loss`
- `low_thrust_energy_loss`
- `bounded_delta_loss`
- 对应 metrics 记录

运行命令：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_composite_loss_p0.py \
  tests/test_trainer_mask_batches.py
```

验证标准：

- 旧 loss 配置仍可运行；
- 新 loss 权重为 0 时不改变旧行为；
- 新模型可输出诊断项。

风险点：

- 正则权重过大可能压制拟合；
- 第一轮应保守设置。

### 阶段 4：8GPU 结构矩阵

阶段目标：

一次性并行比较 8 个结构候选。

最小实现：

```text
configs/launch/pooltest02_stable_core_8gpu_v1.yaml
```

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_stable_core_8gpu_v1.yaml
```

随后运行 replay matrix：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_stable_core_8gpu_v1_replay.yaml
```

验证标准：

- 8 个候选均写出 train summary；
- 成功候选均写出 eval/replay metrics；
- 生成 summary/ranking；
- 能和 `StepBase s11` 对齐比较。

风险点：

- 第一轮结构差异较大，可能有候选训练失败；
- 失败本身是筛选信息，但必须记录失败原因。

---

## 10. 推荐第一版配置范围

为了降低实现和训练风险，第一版建议：

```text
rnn_hidden: 320
rnn_layers: 2
model.name: stable_transition_core
core_type: stable_diag_damp
core_hidden: 160
residual_bound: 0.25
damping_min: 1e-4
damping_max: 20.0
control_bound: 3.0
logvar_clip: [-10, 6]
```

物理正则初值：

```text
residual_budget_weight: 0.05
low_thrust_energy_weight: 0.05
bounded_delta_weight: 0.02
jacobian_contraction_weight: 0.0
```

说明：

- 第一版不启用 Jacobian contraction；
- 先看结构性耗散是否改善 replay；
- 如果第一轮发现 tail 仍发散，再引入局部收缩正则。

---

## 11. 预期结论形态

第一轮结束后应输出：

```text
out/train_matrix/pooltest02_stable_core_8gpu_v1/summary.csv
out/replay_matrix/pooltest02_stable_core_8gpu_v1/ranking.csv
out/replay_matrix/pooltest02_stable_core_8gpu_v1/compare_test/replay_model_compare.png
out/replay_matrix/pooltest02_stable_core_8gpu_v1/compare_test/replay_long_horizon_curves.png
```

最终汇报应回答：

1. 哪类稳定内核在 50s replay 中最抗发散；
2. 是否超过 `StepBase s11`；
3. 若没有超过，失败原因是误差、尾部风险、响应过慢还是残差不足；
4. 物理项是否真正承担主要转移职责；
5. 是否值得进入 Top-K 多 seed 第二轮。

---

## 12. 当前建议

优先实现顺序：

```text
1. StableDiagDamp
2. ImplicitEulerCore
3. ResidualBudgetCore
4. ControlAffineCore
5. EnergyBudgetCore
6. StableCrossDamp
7. ModeGatedStableCore
```

如果实现时间有限，第一批最小候选可收敛为：

```text
StepBase_ref
StableDiagDamp
ImplicitEulerCore
ControlAffineCore
ResidualBudgetCore
EnergyBudgetCore
StableDiagDamp_no_residual
StableDiagDamp_strong_residual_budget
```

这样即使高级结构尚未完成，也能用 8GPU 跑出第一轮稳定性证据。

---

## 13. 第一版 YAML 模板

第一版 stable core 训练配置可在当前 `quality_step_v1` 配置基础上替换 `model` 段。

示例：

```yaml
model:
  name: stable_transition_core
  din: 34
  dout: 9
  pred_len: 1
  rnn_hidden: 320
  rnn_layers: 2
  dropout: 0.0
  u_in_idx: [0, 1, 2, 3, 4, 5, 6, 7]
  y_in_idx: [8, 9, 10, 11, 12, 13, 14, 15, 16]

  core_type: stable_diag_damp
  core_hidden: 160
  residual_bound: 0.25
  damping_min: 0.0001
  damping_max: 20.0
  control_bound: 3.0
  dt: 0.01
```

说明：

- `blocks` 不再需要填写；
- `head_mode / group_head_hidden / use_hydro_feat / use_thruster_as_replacement` 不参与新模型；
- 第一版仍在 z-score 空间执行稳定转移；
- train / eval / replay 仍使用相同的 split/scaler 与 `y_hat = y0 + cumsum(dY)` 契约。
