# 项目架构说明

## 1. 系统目标

当前系统目标不是恢复完整高保真水动力学真值模型，也不是从仓库入口直接展开 ROS 2 / Gazebo 大系统联调。

当前系统的工程目标是：

- 从多源异步观测中构建统一、因果、可递推的状态代理量链路。
- 训练学习型短时状态转移模型，评估其作为求解器内核的可用性。
- 为控制仿真、controller replay 和 model-based RL 提供可复现的实验骨架。

因此，系统的核心对象是：

```text
u_t -> learned transition solver -> x_{t+1}
```

而不是：

```text
sensor streams -> black-box trajectory fitting only
```

## 2. 分层架构

当前仓库可按六层理解。

### 第 1 层：原始观测层

主要输入源：

- PWM：`100 Hz`
- IMU：`100 Hz`
- DVL：`10 Hz`
- Power：`5 Hz`

职责：

- 保留控制输入与多源观测的原始时间关系
- 为后续因果对齐与融合提供数据基础

### 第 2 层：预处理与对齐层

主链：

```text
raw IMU
-> transform
-> gravity compensation
-> bias correction
-> filter
```

关键模块：

- `src/uwnav_dynamics/preprocess/imu/transform.py`
- `src/uwnav_dynamics/preprocess/imu/gravity.py`
- `src/uwnav_dynamics/preprocess/imu/pipeline.py`
- `src/uwnav_dynamics/preprocess/align/aligner.py`

职责：

- 统一坐标表达
- 抑制重力项和 bias 误差
- 将异步、多频传感器收口到 `100 Hz` 主时间轴

### 第 3 层：因果融合层

关键模块：

- `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`
- `src/uwnav_dynamics/preprocess/fusion/cli_fuse_train_base.py`

职责：

- 以 IMU 高频传播为主线
- 用 DVL 速度观测做稀疏校正
- 输出体坐标系、统一时间轴下的状态代理量和质量上下文

当前核心输出包括：

- `AccKf`
- `GyroKf`
- `VelKf`
- `RollKf / PitchKf / YawKf`
- `HasDvlUpdate`
- `DtSinceDvl_s`
- `VelKfVarX / VelKfVarY / VelKfVarZ`

这里的 `KF / ESKF` 输出应解释为工程状态代理量，而非物理真值。

### 第 4 层：数据集构建层

关键模块：

- `src/uwnav_dynamics/preprocess/build_dataset.py`
- `src/uwnav_dynamics/train/data_pipeline.py`

职责：

- 从融合基础表构造滑窗训练样本
- 维护 `input_cols / target_cols` 契约
- 保证共享状态维 `X/Y` scaler 一致

当前活跃数据集分支：

1. `kf_ctx_v2`  
   `29 -> 9`，多步主线

2. `quality_v3`  
   `34 -> 9`，在 `kf_ctx_v2` 上加入状态质量上下文

3. `quality_step_v1`  
   `34 -> 9`，保持输入契约不变，但把监督 horizon 收口到单步

### 第 5 层：学习型动力学模型层

关键模块：

- `src/uwnav_dynamics/models/nets/s1_predictor.py`
- `src/uwnav_dynamics/models/losses/state_transition.py`
- `src/uwnav_dynamics/models/utils/execution_layout.py`

当前默认模型 family：

- `S1Predictor`

当前主语义：

- 输入：历史窗 `X_t`
- 输出：未来状态增量 `dY`
- rollout：`y_hat = y0 + cumsum(dY)`

可选结构模块：

- `thruster_lag`
- `hydro_ssm`
- `damping`
- `uncertainty`

当前不建议默认把 `uncertainty` 当作掩盖长时漂移的通道。

### 第 6 层：训练、评估与服务器矩阵层

训练与评估关键模块：

- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/eval/evaluate.py`
- `src/uwnav_dynamics/cli/train.py`
- `src/uwnav_dynamics/cli/train_matrix.py`

职责：

- 训练单个 run
- 按统一指标生成评估产物
- 在 8 卡服务器上并行运行实验矩阵

当前推荐 8 卡策略不是对小模型强上 DDP，而是：

- 每张卡独占一个单卡实验
- 同时跑满 8 个候选变体
- 用统一评估口径完成横向比较

对应矩阵入口：

- `configs/launch/pooltest02_s1_kf_quality_8gpu_v1.yaml`
- `configs/launch/pooltest02_s1_kf_quality_step_8gpu_v1.yaml`

### 第 7 层：状态求解器与 Replay 验证层

关键模块：

- `src/uwnav_dynamics/solver/transition_solver.py`
- `src/uwnav_dynamics/solver/replay.py`
- `src/uwnav_dynamics/cli/transition_replay.py`

职责：

- 将训练好的网络模型封装成经验型状态求解器
- 提供最小一步状态递推接口
- 基于现有实验数据执行长序列 autoregressive replay

当前边界：

- `pred_len=1` 模型是首选求解器主线
- `pred_len>1` 模型仅保守兼容为“取第一步预测”
- replay 只证明长序列递推可行性，不替代闭环控制验证

## 3. 系统级数据流

```text
PWM   (100 Hz)
IMU   (100 Hz)
DVL   (10 Hz)
Power (5 Hz)

        ↓

alignment
        ↓
causal KF / ESKF fusion
        ↓
proxy-state dataset build
        ↓
S1Predictor training
        ↓
offline rollout evaluation
        ↓
transition solver replay validation
        ↓
transition-solver readiness analysis
        ↓
controller / simulator / RL interface preparation
```

## 4. 当前契约与活跃分支

### 4.1 多步主线

主配置：

- 数据集：`configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
- 训练：`configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`

用途：

- 保持与现有 rollout 评估链的可比性
- 继续筛选长时误差更稳定的候选

### 4.2 单步状态转移实验线

主配置：

- 数据集：`configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
- 训练：`configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`

用途：

- 向 `x_{t+1} = f(x_t, u_t, c_t)` 的形式收口
- 为 controller-in-the-loop 与 RL wrapper 铺路

### 4.3 仍保留的兼容主线

- 数据集：`configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- 训练：`configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`

用途：

- 与历史主线对照
- 检验新增质量上下文和单步分支的增益

## 5. 当前关键工程事实

当前架构上最重要的事实有三条：

1. `KF / ESKF` 初值已改为严格因果 warm-start，不再依赖未来 DVL 观测。
2. 共享状态维 `X/Y` scaler 已收口到单一统计量，保证 rollout 数值语义一致。
3. 项目已经从单一多步主线扩展为“多步主线 + 单步实验线”并行架构。

这意味着当前讨论“状态转移求解器”已经有了真实工程支撑，而不只是理论规划。

## 6. 评估与证据链

当前正式评估由 `src/uwnav_dynamics/eval/evaluate.py` 统一生成 artifact。

推荐重点关注：

- `metrics.yaml`
- `rmse_by_horizon*.csv`
- `mae_by_horizon*.csv`
- `component_metrics*.csv`
- `pred_samples*.npz`
- `pred_context.npz`

当前推荐的关键指标：

- `rmse_global / mae_global`
- `rmse_global_masked / mae_global_masked`
- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- 有效步数与非有限值统计

## 7. 当前边界

当前系统还没有完成以下内容：

- 严格统一的 `step(u_t, dt)` 求解器接口
- 已验证的闭环控制稳定性结论
- 高保真物理仿真替代能力

因此，当前最合适的表述仍然是：

- 已具备面向状态转移求解器的训练与评估骨架
- 正在从多步轨迹预测器向一步动力学算子收口
- 可以为控制与 RL 提供实验入口，但尚未形成闭环完成证明

## 8. 推荐阅读入口

1. `README.md`
2. `docs/handover_guide.md`
3. `docs/project_status.md`
4. `docs/design/transition_solver_phase1_upgrade.md`
5. `docs/handover_kf_training_server_v2.md`
6. `docs/math/main.tex`
