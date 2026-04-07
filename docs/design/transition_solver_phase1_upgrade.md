# 状态转移求解器升级方案（Phase 1）

更新时间：2026-04-07

## 1. 阶段目标

当前阶段的目标不是直接把项目改造成完整控制系统或 RL 环境，
而是先把训练链中最容易破坏“状态转移语义”的两个基础问题收口：

1. 保证状态代理量生成过程满足严格因果约束
2. 保证 rollout 初值 `y0` 与训练目标 `Y` 处于同一 z-score 语义空间

完成后，项目应从“可做长期拟合实验”推进到：

**可以更可信地作为短时状态转移求解器候选进行下一阶段升级**

## 2. 现有问题

### 2.1 融合层初值存在未来观测泄漏

代码位置：

- [kf_eskf.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/fusion/kf_eskf.py)

问题表现：

- 旧实现会从整段序列中寻找第一条有效 DVL 观测作为初始速度
- 若序列起点没有 DVL、后面才出现 DVL，则未来观测会被回填到 `t0`

工程后果：

- 训练基础表不再满足严格因果
- 任何基于该表的 rollout 评估都可能乐观偏置

### 2.2 共享状态维的 scaler 语义错位

代码位置：

- [data_pipeline.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/data_pipeline.py)
- [execution_layout.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/models/utils/execution_layout.py)
- [run_train.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/run_train.py)

问题表现：

- `X` 与 `Y` 分别拟合 `x_scaler / y_scaler`
- 但 rollout 初值 `y0` 直接从 `X[:, -1, y_in_idx]` 中抽取
- 若共享状态维在两套 scaler 中统计量不同，则训练时实际在比较：
  - `y0`：`x_scaler` 空间
  - `Y`：`y_scaler` 空间

工程后果：

- transition loss 的数学语义被破坏
- `delta-cumsum` rollout 会引入非物理的尺度扭曲

## 3. Phase 1 最小实现

### 3.1 融合层因果 warm-start

实现原则：

- 允许在 `t0` 恰好有 DVL 时使用该时刻观测 warm-start
- 若 `t0` 没有 DVL，则只能用零速度或其他先验，不允许读取未来 DVL

本次实现：

- `_select_initial_velocity()` 只读取序列第 `0` 个时刻
- 不再从整段序列扫描“第一条有效 DVL”

### 3.2 共享状态维 scaler 对齐

实现原则：

- 不改 train / eval 主接口
- 不改现有 `x_scaler.npz / y_scaler.npz` artifact 命名
- 只对共享状态维收口单一统计量

本次实现：

- 在 `prepare_train_data()` 中根据 dataset artifact 的 `input_cols / target_cols`
  恢复共享状态列映射
- 把 `x_scaler` 中对应共享状态维的 `mean/std` 直接对齐到 `y_scaler`
- 该修复既作用于新拟合 scaler，也作用于复用旧 run artifact 的场景

## 4. 当前阶段完成后得到什么

Phase 1 完成后，可以确认：

- 训练基础表不再因初始化而偷看未来 DVL
- rollout 初值与目标监督回到同一数值语义空间
- `val_transition_score` 与长期 rollout 指标的解释前提更稳

但仍然不能声称：

- 已经得到严格的一步状态转移算子
- 已经证明闭环控制可用
- 已经具备强化学习环境级别的系统接口

## 5. 运行命令

### 阶段目标

验证 Phase 1 修复没有破坏现有 KF 主线训练契约。

### 最小实现

- 运行融合与数据管线相关测试
- 重建融合基础表
- 重建 KF 数据集
- 进行单卡 smoke

### 运行命令

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_train_data_pipeline_nan_sanitization.py \
  tests/test_transition_balance_design.py \
  tests/test_kf_ctx_training_config_v2.py

python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml

python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml

python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

### 验证标准

- `AccKf / GyroKf / VelKf` 列保持全 finite
- `labels.npz["dvl_mask"]` 仍为 dense supervision 全真
- 训练 run 目录中的 `x_scaler / y_scaler` 对共享状态维给出一致统计量
- `train_summary.yaml` 与 `eval_test/metrics.yaml` 正常产出

### 风险点

- 这一步只修基础契约，不保证指标立即显著提升
- 若模型本体仍保持“历史窗 -> 固定未来块输出”，长期自由递推能力仍有限
- 若后续引入新状态维或质量字段，需要同步更新共享状态映射逻辑

## 6. Phase 2 建议

Phase 2 的主目标应切到“更接近 `x_{t+1}=f(x_t,u_t,c_t)` 的一步状态转移器”：

1. 把 `HasDvlUpdate / DtSinceDvl_s / VelKfVar*` 等质量字段纳入输入
2. 把主任务从固定 `pred_len=10` 轨迹块输出收口成一步状态转移
3. 在损失中加入观测一致性项：
   - DVL velocity consistency
   - IMU accel / gyro consistency
   - 可选 power / thruster consistency
4. 增加最小进程内 `step(u_t, dt)` 接口，支撑 controller replay 与 RL wrapper

### 当前已新增的 Phase 2 入口配置

为避免“一步状态转移”停留在设计层，当前仓库已新增：

- 数据集：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
- 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`

该分支保持：

- `din = 34`
- `dout = 9`
- `hist_len = 100`
- `pred_len = 1`

其用途是：

- 与 `kf_ctx_v2`、`quality_v3` 多步块输出主线做对照
- 验证“质量上下文 + 单步状态转移”是否更利于后续闭环与 RL 接口
