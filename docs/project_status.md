# 项目当前状态

更新时间：2026-03-28

## 1. 当前阶段

项目当前处于：

**`KF / ESKF 融合预处理已落地，新的 KF 状态代理量训练链已接通，但正式重训前还需先修三处关键阻塞`**

当前主目标是得到一个能够长期递推、误差处于允许范围内的状态转移求解器。

## 2. 当前已经完成的内容

### 数据预处理

- IMU 坐标变换、去重力、bias/filter 链已经稳定。
- 多频对齐主链已经稳定。
- `KF / ESKF` 融合模块已落地：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`
  - `src/uwnav_dynamics/preprocess/fusion/cli_fuse_train_base.py`
- 当前融合基础表路径：
  - `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`

### 数据集契约

- 当前主数据集契约：
  - `configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- 输入为 `29` 维：
  - `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8`
- 目标为 `9` 维：
  - `AccKf 3 + GyroKf 3 + VelKf 3`
- `VelKf` 已按 dense supervision 进入训练

### 训练链

- `transition_balance` 损失已接入训练主路径
- grouped head 已接入 `S1Predictor`
- 训练期监控指标已支持：
  - `val_loss`
  - `val_transition_score`
- 当前推荐训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- 当前推荐 8 卡矩阵：
  - `configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml`

## 3. 当前最重要的工程事实

当前要特别明确三件事：

1. 新的状态代理量链已经落地，但还没有完成新一轮正式重训。
2. 当前训练代码已经能用 `val_transition_score` 选 best ckpt，不再只能按 `val_loss`。
3. 当前入口层不再推荐继续沿用旧的 controller / transition validation 叙事。

## 3.1 当前暂停重训的原因

最近一次训练链审查确认，当前还有三处关键问题需要先修：

1. 融合初始化泄漏  
   `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py` 当前会用整段序列里第一条有效 DVL 初始化速度，
   这不满足严格因果约束。

2. 共享状态维 scaler 语义错位  
   `src/uwnav_dynamics/train/data_pipeline.py` 当前仍分别拟合 `x_scaler / y_scaler`，
   但 rollout 训练会把从 `X` 取出的 `y0` 与 `Y` 放到同一状态转移损失中比较。

3. 模型形式仍偏“轨迹预测器”  
   当前 `hist_len=100, pred_len=10`，模型还是“历史窗 -> 固定未来块输出”，
   还不是更接近系统状态方程形式的 `x_{t+1}=f(x_t,u_t)` 一步状态转移模型。

## 4. 当前可直接复用的产物

- 融合基础表：
  - `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- KF 数据集：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/`
- 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- 8 卡矩阵：
  - `configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml`
- 服务器迁移文档：
  - `docs/handover_kf_training_server_v2.md`

## 5. 当前风险与边界

### 已知边界

- KF 输出应表述为“低噪声状态代理量”，不是绝对真值。
- 当前主输出仍是 `9` 维，不包含姿态角主监督。
- 当前数据集 `pred_len` 仍是 `10`，长期能力仍主要依赖递推稳定性和后验筛选，而不是一次性拉长主监督 horizon。

### 当前风险

- 8 卡矩阵尚未正式重跑，当前还没有新的 top2。
- 新图包还没按新模型重生成。
- 若服务器环境缺少字体，论文图风格会回退。

## 6. 当前推荐动作

当前建议严格按这个顺序推进：

1. 修复融合初始化泄漏
2. 修复共享状态维 scaler 契约
3. 收口一步状态转移训练配置
4. 再做融合基础表生成、数据集构建和单卡 smoke
5. 最后才进入 8 卡矩阵与 top2 图包

## 7. 当前不建议再作为入口层保留的叙事

以下内容如果仍然存在，只作为历史参考：

- round4 / round5 的 controller 相关叙述
- 旧的 transition validation 图包结论
- 只用 `val_loss` 判断模型优劣
- 旧的前向填充速度代理量训练口径
