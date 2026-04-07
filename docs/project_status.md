# 项目当前状态

更新时间：2026-04-07

## 1. 当前阶段

项目当前处于：

**`状态转移求解器升级 Phase 1 已完成；Phase 2 已新增 quality-context 与 single-step 两条实验分支`**

当前主目标是：

- 保持现有 KF 状态代理量主线
- 把训练链真正推进到“可解释、可递推、可继续向控制与 RL 接口演进”的状态

## 2. 当前已经完成的内容

### 数据预处理

- IMU `transform -> gravity -> bias -> filter` 链稳定
- 多频对齐主链稳定
- `KF / ESKF` 融合模块已落地
- Phase 1 已修复融合初值的未来 DVL 泄漏风险：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

### 数据集与 scaler 契约

- 当前主数据集契约仍是：
  - `configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- Phase 2 预备配置已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
- Phase 2 单步分支已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
- 输入 `29` 维：
  - `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8`
- quality v3 输入 `34` 维：
  - `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + DvlCtx 5 + Power 8`
- 输出 `9` 维：
  - `AccKf 3 + GyroKf 3 + VelKf 3`
- `VelKf` 仍按 dense supervision 进入训练
- Phase 1 已把共享状态维的 `X/Y` scaler 语义收口到单一真源：
  - `src/uwnav_dynamics/train/data_pipeline.py`
- `DtSinceDvl_s` 现已保证为有限值，可作为模型输入上下文使用：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

### 训练与评估

- `transition_balance` 已接入主训练路径
- grouped head 已接入 `S1Predictor`
- 训练期监控仍支持：
  - `val_loss`
  - `val_transition_score`
- 离线评估仍保留：
  - `rmse_global / mae_global`
  - `final_step`
  - `rollout_growth`
  - `tail_error`
  - `worst_abs_bias`

## 3. 当前最重要的工程事实

当前要特别明确三件事：

1. 现在的训练链在“因果性”和“rollout 数值语义”上比上一轮更可靠。
2. 当前默认主线模型主体仍是“历史窗 -> 固定未来块输出”，还不是严格的一步状态转移器。
3. `quality v3` 与 `quality_step_v1` 两条配置分支已经准备好，可用于验证：
   - 状态代理量 + 质量上下文
   - 单步状态转移
4. 当前离线指标仍然只代表控制前筛查，不代表闭环可用性证明。

## 4. 当前剩余主阻塞

Phase 1 收口后，当前剩余的主阻塞只剩一条：

1. 模型形式仍偏“轨迹预测器”  
   当前默认主线仍是 `hist_len=100, pred_len=10` 的“历史窗 -> 固定未来块输出”，
   还不是更接近系统状态方程形式的 `x_{t+1}=f(x_t,u_t,c_t)` 一步状态转移模型。

## 5. 当前可直接复用的产物

- 融合基础表：
  - `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- KF 数据集：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/`
- 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- quality-context 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- single-step 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`
- 8 卡矩阵：
  - `configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml`
- 升级设计文档：
  - `docs/design/transition_solver_phase1_upgrade.md`

## 6. 当前边界与风险

### 已知边界

- KF 输出仍应表述为“低噪声状态代理量”，不是物理真值
- 当前主输出仍是 `9` 维，不含姿态角主监督
- 当前还没有进程内 `step(u_t, dt)` 形式的统一求解器接口

### 当前风险

- 8 卡矩阵尚未按 Phase 1 修复后的契约正式重跑
- 新图包尚未用修复后的 run 重新生成
- 若模型主体不进一步收口为一步转移器，长期自由递推能力仍可能不足

## 7. 当前推荐动作

当前建议严格按这个顺序推进：

1. 用 Phase 1 修复后的代码重建融合基础表与数据集
2. 做单卡 smoke，确认 `train_summary.yaml / eval_test/metrics.yaml` 正常
3. 对比 v2、quality v3、quality step v1 三条单卡 smoke
4. 基于对比结果选定“一步转移”是否转正为主线
5. 再做 8 卡矩阵与 top2 图包
6. 最后再接最小 controller replay 或 RL wrapper

## 8. 当前不建议的做法

- 不要把 Phase 1 的完成表述成“闭环控制已可用”
- 不要直接恢复旧的 controller 图壳层作为主入口
- 不要在还未形成一步状态转移契约前同时大改模型结构和闭环框架
