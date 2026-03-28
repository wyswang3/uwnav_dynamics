# KF 融合预处理与长期拟合训练设计 V2

更新时间：2026-03-28  
当前工作分支：`feature/kf-preprocess-training-v1`

## 1. 阶段目标

本阶段的目标不是继续堆 controller / solver validation 壳层，而是先把训练输入与监督目标改对：

- 用因果 KF / ESKF 融合层统一多源异步观测
- 用 DVL 速度观测抑制 IMU 加速度链路的高频噪声与偏置漂移
- 用高频 IMU 补足 DVL 速度的时间分辨率
- 形成统一 100 Hz 时间轴上的低噪声状态代理量
- 让网络重点学习“控制信号如何驱动系统运动”
- 把长期递推贴近观测值作为主评价标准

这里要特别强调：

- KF 输出更准确的说法是“低噪声状态代理量”，不是“绝对真值”
- 第一轮网络重训优先保证 `acc / gyro / vel` 的长期拟合能力
- 姿态角先作为上下文输入与绘图审查对象进入训练链，不立即打破当前 9 维主输出契约

## 2. 当前问题与切换原因

当前主链路已经具备这些可靠基础：

- IMU `RFU (X右, Y前, Z上) -> FRD` 坐标变换  
  代码：`src/uwnav_dynamics/preprocess/imu/transform.py`
- 重力补偿与 bias / filter  
  代码：`src/uwnav_dynamics/preprocess/imu/gravity.py`、`src/uwnav_dynamics/preprocess/imu/pipeline.py`
- S1Predictor + grouped head + transition-balance loss  
  代码：`src/uwnav_dynamics/models/nets/s1_predictor.py`、`src/uwnav_dynamics/models/losses/state_transition.py`
- 8 卡并发矩阵调度  
  代码：`src/uwnav_dynamics/cli/train_matrix.py`

当前真正限制长期拟合的问题仍然是状态代理量过粗：

- IMU / PWM / DVL / Power 的统一主要靠插值、hold-last 与 sparse attach
- 速度状态代理量仍然过度依赖前向填充 DVL
- 网络被迫同时承担“补时间轴、去噪、学动力学”三件事

这会直接损害长期 rollout：

- 速度端容易漂
- 加速度端容易放大噪声
- 控制驱动和观测噪声纠缠在一起

## 3. 预处理 V2：因果 KF / ESKF 融合层

### 3.1 设计原则

- 保留现有 IMU `transform -> gravity -> bias -> filter` 链路
- 新增一层因果 KF / ESKF 融合，不使用看未来的 smoother
- 主时间轴固定为 `100 Hz`
- DVL 坐标系先按“与体坐标近似重合”处理，保留后续小外参修正入口

### 3.2 推荐状态与观测

推荐把融合层拆成“传播状态 + 同步输出代理量”两层理解，而不是强行把所有量都塞进一个状态向量：

- 传播核心：
  - `v_body`：体坐标速度
  - `attitude`：`roll / pitch / yaw`
  - `b_a / b_g`：IMU 加速度与角速度慢变偏置
- 高频输入：
  - `a_lin_body_raw`
  - `gyro_body_raw`
  - `dt`
- 低频更新：
  - `dvl_v_body`
- 同步输出代理量：
  - `AccKf*_body_mps2`
  - `GyroKf*_body_rad_s`
  - `VelKf*_body_mps`
  - `RollKf_rad / PitchKf_rad / YawKf_rad`

关键解释：

- 速度由 IMU 传播、DVL 更新统一到 100 Hz
- 加速度代理量不是“单独积分得到”，而是利用 bias 校正和速度更新对 IMU 链路做约束后的同步输出
- 角速度代理量沿用 IMU 高频优势，但在融合层中与 bias / attitude 一起做一致性约束

### 3.3 统一后的基础表列建议

新的基础表建议至少包含：

- 控制与辅助：
  - `ch1_cmd ... ch8_cmd`
  - `P0_W ... P7_W`
- 融合后的状态代理量：
  - `AccKfX_body_mps2`
  - `AccKfY_body_mps2`
  - `AccKfZ_body_mps2`
  - `GyroKfX_body_rad_s`
  - `GyroKfY_body_rad_s`
  - `GyroKfZ_body_rad_s`
  - `VelKfX_body_mps`
  - `VelKfY_body_mps`
  - `VelKfZ_body_mps`
  - `RollKf_rad`
  - `PitchKf_rad`
  - `YawKf_rad`
- 可选质量字段：
  - `HasDvlUpdate`
  - `DtSinceDvl_s`
  - `VelKfVarX / VelKfVarY / VelKfVarZ`

第一轮训练先不把质量字段送进网络，先把融合后的主状态代理量做对。

### 3.4 当前已落地实现

当前仓库已经补上了融合实现与命令行入口：

- 融合模块：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`
- CLI：
  - `src/uwnav_dynamics/preprocess/fusion/cli_fuse_train_base.py`
- 配置：
  - `configs/fusion/pooltest02_kf_eskf_v2.yaml`

当前本地已经实际生成过：

- 融合基础表：
  - `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- 新数据集：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2`

## 4. 网络设计 V2

### 4.1 为什么不直接把姿态角并入主输出

当前训练 / 评估 / 画图主链路的 canonical 输出布局仍然是 9 维：

- `Acc 3 + Gyro 3 + Vel 3`

对应真源代码是：

- `src/uwnav_dynamics/models/utils/semantic_output_layout.py`

如果现在直接把主输出扩成 12 维，会连带修改：

- semantic layout 契约
- grouped head 语义分组
- transition-balance 组权重
- eval artifact 与 compare plot 聚合

这会把“先把数据代理量改对”的任务扩大成“训练 / 评估协议整体重构”。

因此 V2 的最小改动路线是：

- 姿态角进入输入，作为动力学上下文
- 主输出先保持 `acc / gyro / vel` 9 维
- 等第一轮 8 卡结果证明长期拟合改善后，再决定是否把姿态角提升为主输出

### 4.2 输入输出契约

第一轮推荐的输入布局：

`PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8 = 29`

其中：

- `AttCtx 4 = RollKf_rad + PitchKf_rad + SinYawKf + CosYawKf`
- yaw 用 `sin/cos` 是为了避开 `(-pi, pi]` wrap 带来的不连续

第一轮推荐的目标布局：

`AccKf 3 + GyroKf 3 + VelKf 3 = 9`

同时，`VelKf` 在数据集构建阶段按 dense target 监督：

- `labels.npz["dvl_mask"]` 会被置成全真
- 原始 `dvl_mask` 仍保留在基础表和 `features` 侧，用于记录真实 DVL 观测新鲜度
- 这样训练不会把已经融合成 dense 代理量的速度监督重新降回稀疏

### 4.3 模型结构建议

第一轮继续基于 `S1Predictor`：

- backbone：`LSTM`
- 主 head：`grouped`
- rollout：`y_hat = y0 + cumsum(dY)`
- 默认启用：
  - `head_mode=grouped`
  - `transition_balance`
  - `train.metric=val_transition_score`
- 可比较模块：
  - `thruster_lag`
  - `hydro_ssm`

这轮不建议默认打开：

- `uncertainty` block
- 更大规模结构改造
- 12 维主输出协议切换

## 5. 第一轮 8 卡训练矩阵

### 阶段目标

验证“更好的状态代理量 + 更合理的损失与结构归纳偏置”是否能明显改善长期拟合。

### 最小实现

固定：

- 融合后的 KF 状态代理量
- 输入维 `29`
- 输出维 `9`
- `pred_len=10`

比较 4 组思路，每组 2 个 seed：

1. `joint + nll`
2. `grouped + transition_balance`
3. `grouped + transition_balance + stronger tail weighting`
4. `grouped + transition_balance + thruster/hydro blocks`

### 具体 8 组

1. `kfctx_a0_joint_nll_seed8`
2. `kfctx_a0_joint_nll_seed9`
3. `kfctx_b0_grouped_tb_seed8`
4. `kfctx_b0_grouped_tb_seed9`
5. `kfctx_b2_grouped_tb_longtail_seed8`
6. `kfctx_b2_grouped_tb_longtail_seed9`
7. `kfctx_b4_grouped_tb_blocks_seed8`
8. `kfctx_b4_grouped_tb_blocks_seed9`

### 评价标准

主评价标准不再是“图看起来像不像”，而是：

- `rmse_global`
- `mae_global`
- `rmse_global_masked`
- `mae_global_masked`
- `final_step.group_rmse.acc`
- `final_step.group_rmse.gyro`
- `final_step.group_rmse.vel`
- `rollout_growth`
- `tail_error.p95 / p99`
- `worst_abs_bias`

需要明确一个当前实现边界：

- 这些指标仍主要用于 run 级后验筛选
- 当前训练代码已经支持 `val_transition_score` 作为 epoch 级监控信号
- 但 `pred_len` 当前仍为 `10`，长期能力仍需要依赖递推评估与 top2 图包继续确认

## 6. 图包设计思路

### 6.1 目标

图包的目标是给最终模型提供“训练过程、模型比较、长期拟合”的完整证据链，而不是只放单条好看的曲线。

### 6.2 图包组成

推荐最少输出 5 类图：

1. 训练过程图  
   作用：比较多个 run 的 `train/val loss` 收敛质量和是否出现过拟合。

2. Horizon Compare  
   作用：比较多个 run 的 `RMSE / MAE` 随 horizon 的变化，重点看 `acc / gyro / vel` 三组。

3. Long Replay Prediction Trace  
   作用：在真实控制回放下，对比 `Observed vs Predicted` 的长期拟合。

4. Long Replay Error Growth  
   作用：展示 `||e_acc|| / ||e_gyro|| / ||e_vel||` 随时间增长的速度。

5. Top2 Summary Dashboard  
   作用：对最终 2 个候选汇总 `final_step / rollout_growth / tail_error / worst_abs_bias`。

### 6.3 绘图约束

所有这轮图都遵循以下约束：

- 不要图表标题
- 只保留坐标轴标题、图例和必要单位
- 字体统一使用 `Times New Roman`
- 复用 `src/uwnav_dynamics/viz/style/sci_style.py`
- 固定导出长宽比，推荐 `4:3`
- 坐标轴自适应，不手工钉死数值范围
- 图例放在轴内不遮挡主曲线的位置
- 颜色沿用仓库已有语义配色

### 6.4 配套说明文档

每组图片目录下应同时写一个 `figure_notes.md`，至少说明：

- 图片证明的对象是什么
- 图片对应的是哪一组 run / split / replay 窗口
- 为什么这张图能支持“长期拟合更稳”的判断
- 图中是否存在失败片段、异常漂移或解释边界

## 7. 服务器迁移时的边界

### 运行命令

当前已经可以直接使用的命令：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_semantic_output_layout_kf_cols.py \
  tests/test_transition_balance_design.py \
  tests/test_train_matrix_launcher.py \
  tests/test_kf_ctx_training_config_v2.py
```

服务器上的最短命令链应为：

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml
```

### 风险点

- 第一轮主输出仍是 9 维，因此姿态角目前只进输入和图包，不直接参加主 rollout loss
- 如果第一轮已经证明长期拟合明显改善，再考虑 12 维主输出重构
- 当前融合实现是工程化的因果 KF / ESKF 近似，不应表述成高保真物理真值滤波器

## 8. 当前结论

当前最合理的推进顺序是：

1. 复用现有 CLI 生成 `train_base_kf_v2.csv`
2. 构建新的 29 维数据集
3. 跑 8 卡训练矩阵
4. 只对 top2 产出长期 replay 图包

本阶段先把“数据代理量和训练目标改对”，再谈最终 solver 证明。
