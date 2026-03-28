# 项目架构说明

## 1. 当前系统目标

当前仓库的目标不是完整闭式水动力建模，也不是直接交付闭环控制系统。

当前系统目标是：

- 从多源异步观测中构建统一训练样本
- 学习短时状态转移规律
- 通过长期递推筛选可用的学习型状态转移求解器
- 为后续仿真、控制和论文图包提供稳定证据链

## 2. 当前数据链

默认处理的异步数据源：

- PWM：100 Hz，控制输入
- IMU：100 Hz，高频机体观测
- DVL：10 Hz，低频速度观测
- Power：5 Hz，辅助特征

当前主时间轴固定为 `100 Hz`。

## 3. 当前主流程

```text
PWM / IMU / DVL / Power raw logs
    ↓
IMU preprocess
    ↓
multi-rate alignment
    ↓
KF / ESKF fusion
    ↓
sliding-window dataset
    ↓
S1Predictor train
    ↓
offline eval + long-replay figures
```

## 4. 预处理链职责

### IMU

当前 IMU 主链：

```text
raw imu
-> transform
-> gravity compensation
-> bias correction
-> filter
```

关键模块：

- `src/uwnav_dynamics/preprocess/imu/transform.py`
- `src/uwnav_dynamics/preprocess/imu/gravity.py`
- `src/uwnav_dynamics/preprocess/imu/pipeline.py`

### 对齐

多源时间对齐由：

- `src/uwnav_dynamics/preprocess/align/aligner.py`

负责把异步源收口到统一主时间轴。

### 融合

当前新增的训练导向融合层：

- `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`
- `src/uwnav_dynamics/preprocess/fusion/cli_fuse_train_base.py`

职责是：

- 用 DVL 速度约束 IMU 链路漂移与高频噪声
- 用 IMU 高频传播补足速度频率
- 统一输出 `AccKf / GyroKf / VelKf / RollPitchYaw`

## 5. 数据集契约

当前主数据集为 `kf_ctx_v2`。

### 输入

`29` 维：

- `PWM 8`
- `AccKf 3`
- `GyroKf 3`
- `VelKf 3`
- `AttCtx 4`
  - `RollKf`
  - `PitchKf`
  - `SinYawKf`
  - `CosYawKf`
- `Power 8`

### 目标

`9` 维：

- `AccKf 3`
- `GyroKf 3`
- `VelKf 3`

### 监督语义

- `VelKf` 已作为 dense target 使用
- `labels.npz["dvl_mask"]` 在该主线上可被写成全真
- 因此当前 masked metrics 不再等价于旧阶段“仅 DVL 命中时刻监督”

## 6. 模型与 rollout

当前主模型仍是：

- `src/uwnav_dynamics/models/nets/s1_predictor.py`

当前推荐设置：

- backbone：`LSTM`
- head：`grouped`
- 主输出：未来状态增量 `dY`
- rollout：`y_hat = y0 + cumsum(dY)`

可选 blocks：

- `thruster_lag`
- `hydro_ssm`
- `damping`
- `uncertainty`

当前主线不建议默认打开 `uncertainty` block 来吞掉状态误差。

## 7. 训练目标

当前训练主线包含两层目标：

### 优化目标

- `transition_balance`
- 强调 `acc / gyro / vel` 组约束
- 强调 tail horizon
- 同时约束 state 与 delta

### 选模目标

训练期不再只盯 `val_loss`。

当前支持：

- `val_loss`
- `val_transition_score`

其中 `val_transition_score` 更偏向长期状态误差和尾部误差，用于 best ckpt、scheduler 和 early stopping。

## 8. 评估与图包

当前正式评估仍由：

- `src/uwnav_dynamics/eval/evaluate.py`

负责生成：

- `metrics.yaml`
- `rmse_by_horizon*.csv`
- `mae_by_horizon*.csv`
- `component_metrics*.csv`
- `pred_samples*.npz`
- `pred_context.npz`

当前推荐图包聚焦：

- 训练过程对比
- horizon compare
- long replay prediction trace
- long replay error growth
- top2 summary dashboard

所有正式图片应：

- 无标题
- 保留坐标轴标题和图例
- 统一字体
- 固定画布比例

## 9. 当前推荐入口

优先阅读：

1. `README.md`
2. `docs/handover_guide.md`
3. `docs/handover_kf_training_server_v2.md`
4. `docs/快捷命令行.md`
5. `docs/design/kf_fusion_preprocess_training_v2.md`
