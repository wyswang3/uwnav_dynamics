# 项目架构说明（uwnav_dynamics）

## 1. 系统目标与非目标

### 1.1 目标

本项目面向水下机器人动力学辨识与控制前预测建模，当前工程目标是：

- 将 PWM、IMU、DVL、Power 多源日志整理为统一训练样本
- 学习短时多步动力学响应，而不是回归完整水动力参数
- 为未来 MPC / 仿真 / 闭环控制提供可复现的预测模型接口
- 保持工程链路可测试、可追溯、可用于科研审查

### 1.2 非目标

当前仓库不以以下事项为交付目标：

- 完整 6-DOF 刚体-流体动力学闭式建模
- 闭环控制器本体实现
- 实机部署系统集成
- 最终版 mask-aware 稀疏监督训练体系

## 2. 传感器与采样频率

当前默认处理 4 类异步数据源：

- PWM：100 Hz，推进器输入命令
- IMU：100 Hz，机体加速度、角速度、姿态相关量
- DVL：10 Hz，速度相关观测，天然稀疏
- Power：5 Hz，电机功率辅助特征

核心事实：

- 训练主时间轴是 100 Hz
- DVL 不被强行插值为“密集真值”
- Power 当前作为低频辅助特征接入

## 3. 多频数据对齐策略

多频对齐由 [aligner.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/align/aligner.py) 驱动。

当前语义：

- PWM：hold-last 映射到主时间轴，作为控制输入
- IMU：聚合/对齐到主时间轴，作为高频主观测
- DVL：按命中位置 attach 到主轴，未命中时保留缺失语义
- Power：低频 hold-last 到主轴，用于辅助特征

PR6 已修复：

- 浮点时间边界导致的 `power_mask` 漏命中
- `pytest -q` 收集失败与最小冒烟回归缺失

## 4. 数据 pipeline

当前数据流大致为：

```text
raw logs
-> preprocess
-> aligned base table
-> sliding-window dataset
-> split/scaler artifacts
-> train / eval
```

关键阶段：

1. `preprocess`
   - 入口：`src/uwnav_dynamics/preprocess/*`
   - 负责原始 CSV 的字段、时间戳、单位与基础 QA 处理
2. `align`
   - 入口：`src/uwnav_dynamics/preprocess/align/cli_align.py`
   - 产物：对齐后的基础训练表
3. `build_dataset`
   - 入口：`src/uwnav_dynamics/preprocess/build_dataset.py`
   - 产物：`features.npz`、`labels.npz`、`meta.yaml`
4. `train data pipeline`
   - 入口：`src/uwnav_dynamics/train/data_pipeline.py`
   - 产物：`split_indices.npz`、`x_scaler.npz`、`y_scaler.npz`
   - 运行时 batch：`(X, Y, target_mask)`

## 5. 模型 pipeline

当前主模型为 [s1_predictor.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/models/nets/s1_predictor.py)。

主干结构：

- Backbone：LSTM encoder
- Head：输出未来增量 `dY`
- Uncertainty：输出对角 `logvar`
- Rollout：`y_hat = y0 + cumsum(dY)`

当前目标维：

- `dout = 9`
- 语义为 `Acc 3 + Gyro 3 + Vel_state 3`

当前输入维：

- `din = 25`
- 语义为 `PWM 8 + IMU 6 + Vel_state 3 + Power 8`

## 6. 可选物理先验 blocks

当前模型支持按配置开关启用的 blocks：

- `ThrusterLag`
  - 对推进器输入做 deadzone / saturation / lag 建模
- `HydroSSMCell`
  - 以隐状态形式吸收流体记忆效应
- `DampingHead`
  - 显式输出速度相关阻尼项
- `UncertaintyHead`
  - 生成异方差不确定度

这些模块的配置对象在 PR1 中已冻结，以避免 train / eval 之间的结构漂移。

## 7. rollout 语义

当前 rollout 采用：

```text
y0 = x_last_state
y_hat = y0 + cumsum(dY)
```

注意：

- 当前实现已明确 train / eval 必须共享同一配置契约
- PR4 后 rollout layout 已明确分成两层：
  - execution layout contract：唯一执行真源是 `cfg_model.y_in_idx`
  - semantic output layout contract：`Acc / Gyro / Vel` 组件标签与分组语义

当前系统级数据流为：

```text
train yaml
-> cfg_model.y_in_idx
-> execution_layout helper
-> train rollout / eval rollout
-> metrics.yaml.layout.semantic
-> viz grouping / plotting
```

PR5 第一阶段之后，监督有效性链路补充为：

```text
aligned sparse DVL availability
-> labels.npz["dvl_mask"]
-> supervision_mask helper
-> target_mask(batch)
-> masked loss / masked eval metrics
-> rmse_by_horizon_masked.csv / mae_by_horizon_masked.csv
-> horizon/model-compare masked plots
```

这里需要明确区分：

- execution layout contract
  - 只负责 rollout 的 `y0` 提取
  - 唯一执行真源仍是 `cfg_model.y_in_idx`
- semantic output layout contract
  - 只负责 `acc / gyro / vel` 语义分组
  - 供 `target_mask` 构造、masked metric 聚合与 viz 解释使用
- runtime mask contract
  - 训练 / 评估运行时唯一 mask 真源是 batch 内的 `target_mask`
  - `meta.yaml` 与 `metrics.yaml` 只记录，不裁决 mask 执行

## 8. 配置契约

PR1 已建立当前配置系统的核心边界：

- [train/config.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/config.py) 是 canonical parser
- eval 侧通过 [eval/config.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/config.py) 直接复用训练解析结果
- 纯配置 dataclass 冻结，runtime override 使用 `replace()`
- `resolved_train.yaml` 记录最终执行配置与关键 artifact 路径

配置设计的详细说明见 [config_contract.md](/home/wys/uwnav_dynamics/docs/config_contract.md)。

## 9. artifact 契约

一次训练 run 的关键产物位于：

```text
run_dir = run.out_dir / run.variant
```

典型产物包括：

- `resolved_train.yaml`
- `best.pth`
- `last.pth`
- `split_indices.npz`
- `scalers/x_scaler.npz`
- `scalers/y_scaler.npz`
- `eval_<split>/metrics.yaml`
- `eval_<split>/pred_samples.npz`

## 10. 当前升级状态

已完成：

- `PR6`：对齐边界修复与 pytest 冒烟恢复
- `PR1`：配置契约收口与实验追溯增强
- `PR2`：split / scaler 单一真源
- `PR3`：eval-viz 解耦
- `PR4`：execution / semantic layout contract 收口
- `PR5`：mask-aware 训练评估第一阶段（dense/masked 并行 horizon artifact）

待推进：

- `PR5`：mask-aware 训练评估

## 11. 当前限制

当前工程仍存在以下限制：

- sample-level masked visualization 尚未接入
- 更细粒度的稀疏监督类型仍未完全扩展到所有图型
- 理论文档与工程实现仍需持续保持一致
- 未来控制接口仍停留在模型输出可接入阶段，尚未落正式 MPC 实现

已解决的相关限制：

- train / eval / viz 对状态布局的解释已统一，不再由多处 rollout 硬编码副本分别维护
- DVL velocity 稀疏监督已进入训练 loss 与 eval masked metrics 主路径

## 12. 阅读建议

建议阅读顺序：

1. [README.md](/home/wys/uwnav_dynamics/README.md)
2. [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
3. [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)
4. [modeling_roadmap.md](/home/wys/uwnav_dynamics/docs/modeling_roadmap.md)
5. [docs/math/README.md](/home/wys/uwnav_dynamics/docs/math/README.md)
