# uwnav_dynamics

面向水下机器人的科研级数据驱动状态转移建模仓库。

当前主线不是恢复完整精确水动力学，也不是直接交付闭环控制系统，而是把离线训练得到的学习型动力学模型推进到“可验证的状态转移求解器”阶段，使其能够为控制仿真、controller-in-the-loop 验证和后续强化学习环境提供短时动态预测能力。

## 项目定位

本仓库当前聚焦三个问题：

- 如何把异步、多频、含噪传感器链路收口为统一的因果状态代理量。
- 如何训练一个对控制输入敏感、可递推、可评估的短时状态转移模型。
- 如何在不依赖高保真物理真值模型的前提下，为控制与 RL 提供工程上可用的动态内核。

当前对模型的正确定位是：

- 控制器内部短时状态转移模型
- 经验型仿真内核
- 面向 model-based RL 的可递推动态预测器

当前不应把本项目表述为：

- 高保真水下物理仿真器
- 完整闭环控制系统
- 已验证可实机部署的安全控制栈

## 当前主线

当前工程主线已经切到“因果状态代理量训练”：

```text
Raw Logs
-> IMU preprocess
-> Multi-rate alignment
-> KF / ESKF fusion
-> Proxy-state dataset build
-> S1Predictor training
-> Offline transition evaluation
-> Controller / simulator readiness analysis
```

默认主时间轴为 `100 Hz`，主要数据源为：

- PWM: `100 Hz`
- IMU: `100 Hz`
- DVL: `10 Hz`
- Power: `5 Hz`

## 当前数据与模型契约

### 代理状态定义

当前主监督状态代理量为 `9` 维：

- `AccKf 3`
- `GyroKf 3`
- `VelKf 3`

它们应解释为：

- 经过因果滤波和多源校正后的低噪声状态代理量
- 与控制输入对齐的体坐标系短时动态表征
- 用于学习状态转移关系的工程状态，而不是物理真值

### 当前活跃配置分支

1. `kf_ctx_v2` 主线  
   `29 -> 9`，用于多步 block rollout：
   `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8`

2. `quality_v3` 主线增强版  
   `34 -> 9`，在 `kf_ctx_v2` 基础上加入状态质量上下文：
   `HasDvlUpdate 1 + DtSinceDvl_s 1 + VelKfVar 3`

3. `quality_step_v1` 单步状态转移实验分支  
   保持 `34 -> 9`，但把 `pred_len` 收口为 `1`，用于更接近
   `x_{t+1} = f(x_t, u_t, c_t)` 的建模路径

### 当前主模型

- family: `S1Predictor`
- 默认 rollout: `y_hat = y0 + cumsum(dY)`
- 当前推荐损失：`transition_balance`
- 当前推荐训练监控：`val_transition_score`

这意味着当前仓库同时维护两条研究线：

- 多步块输出主线，用于维持现有评估与图包可比性
- 单步状态转移实验线，用于向控制与 RL 接口收口

## 为什么使用 KF 状态代理量

当前工程判断是：对“状态转移求解器”来说，训练目标使用因果滤波后的体坐标系状态代理量，比直接使用原始传感器序列更有代表性。

原因是原始异步观测会把以下问题同时压给网络：

- 时间对齐
- 去噪和 bias 抑制
- 稀疏观测补全
- 受控动力学学习

而 `KF / ESKF` 预处理把学习问题收口为：

```text
给定控制输入 u_t、
代理状态 x_t、
质量上下文 c_t，
学习 x_{t+1} 或短时 rollout。
```

这更接近系统辨识和控制建模问题本身，也更利于后续闭环验证。

## 当前已经完成的关键升级

当前仓库已经完成以下基础升级：

- `KF / ESKF` 初值 warm-start 已改为严格因果，不再从未来 DVL 泄漏初始化速度。
- 共享状态维 `X/Y` scaler 已收口到单一统计量，避免 delta-cumsum rollout 语义失真。
- `quality_v3` 质量上下文配置已经加入输入链。
- `quality_step_v1` 单步状态转移实验分支已经建立。
- 8 卡服务器矩阵已经拆成：
  - `H=10` quality-context 主线矩阵
  - `H=1` single-step transition 矩阵

因此，当前项目已经具备“重建数据集 -> 单卡 smoke -> 8 卡矩阵 -> 离线评估”的完整实验骨架。

## 当前阶段边界

当前仍需保持这几个边界判断：

- KF 输出是状态代理量，不是高保真物理真值。
- 当前默认主模型仍以 `S1Predictor` 为核心。
- 多步主线仍是“历史窗 -> 固定未来块输出”，还不是严格的一步动力学算子。
- 离线指标只能证明 rollout 与短时动态拟合质量，不能直接等价于闭环可用性证明。

## 最短上手路径

仓库根目录：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

建议阅读顺序：

1. `docs/handover_guide.md`
2. `docs/project_status.md`
3. `ARCHITECTURE.md`
4. `docs/design/transition_solver_phase1_upgrade.md`
5. `docs/handover_kf_training_server_v2.md`

## 最短执行顺序

1. 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

2. 构建数据集

`kf_ctx_v2`：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

`quality_v3`：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

`quality_step_v1`：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

3. 单卡 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml
```

或单步实验：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

4. 8 卡矩阵

`H=10` quality-context 主线：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v1.yaml
```

`H=1` single-step 实验线：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v1.yaml
```

## 评估口径

当前推荐的 run 级判断口径至少包括：

- `rmse_global / mae_global`
- `rmse_global_masked / mae_global_masked`
- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- 有效步数、失败步数、非有限值触发次数

如果进入最小闭环，还应额外记录：

- 单步推理延迟
- 控制循环周期
- 实际运行频率

## 状态求解器升级

当前已经新增最小“经验型状态求解器 + 长序列 replay”升级链：

- 求解器模块：`src/uwnav_dynamics/solver/transition_solver.py`
- replay 验证：`src/uwnav_dynamics/cli/transition_replay.py`
- replay 批量排行：`src/uwnav_dynamics/cli/transition_replay_matrix.py`
- 服务器全流程：`src/uwnav_dynamics/cli/server_pipeline.py`

推荐先用 `pred_len=1` 的一步状态转移分支训练，再做长序列 replay：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml

python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_steps 50

python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/replay_matrix_example.yaml

python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_v1.yaml
```

详细升级路线见：

- `docs/design/transition_solver_phase2_replay_upgrade.md`

## 文档入口

- 文档总导航：`docs/README.md`
- 项目状态：`docs/project_status.md`
- 当前交接入口：`docs/handover_guide.md`
- 数据与 git 规则：`docs/data_management.md`
- 服务器执行入口：`docs/handover_kf_training_server_v2.md`
- 命令手册：`docs/reference/quick_commands.md`
- 工程升级路线：`docs/engineering_roadmap.md`
- 建模路线：`docs/modeling_roadmap.md`
- 设计说明：`docs/design/transition_solver_phase1_upgrade.md`
- 状态求解器升级：`docs/design/transition_solver_phase2_replay_upgrade.md`
- 数学原理：`docs/math/main.tex`
- 评估规范：`docs/evaluation_protocol.md`
- 文件索引：`docs/repo_index.md`

## 开发与验证约束

- 修改训练、预处理、评估或接口契约后，必须同步更新文档。
- 新增代码文件必须包含中文模块说明。
- 当前阶段优先做最小闭环验证，不扩大成完整系统重做。
- 本地默认只做轻量测试；正式训练与评估以 8 卡服务器为主。

推荐最小自检：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_train_data_pipeline_nan_sanitization.py \
  tests/test_transition_balance_design.py \
  tests/test_kf_ctx_quality_training_config_v3.py \
  tests/test_kf_ctx_quality_step_config_v1.py \
  tests/test_train_matrix_server_configs.py
```

## License

当前许可证为 `AGPL-3.0-or-later`，见 `LICENSE`。
