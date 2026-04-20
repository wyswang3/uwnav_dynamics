# uwnav_dynamics

面向水下机器人的科研级数据驱动状态转移建模仓库。

当前主线不是恢复完整精确水动力学，也不是直接交付闭环控制系统，而是把离线训练得到的学习型动力学模型推进到“可验证的状态转移求解器”阶段，使其能够承担短时状态递推、经验型仿真内核和 controller-in-the-loop / model-based RL 前置验证中的动态预测职责。

## 当前阶段

当前项目已经从“宽矩阵长时拟合探索”收口到：

**因果 KF 状态代理量 -> 质量上下文增强 -> 一步状态转移建模 -> 长序列 replay 验证**

当前应优先回答三个问题：

- 因果状态代理量链是否足够稳定，能支撑短时递推。
- 学习模型能否更接近 `x_{t+1} = f(x_t, u_t, c_t)` 的一步状态转移算子。
- 离线评估与长序列 replay 是否足以支撑“进入最小控制/仿真闭环前”的工程判断。

截至 2026-04-20，本地已完成多轮 7/8 卡结果回收与评估链升级：

- `quality_step_v1` 的单步状态转移主线优于 `quality_v3` 10-step 主线
- `B4 + blocks` 仍是当前最值得继续推进的结构家族
- 当前推荐默认 solver 候选为
  `out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B4_grouped_tb_blocks_seed10/eval_test`
- 默认评估图包已从短 0.1s 样例转向 50s 长窗口 Acc/Gyro/Vel 三轴诊断图
- 已新增 `step_with_feature_template()`，作为后续 controller / RL wrapper 的最小 step 边界

## 项目定位

本仓库当前聚焦三个问题：

- 如何把异步、多频、含噪传感器链路收口为统一的因果状态代理量。
- 如何训练一个对控制输入敏感、可递推、可评估的短时状态转移模型。
- 如何在不依赖高保真物理真值模型的前提下，为控制验证与 RL 提供工程上可用的动态内核。

当前对模型的正确定位是：

- 控制器内部短时状态转移模型
- 经验型仿真内核
- 面向 model-based RL 的可递推动态预测器

当前不应把本项目表述为：

- 高保真水下物理仿真器
- 完整闭环控制系统
- 已验证可实机部署的安全控制栈

## 方法总览

当前工程主线是：

```text
Raw Logs
-> IMU preprocess
-> Multi-rate alignment
-> KF / ESKF fusion
-> Proxy-state dataset build
-> S1Predictor training
-> Offline transition evaluation
-> Transition replay / solver ranking
-> Learned transition step interface
-> Minimal controller or simulator integration
```

默认主时间轴为 `100 Hz`，主要数据源为：

- PWM: `100 Hz`
- IMU: `100 Hz`
- DVL: `10 Hz`
- Power: `5 Hz`

数据与模型的核心契约如下。

### 代理状态

当前主监督状态代理量为 `9` 维：

- `AccKf 3`
- `GyroKf 3`
- `VelKf 3`

它们应解释为：

- 经过因果滤波和多源校正后的低噪声状态代理量
- 与控制输入对齐的体坐标系短时动态表征
- 用于学习状态转移关系的工程状态，而不是物理真值

### 当前活跃配置分支

1. `kf_ctx_v2`
   `29 -> 9`，作为历史对照主线与旧结果参照。

2. `quality_v3`
   `34 -> 9`，在 `kf_ctx_v2` 基础上加入质量上下文：
   `HasDvlUpdate 1 + DtSinceDvl_s 1 + VelKfVar 3`

3. `quality_step_v1`
   `34 -> 9`，保持相同输入语义，但将 `pred_len` 收口为 `1`，
   作为更接近 `x_{t+1}=f(x_t,u_t,c_t)` 的一步状态转移实验主线。

### 当前主模型

- family: `S1Predictor`
- 当前默认 rollout 语义：`y_hat = y0 + cumsum(dY)`
- 当前推荐损失：`transition_balance`
- 当前推荐训练监控：`val_transition_score`

这意味着仓库当前同时维护两条研究线：

- 多步块输出主线：维持现有 horizon / rollout 图包可比性
- 单步状态转移主线：向 solver、控制和 RL 接口收口

当前训练后求解器接口位于 `src/uwnav_dynamics/solver/transition_solver.py`：

- `predict_next_state(history_window)`：预测下一时刻主状态。
- `step_with_feature_template(history_window, feature_template)`：把预测状态写回下一行 feature 模板，是 controller / RL wrapper 的当前最小接口。
- `rollout_with_feature_templates(initial_history, future_templates)`：基于未来控制/上下文模板做长序列 autoregressive replay。

## 当前训练思路

当前更推荐的训练决策，不是“先把所有长时拟合再补一轮”，而是：

1. 先用 `quality_step_v1` 做一步状态转移矩阵
2. 基于 replay ranking 看 solver 稳定性和尾部误差
3. 再决定是否补跑 `quality_v3` 作为长期拟合对照

这样更符合当前阶段目标，因为我们现在要验证的是“能否作为状态转移部件使用”，而不是只追求长 horizon 图更好看。

当前已回收结果中，默认优先候选是：

```text
out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/
  STEP_B4_grouped_tb_blocks_seed10/eval_test/
```

关键指标：

- `rmse_global_masked = 0.0341668`
- `mae_global_masked = 0.0064416`
- `tail_p95_masked = 0.0198636`
- `tail_p99_masked = 0.1422205`

这些指标只支持“下一阶段 solver / replay / wrapper 优先候选”的判断，
不应表述为闭环控制已经完成验证。

## 核心数学原理

本项目的数学思路可以压缩为三层。

### 1. 因果状态代理量构造

原始观测是异步、多频、带噪且部分缺失的：

- IMU 稠密但噪声和 bias 明显
- DVL 稀疏但能提供强速度校正
- Power 是辅助致动上下文，不是主状态标签

因此当前不直接学习“原始观测流到未来观测流”的映射，而是先构造因果状态代理量：

```math
\hat{x}_k = \mathcal{F}(Y_{0:k}, u_{0:k})
```

这里 `\hat{x}_k` 表示只依赖当前及过去信息的工程状态代理量。  
这一步的目的是把学习问题从“同时做时间对齐、去噪、状态估计、动力学学习”，收口成“在已构造的状态空间上学习受控状态转移”。

### 2. 增量式状态转移建模

当前目标不是恢复完整物理参数，而是在代理状态空间中学习：

```math
x_{k+1} = f_\theta(x_k, u_k, c_k)
```

或等价地学习增量形式：

```math
\Delta x_k = g_\theta(x_k, u_k, c_k), \qquad x_{k+1} = x_k + \Delta x_k
```

其中：

- `x_k` 是 `AccKf + GyroKf + VelKf`
- `u_k` 是 8 维 PWM 控制输入
- `c_k` 是姿态上下文、DVL 新鲜度和功率等辅助上下文

多步主线仍采用 `dY + cumsum` 的 block rollout 语义；单步主线则在形式上更接近显式一步递推。

### 3. 离线筛查不等于闭环证明

当前训练与评估能证明的，是：

- 短时状态转移是否数值自洽
- rollout 是否稳定到足以进入 replay 验证
- 哪些候选在 `final_step / rollout_growth / tail_error` 上更适合继续推进

当前还不能直接证明的，是：

- 全局闭环稳定性
- 实机安全性
- 与高保真物理仿真的等价性

因此，当前推荐的证据链是：

```text
single-step train
-> offline eval
-> long-sequence replay
-> solver ranking
-> minimal controller/simulator integration
```

## 当前执行路线

仓库根目录：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

### 最短执行顺序

1. 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

2. 构建当前主数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

如需补长期拟合主线，再额外构建：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

3. 单卡 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

4. 当前推荐的 8 卡训练矩阵

先跑一步状态转移主线：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

若 replay / solver 复核后仍需要长期拟合对照，再跑：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml
```

5. 8 卡一键全流程

如果希望从预处理一直跑到 matrix + replay 结果整理，可直接执行：

```bash
python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml
```

如果服务器仍需让出一张卡，再回退到 `7gpu_v2` 方案。

6. 单次评估与默认可视化

```bash
python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --trace_seconds 50 \
  --plot_fmt png
```

默认图包重点包含：

- `prediction_trace_acc_axes.*`
- `prediction_trace_gyro_axes.*`
- `prediction_trace_vel_axes.*`

每张图只包含 X/Y/Z 三个共享 x 轴子窗。短窗口样例图需要显式加
`--sample_plots`，避免默认图包拥挤。

7. 长序列 replay 验证

```bash
python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_steps 50
```

replay 与 eval artifact 中的配置快照应保持相对路径，避免绑定某一台机器的绝对目录。

## 当前评估口径

当前推荐的 run 级判断口径至少包括：

- `rmse_global / mae_global`
- `rmse_global_masked / mae_global_masked`
- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- 有效步数、失败步数、非有限值触发次数
- 50s 长窗口 Acc/Gyro/Vel 三轴图是否非空、无遮挡、信息不过密
- replay / eval 配置快照是否保持相对路径

如果进入最小闭环，还应额外记录：

- 单步推理延迟
- 控制循环周期
- 实际运行频率

## 当前阶段边界

当前仍需保持这些边界判断：

- KF 输出是状态代理量，不是高保真物理真值
- `S1Predictor` 仍是当前默认主模型 family
- 多步主线仍是“历史窗 -> 固定未来块输出”，主要用于对照
- 单步主线已有 solver step 接口，但仍需要 replay ranking、最小 wrapper 与在线时延证据形成更强闭环
- 离线指标和 replay 结果只能证明控制前筛查价值，不能直接等价于闭环可用性证明

## 文档入口

- 文档总导航：`docs/README.md`
- 当前交接入口：`docs/handover_guide.md`
- 项目状态：`docs/project_status.md`
- 数据与 git 规则：`docs/data_management.md`
- 命令手册：`docs/reference/quick_commands.md`
- 服务器执行入口：`docs/handover_kf_training_server_v2.md`
- 工程升级路线：`docs/engineering_roadmap.md`
- 建模路线：`docs/modeling_roadmap.md`
- Phase 1 升级设计：`docs/design/transition_solver_phase1_upgrade.md`
- 8 卡训练计划：`docs/design/pooltest02_8gpu_plan_after_7gpu.md`
- 状态求解器升级：`docs/design/transition_solver_phase2_replay_upgrade.md`
- 最终选模与论文产物契约：`docs/design/final_selection_artifact_contract_v1.md`
- 数学原理入口：`docs/math/README.md`
- 主数学文档：`docs/math/main.tex`
- 理论文稿 PDF：`docs/math/main.pdf`
- 评估规范：`docs/evaluation_protocol.md`
- 文件索引：`docs/repo_index.md`

## 开发与验证约束

- 修改训练、预处理、评估或接口契约后，必须同步更新文档。
- 新增代码文件必须包含中文模块说明。
- 当前阶段优先做最小闭环验证，不扩大成完整系统重做。
- 当前正式训练优先以 8 卡并发矩阵为主，不把多卡并发矩阵误写成 DDP。

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
