# 当前状态转移求解器最终方案

更新时间：2026-05-02

本文档是当前阶段的最终选型说明。凡是历史文档中出现的 `B4+U1`、
`StepDyn`、`STEP_B4`、`QualDyn` 等结论，只能作为对应阶段的历史证据；
当前用于后续控制器、经验型仿真内核和最小闭环验证的默认状态转移求解器，
以本文档为准。

## 1. 最终结论

当前经过 50 秒长序列 replay 验证后，推荐固定为：

```text
方案展示名：StepBase s11
候选目录名：step_b0_grouped_tb_seed11
数据路线：quality_step_v1
模型 family：S1Predictor
状态估计路线：因果 KF / ESKF 代理状态
输出契约：AccKf(3) + GyroKf(3) + VelKf(3)
预测步长：pred_len = 1
采样周期：dt = 0.01 s
```

工程表述上，这一版不是“高保真水动力学真值模型”，而是：

- 控制器内部短时状态转移模型；
- 经验型仿真回路内核；
- 可进入最小闭环验证的学习型动态预测器。

## 2. 为什么选择当前这一版

最终选择依据不是短窗口 eval 的单一 RMSE，而是 replay-only 固定口径下的
50 秒 autoregressive replay 排名。

关键证据：

```text
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/summary.csv
out/server_pipeline/replay_only_quality_step_8gpu_v2/final_selection.csv
out/server_pipeline/replay_only_quality_step_8gpu_v2/phase_status.csv
```

`phase_status.csv` 中 replay 阶段状态为：

```text
phase=replay
name=quality_step_v1_replay
status=ok
returncode=0
```

当前排名前两名为：

| 排名 | 候选 | 展示名 | overall_rank_score | 主要判断 |
| --- | --- | --- | ---: | --- |
| 1 | `step_b0_grouped_tb_seed11` | `StepBase s11` | `1.8235294117647058` | 综合最优，误差、尾部风险和增长稳定性更均衡 |
| 2 | `step_b2_grouped_tb_strong_delta_seed11` | `StepDelta s11` | `1.9411764705882353` | 纯误差更低，但 replay 增长风险明显更高 |

`StepBase s11` 的核心 replay 指标为：

| 指标 | 数值 |
| --- | ---: |
| `rmse_global` | `0.2519594653787598` |
| `mae_global` | `0.1161450209257922` |
| `final_step_rmse_global_mean` | `0.18731854021707864` |
| `rmse_growth_p95` | `131.0431527221474` |
| `tail_abs_p95_global` | `0.49592751264572055` |
| `tail_abs_p99_global` | `1.1673504388332405` |
| `worst_abs_bias` | `0.08816736124534703` |
| `nonfinite_trigger_count` | `0` |

`StepDelta s11` 虽然在 `rmse_global=0.1895367446983141`、
`mae_global=0.0939262779706241`、`tail_abs_p95_global=0.34642687737941735`
上更低，但 `rmse_growth_p95=316.2134034385202`，长序列递推增长风险
明显高于 `StepBase s11`。状态转移求解器后续要进入控制或仿真循环，
因此不能只按静态误差最小选型。

## 3. 当前技术思路

当前路线的核心思想是把“状态估计”和“状态转移学习”拆开：

```text
IMU / DVL / PWM / Power
        ↓
多源时间对齐
        ↓
因果 KF / ESKF 融合，形成 AccKf / GyroKf / VelKf 代理状态
        ↓
quality_step_v1 单步状态转移数据集
        ↓
S1Predictor LSTM 历史编码器
        ↓
grouped head 按 acc / gyro / vel 三组分别预测 dY 与 logvar
        ↓
y_hat = y0 + dY
        ↓
50 秒 autoregressive replay 排名
```

这一版采用 `KF / ESKF` 作为因果代理状态链，掩码主要承担质量上下文、
监督有效性和评估统计职责；当前最终方案不再把“掩码补速度”作为主状态估计方法。

## 4. 网络结构与模块组合

当前默认模型结构为：

| 项目 | 当前取值 |
| --- | --- |
| Backbone | `S1Predictor` |
| 历史编码器 | `LSTM` |
| Head | `grouped` |
| `group_head_hidden` | `160` |
| Loss | `transition_balance` |
| `pred_len` | `1` |
| Rollout | `y_hat = y0 + cumsum(dY)`，单步时等价于 `y0 + dY` |
| 输出维度 | `9` |
| 输出分组 | `acc=(0,1,2)`、`gyro=(3,4,5)`、`vel=(6,7,8)` |

`grouped head` 不是额外物理模块。它表示共享 LSTM 编码器和共享 trunk 之后，
按语义输出组分别设置 `acc / gyro / vel` 三个 head，每个 head 输出该组的
`dY + logvar`，最后再拼回 9 维状态增量。

代码证据：

```text
src/uwnav_dynamics/models/nets/s1_predictor.py
src/uwnav_dynamics/models/utils/semantic_output_layout.py
```

最终候选没有启用以下结构增强块：

| 模块 | 当前状态 |
| --- | --- |
| `thruster_lag` | `enabled = false` |
| `hydro_ssm` | `enabled = false` |
| `damping` | `enabled = false` |
| `uncertainty` | `enabled = false` |

因此，当前结论不是“物理先验块越多越好”，而是：

> 在当前数据、KF 代理状态和 50 秒 replay 口径下，简单的 `StepBase + grouped head + transition_balance`
> 比额外增加动力学 block 的方案更适合作为下一阶段默认状态转移求解器。

## 5. 评估数据与图片位置

核心表格：

```text
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/summary.csv
out/server_pipeline/replay_only_quality_step_8gpu_v2/final_selection.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/runs/step_b0_grouped_tb_seed11/metrics.yaml
```

候选配置与权重：

```text
out/server_pipeline/replay_only_quality_step_8gpu_v2/generated_replay_matrix/train_yamls/step_b0_grouped_tb_seed11.yaml
out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B0_grouped_tb_seed11/best.pth
```

可直接用于汇报的图片：

```text
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_model_compare.png
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_long_horizon_curves.png
```

## 6. 复现实验命令

如果训练产物已经存在，只需要运行 replay-only 评估，不需要重新跑完整训练流程：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml
```

服务器相对目录执行方式：

```bash
cd ~/WangYuShu/repos/uwnav_dynamics
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml
```

注意 `-c` 后必须给到具体 YAML 文件，不能只给目录。

## 7. 后续工程边界

当前证据足以支持把 `StepBase s11` 作为后续最小控制闭环或经验型仿真回路的
默认候选，但它还不等价于“闭环控制已经验证完成”。

下一阶段应围绕以下最小闭环继续验证：

```text
controller / policy
        ↓
learned transition model
        ↓
next state
        ↓
logging + metrics + plots
```

需要继续记录：

- 推理延迟；
- 实际循环频率；
- 闭环状态误差；
- 非有限值触发次数；
- 长时间 rollout 失败片段；
- 控制量与预测状态的对应日志。
