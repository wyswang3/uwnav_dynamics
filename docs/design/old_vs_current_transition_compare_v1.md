# 新旧主线长期拟合对比方案 V1

更新时间：2026-04-10

> 历史说明：本文档用于设计“旧主线 vs 当前主线”的对照实验，
> 其中的“当前主线候选”表述不代表 2026-05-02 之后的最终选型。
> 当前最终状态转移求解器是 `StepBase s11 / step_b0_grouped_tb_seed11`，
> 详见 [current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)。

## 1. 目标

本方案的目标不是重新打开全量结构搜索，
而是用最小可复现对比回答下面这个问题：

**过去主线与当前主线，哪一组网络在“长期状态转移拟合”上更接近实际运动观测？**

这里的“更接近实际运动观测”，当前统一收口为：

- 使用长序列 autoregressive replay
- 用网络递推出来的速度状态
- 对齐 `base_csv` 中真实存在的 DVL 体速度观测
- 只在 DVL 有效时刻计入误差

这样可以避免直接拿：

- 旧主线的 `Vel_state`
- 新主线的 `VelKf`

各自对各自代理真源做比较，从而把“代理量定义不同”误当作“网络更强”。

## 2. 当前设计判断

### 2.1 过去主线的问题不主要是 backbone 太弱

过去主线与当前主线的模型 family 本质上都还是：

- `S1Predictor`
- `LSTM` encoder
- `delta_cumsum` rollout

因此，过去主线的问题不主要是“LSTM 不能做状态转移”，
而是网络被迫同时承担了三类任务：

1. 从异步原始观测中补时间轴
2. 从稀疏 DVL 中补速度状态
3. 在补全后的代理量上再学习受控动力学

对应旧路线的关键证据：

- 旧数据集仍用 `VelX_state_mps / VelY_state_mps / VelZ_state_mps`：
  [pooltest02_s1.yaml](/home/wys/uwnav_dynamics/configs/dataset/pooltest02_s1.yaml)
- 旧正式候选仍指向旧数据目录 `data/processed/2026-01-10_pooltest02_s1`：
  [b4u1_seed8.yaml](/home/wys/uwnav_dynamics/configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed8.yaml)
- 旧训练监控仍以 `val_loss` 为主：
  [b4u1_seed8.yaml](/home/wys/uwnav_dynamics/configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed8.yaml)

### 2.2 当前主线更接近“状态转移求解器”语义

当前主线的优势主要来自：

1. 状态代理量改为因果 `KF / ESKF` 融合
2. `quality_step_v1` 已把 `pred_len` 收口为 `1`
3. grouped head + `transition_balance` 更贴近状态组别与递推误差
4. `DtSinceDvl_s / VelKfVar*` 等质量上下文可进入输入

对应证据：

- 当前 step 分支配置：
  [pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml](/home/wys/uwnav_dynamics/configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml)
- Phase 2 目标就是更接近 `x_{t+1}=f(x_t,u_t,c_t)`：
  [transition_solver_phase1_upgrade.md](/home/wys/uwnav_dynamics/docs/design/transition_solver_phase1_upgrade.md)

因此，如果目标是“能不能承担状态转移求解器责任”，
当前建议把**过去主线**与**当前 `quality_step_v1` 主候选**做对照，
而不是优先拿旧主线去和当前多步 block rollout 主线做静态图包比较。

当前简称约定见：

- [model_variant_aliases.md](/home/wys/uwnav_dynamics/docs/design/model_variant_aliases.md)

## 3. 分组方案

### 阶段目标

形成两个训练组，并在统一的 DVL 观测 replay 口径下比较长期拟合。

### 最小实现

训练组 A：过去主线

- 使用已有旧主线 final confirm 矩阵
- 重点看 `Unc` 与 `DynUnc`

训练组 B：当前主线

- 使用当前 `quality_step_v1` 矩阵
- 重点看 `StepBase`，并保留 `StepDelta / StepDyn` 作为对照

### 运行命令

过去主线训练组：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml
```

当前主线训练组：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

统一长期拟合比较：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/pooltest02_old_vs_current_observed_dvl_replay_v1.yaml
```

### 验证标准

重点查看：

- `out/replay_matrix/pooltest02_old_vs_current_observed_dvl_replay_v1/summary.csv`
- `out/replay_matrix/pooltest02_old_vs_current_observed_dvl_replay_v1/ranking.csv`

当前最重要的比较指标：

- `rmse_global`
- `mae_global`
- `final_step_rmse_global_mean`
- `rmse_growth_p95`
- `rmse_threshold_failure_rate`
- `tail_abs_p95_global`
- `tail_abs_p99_global`
- `worst_abs_bias`
- `nonfinite_trigger_count`

其中这些指标现在都按统一评估口径计算：

- 预测侧：
  - 旧主线取 `VelX_state_mps / VelY_state_mps / VelZ_state_mps`
  - 当前主线取 `VelKfX_body_mps / VelKfY_body_mps / VelKfZ_body_mps`
- 真值侧：
  - 一律取 `VelBx_body_mps / VelBy_body_mps / VelBz_body_mps`
- 掩码侧：
  - 一律只在 DVL 有效时刻计分

### 风险点

1. 这仍然是离线 replay，不是闭环控制证明。
2. 当前统一口径主要比较的是**速度长期拟合**，不是完整 9 维状态的“真实物理真值”。
3. 旧主线与当前主线的数据目录不同，必须先各自完成训练，之后再做统一 replay。
4. `configs/train/generated/pooltest02_s1_kf_quality_step_8gpu_v1/*.yaml` 由 train matrix 运行后生成；若训练组 B 尚未运行，统一 replay 会因为缺少这些生成配置而失败。

## 4. 当前建议的判读方式

如果统一 DVL 观测 replay 下出现下面结果，可以这样判读：

1. 当前最终候选 `StepBase s11` 的 `rmse_global / rmse_growth_p95 / tail_abs_p95_global`
   持续优于过去主线 `DynUnc`
   说明当前网络设计更适合作为状态转移求解器候选。

2. 当前主线在 `rmse_global` 更优，但 `nonfinite_trigger_count` 或 `tail_abs_p99_global`
   更差
   说明当前设计更准确，但递推稳定性还需要继续打磨。

3. 过去主线在统一 DVL 口径下仍更优
   说明当前问题不只是数据契约修正，
   还需要继续回头检查：
   - step 任务定义
   - loss 权重
   - grouped head / block 组合
   - 质量上下文是否真的提高了状态转移可辨识性

## 5. 本次新增入口

统一对比入口：

- [pooltest02_old_vs_current_observed_dvl_replay_v1.yaml](/home/wys/uwnav_dynamics/configs/launch/pooltest02_old_vs_current_observed_dvl_replay_v1.yaml)

该入口依赖 replay matrix 新增的自定义评估口径能力：

- 可以从 `feature_row` 中指定预测列
- 可以从 `base_csv` 中指定统一目标列
- 可以指定 DVL 有效性 mask 候选列
