# 项目交接指南

更新时间：2026-05-02

## 1. 当前接手时先知道什么

当前项目主线已经从“先跑长时拟合矩阵”进一步收口为：

**`因果 KF 状态代理量 -> 单步状态转移求解器 -> 长序列 replay / RL-ready 基础接口`**

当前不要再从旧的 controller / transition validation 壳层切入。  
当前最重要的问题已经变成：

- 状态转移模型是否可以稳定执行超过 10 步的长序列递推
- 评估、replay 与图表 artifact 是否足够清晰、可复现、可比较
- 下一阶段如何把 `step_with_feature_template()` 包装成 controller / RL 环境接口

## 2. 先读哪几份文档

建议顺序：

1. [README.md](/home/wys/uwnav_dynamics/docs/README.md)
2. [current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)
3. [transition_solver_phase1_upgrade.md](/home/wys/uwnav_dynamics/docs/design/transition_solver_phase1_upgrade.md)
4. [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
5. [kf_fusion_preprocess_training_v2.md](/home/wys/uwnav_dynamics/docs/design/kf_fusion_preprocess_training_v2.md)
6. [common_network_baseline_protocol_v1.md](/home/wys/uwnav_dynamics/docs/design/common_network_baseline_protocol_v1.md)
7. [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
8. [quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)

如果只是要尽快恢复本地上下文，前 3 份足够。

## 2.1 下次接手的 5 分钟恢复顺序

在仓库根目录执行：

```bash
cd /home/wys/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
export PYTHONPATH=src
```

然后按这个顺序恢复上下文：

1. 先看 `docs/design/current_transition_solver_selection.md`，确认当前最终 solver 方案。
2. 再看本文件第 `3.1` 节，确认当前 solver / replay / visualization 已完成内容。
3. 再看 `docs/design/transition_solver_phase2_replay_upgrade.md`，确认状态求解器 replay 协议。
4. 再看 `docs/handover_kf_training_server_v2.md` 第 `4.2` 节，确认服务器上的执行顺序。
5. 若要做论文网络对照，先看 `docs/design/common_network_baseline_protocol_v1.md`，不要直接重开无约束搜索。
6. 需要查命令时，去 `docs/reference/quick_commands.md`，不要把它当主交接文档。
7. 如果要继续修代码，优先从 `data_pipeline.py`、`s1_predictor.py`、`run_train.py` 三处开始。

## 3. 当前已经落地的事实

当前可以直接依赖的事实：

- IMU `transform -> gravity -> bias -> filter` 预处理链稳定。
- `KF / ESKF` 融合模块已经落地到 `src/uwnav_dynamics/preprocess/fusion/`。
- `kf_ctx_v2` 数据集契约已经固定为 `29` 维输入、`9` 维目标。
- 训练损失已经支持 `transition_balance`。
- 训练期 best ckpt 已经支持按 `val_transition_score` 选模。
- Phase 1 已修复：
  - 融合初始化未来 DVL 泄漏
  - 共享状态维 `X/Y` scaler 语义错位
- Phase 2 预备配置已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
  - `configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- Phase 2 单步分支已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
  - `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`
- 训练后状态求解器已具备最小 step 接口：
  - `src/uwnav_dynamics/solver/transition_solver.py`
  - `TrainedTransitionSolver.predict_next_state()`
  - `TrainedTransitionSolver.step_with_feature_template()`
  - `TrainedTransitionSolver.rollout_with_feature_templates()`
- 数值评估已新增 50s 诊断 trace artifact：
  - `eval_<split>/pred_trace.npz`
- 默认评估出图已改为信息适度的长时序图：
  - `prediction_trace_acc_axes.png`
  - `prediction_trace_gyro_axes.png`
  - `prediction_trace_vel_axes.png`
  - 每张图 3 个子窗，X/Y/Z 三轴共享 x 轴
- replay / eval 配置快照继续遵循相对路径契约，避免把本机绝对目录写死进可复现实验产物。

## 3.1 当前阶段结论

Phase 1 已完成并收口了前两个基础阻塞：

1. `KF / ESKF` 融合初始化已改为严格因果 warm-start  
   代码位置：`src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

2. rollout 训练的共享状态维 scaler 已收口为单一统计量  
   代码位置：`src/uwnav_dynamics/train/data_pipeline.py`

当前剩余主阻塞已经从“有没有一步求解器接口”转为：

3. 还缺少正式 controller / RL wrapper。
   当前已有 `step_with_feature_template()` 与 replay 验证，
   但尚未实现完整 `reset()/step()/reward()/done` 环境，也尚未完成闭环控制证明。

补充：截至 2026-05-02，本地 `out/ckpts` 与 `out/replay_matrix` 中已经保留 8 GPU 训练与 50s replay-only 复核结果。

- 短 horizon / 单步 eval 曾经优先推荐：
  - `out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B4_grouped_tb_blocks_seed10/eval_test`
  - `rmse_global_masked = 0.0341668`
  - `mae_global_masked = 0.0064416`
  - `tail_p95_masked = 0.0198636`
  - `tail_p99_masked = 0.1422205`
- 但 50s 长序列 replay-only 复核后，当前默认 solver 候选已经改为：
  - `StepBase s11`
  - `step_b0_grouped_tb_seed11`
  - `out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B0_grouped_tb_seed11`
- 该候选模块组合为：
  - `S1Predictor`
  - `pred_len = 1`
  - `head_mode = grouped`
  - `transition_balance`
  - `thruster_lag / hydro_ssm / damping / uncertainty` blocks 均关闭
- 50s replay 关键指标：
  - `overall_rank = 1`
  - `overall_rank_score = 1.823529`
  - `rmse_global = 0.251959`
  - `mae_global = 0.116145`
  - `final_step_rmse_global_mean = 0.187319`
  - `rmse_growth_p95 = 131.043153`
  - `tail_abs_p95_global = 0.495928`
  - `tail_abs_p99_global = 1.167350`
  - `worst_abs_bias = 0.088167`
  - `nonfinite_trigger_count = 0`
- 核心证据文件：
  - `out/server_pipeline/replay_only_quality_step_8gpu_v2/phase_status.csv`
  - `out/server_pipeline/replay_only_quality_step_8gpu_v2/final_selection.csv`
  - `out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/ranking.csv`
  - `out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/summary.csv`
  - `out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/runs/step_b0_grouped_tb_seed11/metrics.yaml`
- 图表位置：
  - `out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_model_compare.png`
  - `out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_long_horizon_curves.png`
- 解释：`StepDelta s11` 的全局误差更低，但 `rmse_growth_p95 = 316.213403`，
  长时误差增长风险高于 `StepBase s11`，所以综合 ranking 选择 `StepBase s11`。
- 服务器侧 `fusion + dataset` 预处理阶段已经完成；若数据和配置未变化，下次进入服务器可直接从训练、评估和 replay 验证开始。
- 下一轮正式工作优先顺序：基于 `StepBase s11` 做 controller / RL wrapper smoke，并记录推理延迟、循环周期和失败触发。

## 4. 当前最短工作流

仓库根目录：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

1. 自检

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_train_data_pipeline_nan_sanitization.py \
  tests/test_transition_balance_design.py \
  tests/test_kf_ctx_training_config_v2.py
```

2. 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

3. 构建数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

如果要测试质量上下文版输入链，改用：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

如果要测试单步状态转移版数据集，改用：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

4. 单卡训练 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

质量上下文版 smoke：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml
```

单步状态转移版 smoke：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

5. 当前推荐训练矩阵顺序

说明：

- 当前训练采用多卡并发单卡实验矩阵，不是 DDP。
- 若 8 张卡都可用，优先使用 `8gpu_v2`；
- 若仍需让出一张卡，再回退到 `7gpu_v2`。
- 当前更推荐先做单步状态转移主线，再决定是否补长期拟合对照。

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml

python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml

PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
    -c configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml
```

如果当前只有 7 张卡可用，则回退到：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml

python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml
```

如果已有 8 GPU 训练矩阵产物，只需要重跑 50s replay 复核，不要重跑全流程：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml
```

6. 单次评估、可视化与 replay 验证

```bash
python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --trace_seconds 50 \
  --plot_fmt png

python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_seconds 50 \
  --max_seconds_per_segment 50 \
  --dt 0.01
```

7. 保存关键产物

```bash
RUN_DIR=run.out_dir/run.variant
mkdir -p out/archive/step_transition_main
cp -r "$RUN_DIR"/resolved_train.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/train_summary.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/best.pth out/archive/step_transition_main/
cp -r "$RUN_DIR"/eval_test out/archive/step_transition_main/
cp -r "$RUN_DIR"/replay_test out/archive/step_transition_main/
```

## 5. 当前最该盯的产物

训练前：

- `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/`

训练后：

- `run_dir/resolved_train.yaml`
- `run_dir/train_history.csv`
- `run_dir/train_summary.yaml`
- `run_dir/best.pth`
- `run_dir/eval_test/metrics.yaml`

矩阵后：

- `out/train_matrix/<variant>/summary.csv`

## 6. 当前选模口径

当前不再只看 `val_loss`。

训练期：

- ckpt 选择和 early stopping 优先按 `train.metric`
- 当前推荐值是 `val_transition_score`
- `train_history.csv` / `train_summary.yaml` 还应并行保留
  `val_rmse_global_zspace / val_mae_global_zspace`，
  作为训练空间中的通用误差基线

run 级筛选：

- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- `acc / gyro / vel` 组误差
- `pred_trace.npz` 对应的 50s 三轴图是否非空、无遮挡、信息不过密
- replay `metrics.yaml / resolved_replay.yaml` 中的路径是否保持相对路径快照
- 长时长 replay 是否真的使用 `min_seconds: 50`，而不是误用 `min_steps: 50`

## 7. 现在不要先做什么

当前不建议先做：

- 恢复旧的 controller 验证文档链
- 直接扩成 12 维主输出协议
- 先画大量图再看数值
- 把 KF 输出当成“绝对真值”来表述

## 8. 当前下一步

下一步最合理的顺序是：

1. 固定 `StepBase s11 / step_b0_grouped_tb_seed11` 为当前默认状态转移求解器候选
2. 基于 `step_with_feature_template()` 做最小 controller / RL wrapper smoke
3. 记录推理延迟、循环周期、失败步数、非有限值触发次数
4. 保持默认图表为 3 到 4 个子窗；长时序诊断优先使用 replay compare 图和 Acc/Gyro/Vel 三轴图

如果后续要继续扩展，应优先扩训练与评估链、求解器接口与最小 replay 验证，
而不是重新打开旧阶段的大型验证壳层。
