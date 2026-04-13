# KF 融合训练服务器迁移交接文档

更新时间：2026-04-12  
当前工作分支：`feature/kf-preprocess-training-v1`

## 1. 迁移目标

服务器上的目标不是恢复旧的 controller 验证流程，而是尽快把当前主线跑通：

1. 生成新的 KF 融合基础表
2. 构建新的 KF 数据集
3. 单卡 smoke 验证训练链
4. 优先启动 8 卡 `single-step` 矩阵
5. 基于 replay / solver 结果决定是否补跑长期拟合对照

注意：

- 当前文档里的“直接上服务器训练”顺序只在完成 Phase 1 修复后才成立。
- 当前前两个基础阻塞已经收口，剩余主阻塞是一歩状态转移建模尚未完成。

## 2. 当前主线摘要

当前主线已经固定为：

- 因果 `KF / ESKF` 融合预处理
- `29` 维输入、`9` 维主监督
- `S1Predictor + grouped head + transition_balance`
- 训练期按 `val_transition_score` 选 best ckpt
- run 级按长期 rollout 相关指标筛选

当前服务器侧推荐拆成两个 8 卡批次：

1. `H=1` 的 single-step transition 主筛选批次
2. `H=10` 的 quality-context 对照批次

这样做的原因是：

- 当前 8 张卡已可全部用于并发矩阵
- 避免 `pred_len=10` 与 `pred_len=1` 混入同一 compare 链
- 让一步分支先回答“是否适合做状态转移求解器”，再决定是否补长期拟合对照

当前已经完成的基础修复是：

1. `KF / ESKF` 初始化速度已改为严格因果 warm-start。
2. 共享状态维的 `x_scaler / y_scaler` 已收口为单一统计量。

当前仍未闭合的主阻塞是：

3. 当前模型还是 `hist -> future block` 预测器，不是严格的一步状态转移算子。

## 3. 需要同步到服务器的核心路径

代码与配置：

- `src/uwnav_dynamics/`
- `configs/fusion/pooltest02_kf_eskf_v2.yaml`
- `configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- `configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
- `configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
- `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`
- `configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml`
- `configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml`
- `configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml`
- `configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml`
- `configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml`
- `configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml`

文档：

- `docs/README.md`
- `README.md`
- `docs/handover_guide.md`
- `docs/handover_kf_training_server_v2.md`
- `docs/data_management.md`
- `docs/reference/quick_commands.md`
- `docs/design/kf_fusion_preprocess_training_v2.md`

## 3.1 下次进入服务器后的最小恢复动作

```bash
cd /path/to/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
git log --oneline -3
export PYTHONPATH=src
```

然后先做两件事：

1. 读 `docs/handover_guide.md` 第 `3.1` 节
2. 跑本页第 `4.1` 节自检命令

## 4. 服务器上的最短执行顺序

```bash
cd /path/to/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
export PYTHONPATH=src
```

### 4.1 配置与代码自检

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_transition_balance_design.py \
  tests/test_trainer_controls.py \
  tests/test_kf_ctx_training_config_v2.py
```

### 4.2 当前更推荐的顺序

当前更推荐的顺序是：

1. 先确认 Phase 1 自检通过
2. 重建融合基础表与数据集
3. 做单卡 smoke
4. 再进入 8 卡服务器批次

一步状态转移训练配置仍是后续主线，不属于本页的已完成部分。

### 4.2.1 一键全流程入口

如果服务器环境已经就绪，且 8 张卡都可用，
当前更推荐直接使用 8 卡全流程总控入口：

```bash
python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml
```

该入口会按顺序执行：

1. KF / ESKF 融合基础表生成
2. `quality_v3` 与 `quality_step_v1` 数据集构建
3. 单卡 smoke
4. 两个 8 卡矩阵批次
5. 基于 `train_matrix/summary.csv` 自动生成 replay matrix 并完成长序列方案评估

主要总控产物：

```text
out/server_pipeline/pooltest02_kf_full_8gpu_v2/manifest.yaml
out/server_pipeline/pooltest02_kf_full_8gpu_v2/phase_status.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/generated_replay_matrix/*.yaml
out/server_pipeline/pooltest02_kf_full_8gpu_v2/logs/*.log
```

当前 8 卡总控最终结果重点看：

```text
out/train_matrix/pooltest02_s1_kf_quality_8gpu_v2/summary.csv
out/train_matrix/pooltest02_s1_kf_quality_step_8gpu_v2/summary.csv
out/replay_matrix/pooltest02_s1_kf_quality_8gpu_v2/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2/ranking.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/final_selection.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/paper_artifact_manifest.yaml
```

如果当前仍需让出一张卡，再回退使用：

```bash
python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml
```

对应的 7 卡回退结果重点看：

```text
out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/summary.csv
out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv
out/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/ranking.csv
out/server_pipeline/pooltest02_kf_full_7gpu_v2/final_selection.csv
out/server_pipeline/pooltest02_kf_full_7gpu_v2/paper_artifact_manifest.yaml
```

### 4.2.2 本轮服务器执行记录（2026-04-11 7 GPU 基线批次）

本轮已确认：

- `fusion`：成功
- `dataset`：成功
- `quality_v3_smoke`：成功
- `quality_step_v1_smoke`：成功
- `pooltest02_s1_kf_quality_7gpu_v2`：部分成功，phase 记为 `failed`
- `pooltest02_s1_kf_quality_step_7gpu_v2`：部分成功，phase 记为 `failed`
- 两个 replay phase：已执行并写出 `summary.csv / ranking.csv`

证据入口：

- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/phase_status.csv`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/logs/train_matrix_pooltest02_s1_kf_quality_7gpu_v2.log`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/logs/train_matrix_pooltest02_s1_kf_quality_step_7gpu_v2.log`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/logs/replay_quality_v3_replay.log`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/logs/replay_quality_step_v1_replay.log`

当前离线最优候选：

- `quality_v3`：
  - `QV3_B4_grouped_tb_blocks_seed8`
  - `rmse_global = 0.05492`
  - `mae_global = 0.01779`
- `quality_step_v1`：
  - `STEP_B4_grouped_tb_blocks_seed9`
  - `rmse_global = 0.03635`
  - `mae_global = 0.00690`

当前结论：

- `quality_step_v1` 明显优于 `quality_v3`，是下一步状态转移求解器主候选。
- 两个最优 run 都已确认 `runtime_device = cuda`，不是 CPU fallback。
- 服务器端 `fusion + dataset` 预处理阶段已经完成；若原始 CSV 与配置未变化，下次进入服务器默认不需要先重跑 `4.3 / 4.4`。

当前失败原因：

- `quality_v3` 的 `V2_B4` 两个候选失败，原因是缺少：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/features.npz`
- `quality_step_v1` 的 `step_a0_joint_nll_seed8/9` 两个候选失败，原因是：
  - `loss.type=nll_diag` 与当前 `transition_balance` 字段组合不满足配置契约

一个必须注意的路径问题：

- replay 日志显示结果写到了仓库外路径，而不是仓库内 `out/replay_matrix/`：
  - `/home/wys/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2/`
  - `/home/wys/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/`

所以下次接手时，优先先把这两个目录回收到仓库内，或者直接把其中的：

- `summary.csv`
- `ranking.csv`
- 关键 run 目录

拷回 `out/replay_matrix/`，再做最终 solver 选型。

### 4.2.3 当前更实用的命令行顺序：训练 -> 评估 -> 验证 -> 可视化 -> 保存

如果服务器端预处理产物仍然有效，当前更推荐直接从网络训练开始：

1. 启动单步主线网络训练

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

2. 训练后做测试集评估并自动出图

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

3. 对同一权重做长序列状态转移 replay 验证

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_steps 50
```

4. 如需进入下一轮正式训练，优先启动 8 卡矩阵

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

5. 如 replay / solver 复核后仍需长期拟合对照，再启动：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml
```

6. 若当前服务器只允许 7 张卡并发，再回退到：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml
```

7. 保存当前主结果时，至少保留以下产物

```bash
RUN_DIR=run.out_dir/run.variant
mkdir -p out/archive/step_transition_main
cp -r "$RUN_DIR"/resolved_train.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/train_history.csv out/archive/step_transition_main/
cp -r "$RUN_DIR"/train_summary.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/best.pth out/archive/step_transition_main/
cp -r "$RUN_DIR"/eval_test out/archive/step_transition_main/
cp -r "$RUN_DIR"/replay_test out/archive/step_transition_main/
```

这里的 `RUN_DIR` 需要替换成真实运行目录，例如：

```text
out/ckpts/pooltest02_s1_kf_quality_step_7gpu_v2/STEP_B4_grouped_tb_blocks_seed9
```

建议保存后至少复核：

```text
RUN_DIR/eval_test/metrics.yaml
RUN_DIR/eval_test/plots/
RUN_DIR/replay_test/metrics.yaml
RUN_DIR/replay_test/segment_metrics.csv
```

### 4.3 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

### 4.4 构建数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

### 4.5 单卡 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml
```

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

### 4.6 8 卡矩阵批次 A：single-step 主线

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

### 4.7 8 卡矩阵批次 B：quality-context 对照

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml
```

### 4.8 7 卡回退批次

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml

python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml
```

如果只是接着今天的工作继续，不要先重跑整批；先回收 replay 结果并核对：

```bash
ls -lah /home/wys/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2
ls -lah /home/wys/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2
```

## 5. 迁移后先检查什么

融合基础表：

- `AccKf / GyroKf / VelKf` 列全 finite
- 没有 NaN 注入训练表

数据集：

- `features.npz["X"]` 形状正确
- `labels.npz["Y"]` 形状正确
- `labels.npz["dvl_mask"]` 均值应为 `1.0`

训练：

- `train_summary.yaml` 中能看到：
  - `monitor_name: val_transition_score`
  - `best_monitor`
  - `best_val_loss`
- `train_history.csv` 中应有：
  - `val_loss`
  - `monitor_value`

矩阵：

- `summary.csv` 正常产出
- 每个 run 都有 `train_summary.yaml` 和 `eval_test/metrics.yaml`
- `quality` 与 `quality_step` 两个矩阵各自产出独立 compare 目录
- `server_pipeline` 最终还应产出 `final_selection.csv` 与 `paper_artifact_manifest.yaml`

## 6. 当前筛选标准

训练期：

- 先看 `val_transition_score`
- 再看 `selected_val_loss / best_val_loss`

run 级：

- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- `acc / gyro / vel` 分组误差

补充说明：

- `H=10` 主线批次优先看长期 rollout 指标
- `H=1` 单步批次优先看 `final_step`、`delta` 收敛与 bias 稳定性

## 7. 图包约束

图包仍沿用统一风格：

- 无标题
- 只保留坐标轴标题、图例和单位
- `Times New Roman`
- 固定 `4:3`
- 配套 `figure_notes.md`

## 8. 当前不要做的事

- 不要先恢复旧的 controller 图壳层
- 不要先改 12 维主输出协议
- 不要只靠 `val_loss` 决定模型
- 不要把 KF 输出写成高保真物理真值
- 不要把 `pred_len=1` 与 `pred_len=10` 的 run 混到同一个 compare 批次
