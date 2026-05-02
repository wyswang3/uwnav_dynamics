# 快捷命令行

说明：

- 本文是当前主线的命令参考手册，不是交接第一入口。
- 初次接手请先读 [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)。
- 数据目录与 git 规则见 [data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)。
- 当前命令分成两层：
  - 原始传感器重建链
  - 当前正式状态转移求解器验证链

当前最终方案见
[current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)：
`StepBase s11 / step_b0_grouped_tb_seed11` 是 2026-05-02 replay-only
验证后的默认状态转移求解器。

## 1. 环境

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

## 2. 本地最小自检

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_train_data_pipeline_nan_sanitization.py \
  tests/test_transition_balance_design.py \
  tests/test_kf_ctx_quality_training_config_v3.py \
  tests/test_kf_ctx_quality_step_config_v1.py \
  tests/test_train_matrix_server_configs.py
```

## 3. 从原始传感器开始的预处理链

说明：

- 当前 IMU / DVL / Power 的“原始 CSV -> 处理后 CSV”入口仍是 `apps/dev/` 下的联调脚本。
- 它们会生成当前主线真正消费的 `imu_proc / dvl_proc / aux_power` 产物。
- PWM 对齐后的 `out/pwm/..._aligned.csv` 属于中间产物，不进入 `data/raw/`。

### 3.1 IMU 预处理

```bash
PYTHONPATH=src python apps/dev/test_imu_pipeline.py
```

主要产物：

```text
out/imu_proc/min_imu_tb_20260110_193348_proc.csv
out/imu_plots/min_imu_tb_20260110_193348/plots/imu_raw_9axis.png
out/imu_plots_proc/min_imu_tb_20260110_193348_proc/plots/imu_proc_3rows.png
```

### 3.2 DVL 预处理

```bash
PYTHONPATH=src python apps/dev/test_dvl_plots.py
```

主要产物：

```text
out/dvl_proc/dvl_nav_state_tb_20260110_193538_proc.csv
out/dvl_plots/dvl_nav_state_tb_20260110_193538/plots/dvl_vel_BI_BE.png
out/dvl_plots_proc/dvl_nav_state_tb_20260110_193538_proc/plots/dvl_proc_BI_BE_BD.png
```

### 3.3 电机功率辅助数据

```bash
PYTHONPATH=src python apps/dev/test_power_plots.py \
  -y configs/dataset/pooltest02.yaml \
  --rel-time \
  --window-mode peak_total_power \
  --window-s 60
```

主要产物：

```text
out/aux_power/motor_data_20260110_193455_power8.csv
out/power_plots/motor_data_20260110_193455/plots/power_sync_overview_8motors.png
```

## 4. 多源时间对齐，生成 train_base.csv

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml
```

主要产物：

```text
out/train/2026-01-10_pooltest02_train_base.csv
```

## 5. 因果 KF / ESKF 融合，生成 train_base_kf_v2.csv

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

主要产物：

```text
out/train/2026-01-10_pooltest02_train_base_kf_v2.csv
docs/math/figures/pooltest02_kf_proxy_overview.png
docs/math/figures/pooltest02_kf_proxy_fullrun.png
```

## 6. 构建数据集

### 6.1 历史对照主线 `kf_ctx_v2`

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

### 6.2 当前推荐多步主线 `quality_v3`

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

### 6.3 单步状态转移实验线 `quality_step_v1`

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

主要产物：

```text
data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/
data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_v3/
data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1/
```

## 7. 单卡训练 smoke

### 7.1 多步主线 `quality_v3`

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml
```

### 7.2 单步实验线 `quality_step_v1`

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

## 8. 单次评估与出图

### 8.1 评估 `quality_v3`

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

### 8.2 评估 `quality_step_v1`

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

评估后重点检查：

```text
run_dir/eval_test/metrics.yaml
run_dir/eval_test/rmse_by_horizon.csv
run_dir/eval_test/mae_by_horizon.csv
run_dir/eval_test/plots/
```

## 8.3 长序列状态求解器 replay

推荐优先对 `quality_step_v1` 训练结果做长序列 replay：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_seconds 50 \
  --max_seconds_per_segment 50 \
  --dt 0.01
```

注意：100 Hz 数据中 `50 steps = 0.5s`，50 秒长时验证应使用 `--min_seconds 50`。

主要产物：

```text
run.out_dir/run.variant/replay_test/metrics.yaml
run.out_dir/run.variant/replay_test/segment_metrics.csv
run.out_dir/run.variant/replay_test/component_metrics.csv
run.out_dir/run.variant/replay_test/pred_samples.npz
```

## 8.4 训练 -> 评估 -> 验证 -> 可视化 -> 保存

如果服务器侧 `fusion + dataset` 已经完成，当前最推荐直接从训练链开始：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml

PYTHONPATH=src python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --plot_fmt png

PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_seconds 50 \
  --max_seconds_per_segment 50 \
  --dt 0.01
```

保存最小结果包时，至少保留：

```bash
RUN_DIR=run.out_dir/run.variant
mkdir -p out/archive/step_transition_main
cp -r "$RUN_DIR"/resolved_train.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/train_summary.yaml out/archive/step_transition_main/
cp -r "$RUN_DIR"/best.pth out/archive/step_transition_main/
cp -r "$RUN_DIR"/eval_test out/archive/step_transition_main/
cp -r "$RUN_DIR"/replay_test out/archive/step_transition_main/
```

## 8.5 多方案 replay 统一排行

当你需要比较多种候选方案时，优先使用 replay matrix：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/replay_matrix_example.yaml
```

批量评估后重点检查：

```text
out/replay_matrix/<variant>/manifest.yaml
out/replay_matrix/<variant>/summary.csv
out/replay_matrix/<variant>/ranking.csv
out/replay_matrix/<variant>/runs/<candidate>/metrics.yaml
```

推荐主比较指标：

```text
rmse_global
mae_global
final_step.rmse_global_mean
rollout_growth.rmse_last_over_first_p95
tail_error.abs_p95_global
tail_error.abs_p99_global
bias.worst_abs_bias
```

## 8.6 当前最终方案 replay-only 复核

如果服务器或本地已经有 8 GPU 训练产物，当前不需要再跑全流程训练，
直接调用模型做 replay-only 评估即可：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml
```

重点检查：

```text
out/server_pipeline/replay_only_quality_step_8gpu_v2/phase_status.csv
out/server_pipeline/replay_only_quality_step_8gpu_v2/final_selection.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_model_compare.png
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2_fixed/compare_test/replay_long_horizon_curves.png
```

注意：`-c` 后必须给到具体 YAML 文件，不能只给 `configs/launch/` 目录。

## 9. 服务器重训

说明：

- 当前推荐策略不是单模型 DDP，而是多张 GPU 上并发跑独立单卡实验。
- 若服务器 8 张卡都可用，优先使用 `8gpu_v2` 方案；若仍需让出一张卡，再回退到 `7gpu_v2`。
- `pred_len=10` 与 `pred_len=1` 分开跑，避免混入同一个 compare 链。

### 9.0 一键服务器全流程（8 卡优先）

如果 8 GPU 训练产物已经存在，优先执行 replay-only 复核：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml
```

如果 8 张卡都可用，并希望从预处理一直跑到最终方案评估，直接执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml
```

总控产物：

```text
out/server_pipeline/pooltest02_kf_full_8gpu_v2/manifest.yaml
out/server_pipeline/pooltest02_kf_full_8gpu_v2/phase_status.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/generated_replay_matrix/
out/server_pipeline/pooltest02_kf_full_8gpu_v2/logs/
```

如果当前仍按 7 卡资源约束运行，再执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml
```

总控产物：

```text
out/server_pipeline/pooltest02_kf_full_7gpu_v2/manifest.yaml
out/server_pipeline/pooltest02_kf_full_7gpu_v2/phase_status.csv
out/server_pipeline/pooltest02_kf_full_7gpu_v2/generated_replay_matrix/
out/server_pipeline/pooltest02_kf_full_7gpu_v2/logs/
```

8 卡全流程最终重点查看：

```text
out/train_matrix/pooltest02_s1_kf_quality_8gpu_v2/summary.csv
out/train_matrix/pooltest02_s1_kf_quality_step_8gpu_v2/summary.csv
out/replay_matrix/pooltest02_s1_kf_quality_8gpu_v2/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v2/ranking.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/final_selection.csv
out/server_pipeline/pooltest02_kf_full_8gpu_v2/paper_artifact_manifest.yaml
```

7 卡回退方案最终重点查看：

```text
out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/summary.csv
out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv
out/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/ranking.csv
out/server_pipeline/pooltest02_kf_full_7gpu_v2/final_selection.csv
out/server_pipeline/pooltest02_kf_full_7gpu_v2/paper_artifact_manifest.yaml
```

### 9.1 论文结果一键包

如果希望从原始观测预处理/观测图开始，一直收口到训练示例、模块对比图、路线对比图和 compare 导出，直接执行：

```bash
bash scripts/run_paper_results_bundle.sh
```

如果要显式指定配置，执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.paper_results_bundle \
  -c configs/launch/pooltest02_paper_results_bundle_7gpu_v1.yaml
```

如果希望把传感器预处理图、训练示例图、模块/路线对比图与 `8gpu_v2`
服务器全流程一起收口，执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.paper_results_bundle \
  -c configs/launch/pooltest02_paper_results_bundle_8gpu_v1.yaml
```

主要产物：

```text
out/paper_results_bundle/pooltest02_7gpu_v1/paper_results_bundle_manifest.yaml
out/paper_results_bundle/pooltest02_7gpu_v1/figures/sensors/
out/paper_results_bundle/pooltest02_7gpu_v1/figures/training_examples/
out/paper_results_bundle/pooltest02_7gpu_v1/figures/summaries/
out/paper_results_bundle/pooltest02_7gpu_v1/compare_exports/
out/paper_results_bundle/pooltest02_7gpu_v1/plot_warning_summary.yaml
out/paper_results_bundle/pooltest02_7gpu_v1/plot_warning_summary.txt
```

`8gpu` bundle 的对应产物路径为：

```text
out/paper_results_bundle/pooltest02_8gpu_v1/paper_results_bundle_manifest.yaml
out/paper_results_bundle/pooltest02_8gpu_v1/figures/sensors/
out/paper_results_bundle/pooltest02_8gpu_v1/figures/training_examples/
out/paper_results_bundle/pooltest02_8gpu_v1/figures/summaries/
out/paper_results_bundle/pooltest02_8gpu_v1/compare_exports/
out/paper_results_bundle/pooltest02_8gpu_v1/plot_warning_summary.yaml
out/paper_results_bundle/pooltest02_8gpu_v1/plot_warning_summary.txt
```

其中 `plot_warning_summary.*` 会汇总所有“单点/稀疏序列未绘制折线”的 sidecar 记录，便于后续集中排查。

当前默认 bundle 还会额外生成一组“路线内选优 + 跨路线赢家对比”图：

```text
out/paper_results_bundle/pooltest02_7gpu_v1/figures/summaries/route_comparison_suite/
  route_<scope_a>_module_compare_final_selection_replay.png
  route_<scope_b>_module_compare_final_selection_replay.png
  route_winner_compare_final_selection_replay.png
```

如果只想单独重画这组三图，可直接执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.viz.eval.plot_paper_ablation_summary \
  --csv out/server_pipeline/pooltest02_kf_full_7gpu_v2/final_selection.csv \
  --mode final_selection_replay \
  --route-suite \
  --scope quality_v3_replay quality_step_v1_replay \
  --route-name "KF Route" "Step Route"
```

### 9.2 当前更推荐的 8 卡训练顺序

当前阶段更推荐先跑单步状态转移主线，再决定是否补跑长期拟合主线：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

如果 replay / solver 复核后仍需补长期拟合对照，再跑：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml
```

### 9.3 7 卡回退矩阵

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml
```

### 9.4 单步实验线 7 卡矩阵

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml
```

矩阵后重点检查：

```text
out/train_matrix/<matrix_variant>/summary.csv
out/train_matrix/<matrix_variant>/logs/
run.out_dir/run.variant/eval_test/metrics.yaml
```

补充：

- `eval_test/pred_sample_manifest.csv` 与 `replay_test/pred_sample_manifest.csv` 现在保存的是代表性样例，不再是“前 N 个”。

## 10. 最短推荐顺序

如果 8 GPU 训练产物已经存在，当前最短顺序是：

1. 本地最小自检
2. 执行 `pooltest02_s1_kf_quality_step_8gpu_v2_replay_only.yaml`
3. 检查 `ranking.csv`、`final_selection.csv` 与 compare 图
4. 固定 `StepBase s11 / step_b0_grouped_tb_seed11` 作为后续最小闭环默认候选

只有当训练产物缺失或需要重建证据链时，才回到完整顺序：

1. 运行对齐
2. 运行 KF / ESKF 融合
3. 构建 `quality_step_v1` 数据集
4. 本地只做单卡 smoke
5. 服务器跑 `pooltest02_s1_kf_quality_step_8gpu_v2`
6. 执行 replay-only 复核
7. 检查最终图表与指标

如果要从原始传感器开始重建，则把第 `3` 节先跑完，再进入上述顺序。

## 11. 关键目录

对齐基础表：

```text
out/train/2026-01-10_pooltest02_train_base.csv
```

KF 融合基础表：

```text
out/train/2026-01-10_pooltest02_train_base_kf_v2.csv
```

数据集：

```text
data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_v3/
data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1/
```

训练输出：

```text
run.out_dir/run.variant/
```

矩阵汇总：

```text
out/train_matrix/<matrix_variant>/summary.csv
```
