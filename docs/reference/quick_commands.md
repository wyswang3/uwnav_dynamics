# 快捷命令行

说明：

- 本文是当前主线的命令参考手册，不是交接第一入口。
- 初次接手请先读 [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)。
- 数据目录与 git 规则见 [data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)。
- 当前命令分成两层：
  - 原始传感器重建链
  - 当前正式训练主线

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
  -y configs/dataset/pooltest02.yaml
```

主要产物：

```text
out/aux_power/motor_data_20260110_193455_power8.csv
out/power_plots/motor_data_20260110_193455/plots/power_currents_8motors.png
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

## 8.1 长序列状态求解器 replay

推荐优先对 `quality_step_v1` 训练结果做长序列 replay：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_steps 50
```

主要产物：

```text
run.out_dir/run.variant/replay_test/metrics.yaml
run.out_dir/run.variant/replay_test/segment_metrics.csv
run.out_dir/run.variant/replay_test/component_metrics.csv
run.out_dir/run.variant/replay_test/pred_samples.npz
```

## 8.2 多方案 replay 统一排行

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

## 9. 8 卡服务器重训

说明：

- 当前推荐策略不是单模型 DDP，而是每张卡独占一个单卡实验并行跑满 8 个变体。
- `pred_len=10` 与 `pred_len=1` 分开跑，避免混入同一个 compare 链。

### 9.0 一键服务器全流程

如果需要在 8 卡服务器上从预处理一直跑到最终方案评估，直接执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_v1.yaml
```

总控产物：

```text
out/server_pipeline/pooltest02_kf_full_v1/manifest.yaml
out/server_pipeline/pooltest02_kf_full_v1/phase_status.csv
out/server_pipeline/pooltest02_kf_full_v1/generated_replay_matrix/
out/server_pipeline/pooltest02_kf_full_v1/logs/
```

最终重点查看：

```text
out/train_matrix/pooltest02_s1_kf_quality_8gpu_v1/summary.csv
out/train_matrix/pooltest02_s1_kf_quality_step_8gpu_v1/summary.csv
out/replay_matrix/pooltest02_s1_kf_quality_8gpu_v1/ranking.csv
out/replay_matrix/pooltest02_s1_kf_quality_step_8gpu_v1/ranking.csv
```

### 9.1 多步主线 8 卡矩阵

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v1.yaml
```

### 9.2 单步实验线 8 卡矩阵

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v1.yaml
```

矩阵后重点检查：

```text
out/train_matrix/<matrix_variant>/summary.csv
out/train_matrix/<matrix_variant>/logs/
run.out_dir/run.variant/eval_test/metrics.yaml
```

## 10. 最短推荐顺序

如果上游原始预处理产物已经存在，当前最推荐顺序是：

1. 本地最小自检
2. 运行对齐
3. 运行 KF / ESKF 融合
4. 构建 `quality_v3` 与 `quality_step_v1` 数据集
5. 本地只做单卡 smoke
6. 单次评估确认 `metrics.yaml` 与图包正常
7. 服务器先跑 `pooltest02_s1_kf_quality_8gpu_v1`
8. 再跑 `pooltest02_s1_kf_quality_step_8gpu_v1`
9. 对各批次 top2 做正式评估与图包

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
