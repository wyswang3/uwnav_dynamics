# 状态转移求解器 7 GPU 全流程设计

更新时间：2026-04-11

> 历史说明：本文档是 7 GPU 全流程设计记录，不是当前最终选型入口。
> 2026-05-02 replay-only 复核后，当前默认状态转移求解器为
> `StepBase s11 / step_b0_grouped_tb_seed11`，详见
> [current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)。

## 1. 阶段目标

当前阶段的目标不是重新打开大规模结构搜索，而是在“只使用第 1 到第 7 张 GPU 卡”的资源约束下，跑通一条可复现的正式主线：

1. 从原始传感器预处理开始，重建训练基础表；
2. 生成 `quality_v3` 与 `quality_step_v1` 两条数据集；
3. 先做单卡 smoke，确认训练/评估/出图链正常；
4. 用 7 GPU 跑两批正式矩阵；
5. 用离线评估与长序列 replay 共同筛出最终状态转移求解器；
6. 固化图表与表格产物，服务后续 SCI 投稿图包与结果表。

说明：

- 若服务器按 0-based CUDA 编号暴露 GPU，则“第 1 到第 7 张卡”对应 `launcher.gpus: [0,1,2,3,4,5,6]`；
- 第 8 张物理卡对应 CUDA id `7`，在当前方案中不占用。

## 2. 当前主线结论

### 2.1 模型主线

- 默认 family 仍是 `S1Predictor`
- 默认 rollout 契约仍是 `y_hat = y0 + cumsum(dY)`
- 默认 solver 候选主线优先从 `quality_step_v1` 中筛选
- `quality_v3` 作为多步离线筛查与图表支撑线保留

### 2.2 为什么保留两条线

- `quality_v3 (pred_len=10)` 更适合看 horizon 误差、tail growth 和多步对比图；
- `quality_step_v1 (pred_len=1)` 更接近最终 `x_{t+1}=f(x_t,u_t,c_t)` 的状态转移求解器语义；
- 两条线共用同一套 KF 代理状态、训练器、评估器和可视化工具，证据链一致。

## 3. 全流程方案

### 阶段 1：原始数据预处理

阶段目标：

- 从原始 IMU / DVL / PWM / Power 数据重建统一可训练输入；
- 固化用于后续对齐、融合和可视化的中间产物。

最小实现：

1. IMU：`transform -> gravity -> bias -> filter`
2. DVL：统一速度列、有效性列与处理后图
3. Power：生成 8 路功率辅助表
4. PWM：对时并导出 8 路命令表

运行命令：

```bash
PYTHONPATH=src python apps/dev/test_imu_pipeline.py
PYTHONPATH=src python apps/dev/test_dvl_plots.py
PYTHONPATH=src python apps/dev/test_power_plots.py -y configs/dataset/pooltest02.yaml
PYTHONPATH=src python apps/tools/pwm_preprocess_and_plot.py -y configs/dataset/pooltest02.yaml
```

验证标准：

- 生成 `out/imu_proc/*.csv`、`out/dvl_proc/*.csv`、`out/aux_power/*_power8.csv`
- 原始图与处理后图都能正常输出
- 关键处理后列全 finite，没有明显时间轴异常

风险点：

- `apps/dev/` 与 `apps/tools/` 仍属于联调入口，适合当前阶段复现，不应在此轮顺手重构为新的业务框架；
- 若原始日志文件名变化，需要先同步 `configs/dataset/pooltest02.yaml`。

### 阶段 2：时间对齐与 KF 融合

阶段目标：

- 形成统一主时间轴的 `train_base.csv`
- 再用因果 KF / ESKF 生成 `train_base_kf_v2.csv`

最小实现：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml

PYTHONPATH=src python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

验证标准：

- 生成 `out/train/2026-01-10_pooltest02_train_base.csv`
- 生成 `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- `AccKf* / GyroKf* / VelKf* / HasDvlUpdate / DtSinceDvl_s` 全 finite

风险点：

- 当前融合输出仍是“低噪声状态代理量”，不是精确物理真值；
- 若 `train_base.csv` 上游中间表缺失，应先补齐阶段 1 产物，不要直接绕过。

### 阶段 3：数据集构建

阶段目标：

- 生成正式训练消费的数据集 artifact
- 保持多步主线与单步 solver 主线并行

最小实现：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml

PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

验证标准：

- 生成 `data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_v3/`
- 生成 `data/processed/2026-01-10_pooltest02_s1_kf_ctx_quality_step_v1/`
- `meta.yaml`、`features.npz`、`labels.npz` 完整存在

风险点：

- `quality_v3` 与 `quality_step_v1` 不要混入同一 compare 链；
- 单步线是 solver 主筛选线，多步线是支撑证据线，职责要分清。

### 阶段 4：单卡 smoke

阶段目标：

- 在正式矩阵前验证训练、评估、图表和 artifact 契约都正常

最小实现：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.pipeline \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml \
  --device cuda:0 \
  --epochs 2 \
  --plots \
  --plot_fmt png

PYTHONPATH=src python -m uwnav_dynamics.cli.pipeline \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --device cuda:0 \
  --epochs 2 \
  --plots \
  --plot_fmt png
```

验证标准：

- `run_dir/train_summary.yaml`、`train_history.csv` 存在
- `eval_test/metrics.yaml`、`rmse_by_horizon.csv`、`mae_by_horizon.csv` 存在
- `eval_test/plots/` 内图表可正常打开

风险点：

- smoke 通过只说明链路正常，不代表模型已达到 solver 目标；
- 若 smoke 失败，应先修配置或产物契约，不要直接上矩阵。

### 阶段 5：7 GPU 正式矩阵

阶段目标：

- 在第 8 张卡让给他人的前提下，完成两条正式矩阵训练与统一离线评估

最小实现：

多步主线：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml
```

单步 solver 主线：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml
```

验证标准：

- `out/train_matrix/*/summary.csv` 成功生成
- `configs/train/generated/<variant>/` 生成本轮物化 YAML
- `out/ckpts/*/<run>/eval_test/metrics.yaml` 完整存在

风险点：

- 当前每批仍保留 8 个候选，因此 7 GPU 下会有 1 个任务排队，这属于预期行为；
- 不应把“7 GPU 并发矩阵”写成 DDP。

### 阶段 6：状态转移求解器验证与最终筛选

阶段目标：

- 对单步候选执行长序列 autoregressive replay
- 用 replay 排行筛出最终 solver 候选

最小实现：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml
```

或单独运行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/replay_matrix_example.yaml
```

验证标准：

- 生成 `out/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/ranking.csv`
- 生成每个候选的 `metrics.yaml`、`segment_metrics.csv`、`component_metrics.csv`
- `segment_count > 0` 且无异常的 `nonfinite_trigger_count`

风险点：

- replay 是开环重放验证，不替代闭环控制证明；
- 最终 solver 选择应优先看 step 线 replay，不要只看多步 horizon 图。

## 4. 推荐产物清单

### 表格产物

- `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/summary.csv`
- `out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv`
- `out/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2/ranking.csv`
- `out/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/ranking.csv`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/final_selection.csv`
- `out/server_pipeline/pooltest02_kf_full_7gpu_v2/paper_artifact_manifest.yaml`
- `run_dir/eval_test/rmse_by_horizon.csv`
- `run_dir/eval_test/mae_by_horizon.csv`
- `run_dir/replay_test/segment_metrics.csv`
- `run_dir/eval_test/pred_sample_manifest.csv`
- `run_dir/replay_test/pred_sample_manifest.csv`

### 图表产物

- IMU / DVL / Power 原始与处理后图
- `eval_test/plots/horizon_metrics*`
- `eval_test/plots/control_readiness*`
- `eval_test/plots/rollout_sample_*`
- `replay_matrix/*/compare_*`

### 最终论文优先复用的证据

1. 多步主线 compare 图：证明长期离线预测改进；
2. 单步 replay 排行表：证明 solver 候选的长序列稳定性；
3. top2 的 rollout sample 与 replay sample 图：证明误差增长与末步行为；
4. `summary.csv + ranking.csv + final_selection.csv`：作为正文或附录表格来源。

## 5. 当前推荐命令顺序

```bash
export PYTHONPATH=src

python apps/dev/test_imu_pipeline.py
python apps/dev/test_dvl_plots.py
python apps/dev/test_power_plots.py -y configs/dataset/pooltest02.yaml
python apps/tools/pwm_preprocess_and_plot.py -y configs/dataset/pooltest02.yaml

python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml

python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml

python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml

python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml

python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_7gpu_v2.yaml
```

## 6. 当前结论

当前资源约束下，最稳妥的项目更新方式是：

- 不重写训练/评估业务逻辑；
- 用 `quality_v3 + quality_step_v1` 双线组织全流程；
- 用新增的 7 GPU launch/server 配置替换原推荐的 8 GPU 入口；
- 用 `final_selection.csv` 与 `paper_artifact_manifest.yaml` 固化最终选模与论文图表入口；
- 最终 solver 以 `quality_step_v1` 的 replay 结果为主判据，以多步主线图表作论文支撑证据。
