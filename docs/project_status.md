# 项目当前状态

更新时间：2026-04-20

## 1. 当前阶段

项目当前处于：

**`状态转移求解器升级已进入长序列 replay 与 RL-ready 基础接口阶段；当前推荐候选收口为 quality_step_8gpu_v2 的 B4+blocks 单步模型`**

当前主目标是：

- 保持现有 KF 状态代理量主线
- 把训练链真正推进到“可解释、可递推、可继续向控制与 RL 接口演进”的状态
- 让默认评估图表从短 0.1s 样例转向 50s 长窗口诊断，单图保持 3 到 4 个子窗

## 2. 当前已经完成的内容

### 数据预处理

- IMU `transform -> gravity -> bias -> filter` 链稳定
- 多频对齐主链稳定
- `KF / ESKF` 融合模块已落地
- Phase 1 已修复融合初值的未来 DVL 泄漏风险：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

### 数据集与 scaler 契约

- 当前主数据集契约仍是：
  - `configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- Phase 2 预备配置已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
- Phase 2 单步分支已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
- 输入 `29` 维：
  - `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8`
- quality v3 输入 `34` 维：
  - `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + DvlCtx 5 + Power 8`
- 输出 `9` 维：
  - `AccKf 3 + GyroKf 3 + VelKf 3`
- `VelKf` 仍按 dense supervision 进入训练
- Phase 1 已把共享状态维的 `X/Y` scaler 语义收口到单一真源：
  - `src/uwnav_dynamics/train/data_pipeline.py`
- `DtSinceDvl_s` 现已保证为有限值，可作为模型输入上下文使用：
  - `src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

### 训练与评估

- `transition_balance` 已接入主训练路径
- `transition_balance` 当前已补入保守的 `state_mse_weight`，用于提供通用 state regression 基线
- grouped head 已接入 `S1Predictor`
- 训练期监控仍支持：
  - `val_loss`
  - `val_transition_score`
- 训练期 history / summary 现并行记录：
  - `val_rmse_global_zspace`
  - `val_mae_global_zspace`
- 离线评估仍保留：
  - `rmse_global / mae_global`
  - `final_step`
  - `rollout_growth`
  - `tail_error`
  - `worst_abs_bias`

### 状态求解器升级

- 已新增训练后状态求解器封装：
  - `src/uwnav_dynamics/solver/transition_solver.py`
- 状态求解器现在提供三层接口：
  - `predict_next_state(history_window)`
  - `step_with_feature_template(history_window, feature_template)`
  - `rollout_with_feature_templates(initial_history, future_templates)`
- 已新增长序列 autoregressive replay 验证入口：
  - `src/uwnav_dynamics/solver/replay.py`
  - `src/uwnav_dynamics/cli/transition_replay.py`
- 当前建议优先基于 `quality_step_v1` 分支验证经验型状态求解器

### 评估与可视化

- 数值评估已新增长时序 artifact：
  - `eval_<split>/pred_trace.npz`
- `uwnav_dynamics.cli.eval --plots` 默认输出：
  - horizon RMSE / MAE
  - long-horizon summary
  - control-readiness summary
  - 50s Acc/Gyro/Vel 三轴预测对比图
  - metrics dashboard
- 短窗口样例图已移到 `--sample_plots` 显式开启，避免默认图包拥挤。
- `plot_replay_compare` 的 summary 图已收敛为 2x2 四窗，replay long-horizon 图保持 2x2 四窗。
- 所有新增评估和 replay 快照应继续使用相对路径，不把本机绝对路径写入交接 artifact。

### 2026-04-11 服务器训练快照（已完成 7 GPU 基线批次）

- 已完成：
  - `fusion`
  - `quality_v3` / `quality_step_v1` 数据集构建
  - 两个 single-card smoke
  - 两个 7 GPU train matrix
  - 两个 replay matrix 启动
- 因此服务器侧数据预处理已经完成；若原始 CSV 与配置未变化，下一次进入服务器可直接从训练与评估链开始，而不是默认重跑 `fusion / dataset`
- 证据：
  - `out/server_pipeline/pooltest02_kf_full_7gpu_v2/phase_status.csv`
- 当前 train matrix 的状态应理解为：
  - 两条矩阵都已有有效结果；
  - 但因部分候选失败，`server_pipeline` 中对应 phase 被记为 `failed`，不代表整批不可用。

当前离线最优候选：

- `quality_v3` 最优：
  - `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/summary.csv`
  - 当前最优 run：`QV3_B4_grouped_tb_blocks_seed8`
  - 关键指标：
    - `rmse_global = 0.05492`
    - `mae_global = 0.01779`
    - `tail_abs_p95_dense = 0.08882`
    - `tail_abs_p99_dense = 0.26575`
- `quality_step_v1` 最优：
  - `out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv`
  - 当前最优 run：`STEP_B4_grouped_tb_blocks_seed9`
  - 关键指标：
    - `rmse_global = 0.03635`
    - `mae_global = 0.00690`
    - `tail_abs_p95_dense = 0.02134`
    - `tail_abs_p99_dense = 0.16039`

2026-04-20 本地 `out/ckpts` 多轮结果补充比较：

- 当前更推荐的默认 solver 候选：
  - `out/ckpts/pooltest02_s1_kf_quality_step_8gpu_v2/STEP_B4_grouped_tb_blocks_seed10/eval_test`
  - `rmse_global_masked = 0.0341668`
  - `mae_global_masked = 0.0064416`
  - `tail_p95_masked = 0.0198636`
  - `tail_p99_masked = 0.1422205`
  - `worst_bias_masked = 0.0016737`
- 当前组均值显示：
  - `STEP_B4_blocks` 优于 `STEP_B0_base / STEP_B2_strong_delta`
  - `quality_step_v1` 单步主线优于 `quality_v3` 10-step 主线
- 注意：`quality_step_v1` 与 `quality_v3` 的 horizon 语义不同，不能只凭单个指标直接当作闭环结论；但从“短时状态转移求解器”定位看，`STEP_B4_grouped_tb_blocks_seed10` 是当前优先候选。

当前阶段可下的结论：

- 单步状态转移主线 `quality_step_v1` 明显优于 `quality_v3`。
- `B4 + blocks` 在当前单步主线上仍是最值得继续推进的候选。
- 下一轮正式训练应优先启动 `pooltest02_s1_kf_quality_step_8gpu_v2`，而不是默认重开 `quality_v3` 宽矩阵。
- 最优 run 的 `resolved_train.yaml` 中 `_meta.runtime_device = cuda`，说明主结果确实来自 GPU 训练：
  - `out/ckpts/pooltest02_s1_kf_quality_step_7gpu_v2/STEP_B4_grouped_tb_blocks_seed9/resolved_train.yaml`
  - `out/ckpts/pooltest02_s1_kf_quality_7gpu_v2/QV3_B4_grouped_tb_blocks_seed8/resolved_train.yaml`

本轮失败原因也已明确：

- `quality_v3` 中两个 `V2_B4` 候选失败，是因为缺少旧版数据集：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/features.npz`
- `quality_step_v1` 中两个 `A0 NLL` 候选失败，是因为 `loss.type=nll_diag` 与当前 `transition_balance` 字段组合不满足配置契约。

## 3. 当前最重要的工程事实

当前要特别明确四件事：

1. 现在的训练链在“因果性”和“rollout 数值语义”上比上一轮更可靠。
2. 当前默认主线已经有单步 solver 接口，但还不是完整 controller / RL 环境。
3. `quality v3` 与 `quality_step_v1` 两条配置分支已经准备好，可用于验证：
   - 状态代理量 + 质量上下文
   - 单步状态转移
4. 当前离线指标仍然只代表控制前筛查，不代表闭环可用性证明。
5. 历史服务器 replay 日志曾显示结果被写到了仓库外路径：
   - `/home/wys/replay_matrix/pooltest02_s1_kf_quality_7gpu_v2/`
   - `/home/wys/replay_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/`

## 4. 当前剩余主阻塞

当前剩余的主阻塞只剩一条：

1. 还缺少正式 controller / RL wrapper
   当前有 `step_with_feature_template()`、长序列 replay 和多方案 ranking，
   但还没有统一的 `reset()/step()/reward()/done` 环境封装，也没有闭环控制证明。

补充说明：

- 从当前离线结果看，`pred_len=1` 的单步主线已经优于 `pred_len=10` 质量上下文主线；
- 但最终是否转正为控制或 RL 内核，仍需要 replay ranking、最小闭环 smoke 与在线时延证据闭环。

## 5. 当前可直接复用的产物

- 融合基础表：
  - `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- KF 数据集：
  - `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/`
- 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- quality-context 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- single-step 训练配置：
  - `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`
- 8 GPU 矩阵：
  - `configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml`
  - `configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml`
- 8 GPU 全流程：
  - `configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml`
- 7 GPU 矩阵：
  - `configs/launch/pooltest02_s1_kf_quality_7gpu_v2.yaml`
  - `configs/launch/pooltest02_s1_kf_quality_step_7gpu_v2.yaml`
- 7 GPU 全流程设计：
  - `docs/design/transition_solver_full_pipeline_7gpu_v2.md`
- 8 GPU 下一轮计划：
  - `docs/design/pooltest02_8gpu_plan_after_7gpu.md`
- 升级设计文档：
  - `docs/design/transition_solver_phase1_upgrade.md`

## 6. 当前边界与风险

### 已知边界

- KF 输出仍应表述为“低噪声状态代理量”，不是物理真值
- 当前主输出仍是 `9` 维，不含姿态角主监督
- 当前 `step_with_feature_template()` 仍要求调用方提供下一时刻控制/上下文 feature 模板，
  尚未封装成完全独立的 `step(u_t, dt)` 环境接口

### 当前风险

- 本轮 7 GPU 正式矩阵已跑通主体，但有 2+2 个候选失败，导致 phase 状态不是全绿
- 历史 replay 结果若仍落在仓库外的 `/home/wys/replay_matrix/`，需要整理回 `out/replay_matrix/` 才适合长期引用
- `final_selection.csv` 与 `paper_artifact_manifest.yaml` 尚未在当前本地 `out/server_pipeline/` 中落盘确认
- 新图包尚未基于 replay ranking 做最终筛选
- 若模型主体不进一步收口为一步转移器，长期自由递推能力仍可能不足
- 当前 replay 仍属于开环重放验证，还不是闭环控制证明

## 7. 当前推荐动作

当前建议严格按这个顺序推进：

1. 以 `STEP_B4_grouped_tb_blocks_seed10` 为默认候选，补齐 replay matrix 与最终选模表
2. 若数据和配置未变化，不重跑 `fusion / dataset`，先做一次单卡 smoke 复核
3. 基于 `step_with_feature_template()` 实现最小 controller / RL wrapper smoke
4. 记录推理延迟、循环周期、失败步数与非有限值触发次数
5. 基于 replay / solver / wrapper 指标决定是否补跑 `configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml`
6. 整理 top2 图包，默认保持单图 3 到 4 个子窗，不恢复拥挤的短窗口样例图作为主图

## 8. 当前不建议的做法

- 不要把 Phase 1 的完成表述成“闭环控制已可用”
- 不要直接恢复旧的 controller 图壳层作为主入口
- 不要在还未形成一步状态转移契约前同时大改模型结构和闭环框架
