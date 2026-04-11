# 基于 7 GPU 结果的 8 GPU 实验方案（2026-04-11）

## 1. 任务目标

当前 8 GPU 方案不再追求“把所有分支再跑一遍”，而是按当前阶段目标收口为两条线：

1. **主线优先**：更贴近实际需要的单步状态转移候选筛选  
   用于回答“哪类模型更适合做控制器内部的状态转移求解器”。
2. **辅助对照**：长期 rollout 拟合能力确认  
   用于保留多步长期拟合证据链，避免只看一步误差。

这样做的原因是：

- 当前阶段默认主目标是“可验证的状态转移模型”，不是继续做大而全的离线宽矩阵。
- 上一轮 7 GPU 已经给出明显淘汰信号，下一轮不需要再为已失败或明显偏弱的方案分配同等预算。

## 2. 上一轮 7 GPU 的直接证据

### 2.1 `quality_v3` 多步长期拟合线

证据文件：

- `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/summary.csv`
- `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/compare_test/control_readiness_compare.png`
- `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/compare_test/rmse_model_compare_horizon.png`
- `out/train_matrix/pooltest02_s1_kf_quality_7gpu_v2/compare_test/mae_model_compare_horizon.png`

按 family 聚合后的结论：

| family | seeds | rmse_global | mae_global | final_step_rmse | tail_abs_p95 | rmse_growth |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `QualBase` | 8,9 | `0.05952` | `0.02052` | `0.06161` | `0.10467` | `1.12895` |
| `QualTail` | 8,9 | `0.05966` | `0.02055` | `0.05864` | `0.10449` | `1.03006` |
| `QualDyn` | 8,9 | `0.05539` | `0.01798` | `0.05644` | `0.08989` | `1.05728` |

结论：

- `QualDyn (B4)` 是当前长期拟合主线，应继续扩种子确认。
- `QualTail (B2-longtail)` 虽然整体误差不如 `QualDyn`，但误差增长最低，适合保留为“抑制漂移”的对照分支。
- `QualBase` 已经完成基线职责，不再值得占用下一轮主要训练预算。
- `V2_B4` 上一轮训练失败，失败原因是缺失 `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/features.npz`，不应继续放在下一轮主矩阵里。

### 2.2 `quality_step_v1` 单步状态转移线

证据文件：

- `out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv`
- `out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/compare_test/control_readiness_compare.png`

按 family 聚合后的结论：

| family | seeds | rmse_global | mae_global | tail_abs_p95 | worst_abs_bias |
| --- | --- | ---: | ---: | ---: | ---: |
| `StepBase` | 8,9 | `0.03718` | `0.00736` | `0.02392` | `0.00456` |
| `StepDelta` | 8,9 | `0.04104` | `0.00850` | `0.02873` | `0.00341` |
| `StepDyn` | 8,9 | `0.03744` | `0.00743` | `0.02378` | `0.00644` |

补充：

- 单个最优 run 是 `STEP_B4_grouped_tb_blocks_seed9`，其 `rmse_global=0.03635`、`mae_global=0.00690`。  
  证据：`out/train_matrix/pooltest02_s1_kf_quality_step_7gpu_v2/summary.csv`
- `StepBase` 与 `StepDyn` 的 family 均值非常接近，因此下一轮更合理的问题不是“要不要继续保留 B0”，而是“B4 是否能在更多 seeds 下稳定拉开与 B0 的差距”。
- `StepDelta` 虽然整体误差更差，但 `worst_abs_bias` 最低，适合作为偏差控制方向的保底分支。
- `StepNLL` 两个 seed 均训练失败，原因是 `loss.type=nll_diag` 与当前 `transition_balance` 配置契约不兼容，不应继续占用 8 GPU 主矩阵名额。

## 3. 推荐 8 GPU 方案

### 3.1 Phase A：单步状态转移主线

阶段目标：

把 8 GPU 预算优先用于更贴近实际需求的状态转移求解器筛选。

最小实现：

- `StepDyn` 新增 `4` 个 seeds：`10,11,12,13`
- `StepBase` 新增 `2` 个 seeds：`10,11`
- `StepDelta` 新增 `2` 个 seeds：`10,11`
- 不再纳入 `StepNLL`

运行命令：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

验证标准：

- `out/train_matrix/pooltest02_s1_kf_quality_step_8gpu_v2/summary.csv` 成功写出 8 条记录。
- `StepDyn` 与 `StepBase` 至少形成 `n>=4` 的可比 family 证据。
- 进入下一步 replay/solver 验证前，优先看：
  - `rmse_global / mae_global`
  - `final_step`
  - `tail_abs_p95`
  - `worst_abs_bias`

风险点：

- `pred_len=1` 的离线评估不能替代长序列 autoregressive replay，训练完成后仍应进入 replay 确认。
- `StepDelta` 更偏“稳偏差”而非“最低均值误差”，不应只按单一 `rmse_global` 决策。

### 3.2 Phase B：长期拟合辅助对照

阶段目标：

保留长期 rollout 证据链，但预算集中给当前最强的 `B4` 主线。

最小实现：

- `QualDyn` 新增 `6` 个 seeds：`10,11,12,13,14,15`
- `QualTail` 新增 `2` 个 seeds：`10,11`
- 不再纳入新的 `QualBase` 补跑
- 不再纳入 `V2_B4`

运行命令：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml
```

验证标准：

- `out/train_matrix/pooltest02_s1_kf_quality_8gpu_v2/summary.csv` 成功写出 8 条记录。
- `QualDyn` family 的 `rmse_global / mae_global / final_step / tail_abs_p95` 继续稳定优于 `QualTail`。
- `QualTail` 只作为“增长率更低”的长期漂移对照，不再作为主线最终候选。

风险点：

- 该阶段只能补“固定窗口多步 rollout”证据，不能直接替代状态转移求解器验证。
- 若后续 replay 证据与 `quality_v3` 排序相冲突，仍应以状态转移主线为主。

## 4. 验证可视化

本次已基于上一轮 `quality_v3` 的评估产物，补做一张 family 级长期拟合对比图：

- 输出目录：`out/analysis/pooltest02_8gpu_plan_2026-04-11/`
- 主要图：`quality_family_long_fit_compare.png`
- 摘要表：`quality_family_summary.csv`

图的用途：

- 直接比较 `QualBase / QualTail / QualDyn` 三类方案在 horizon 维度上的长期误差走势；
- 用于说明为什么下一轮长期拟合线只继续保留 `B4` 主线和 `B2-longtail` 对照。

## 5. 本轮配置落地

本页对应的新配置：

- `configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml`
- `configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml`

建议执行顺序：

1. 先跑 `quality_step_8gpu_v2`
2. 训练后先做 replay / solver 复核
3. 再决定是否启动 `quality_8gpu_v2`

这比“先把所有长期拟合方案再补一轮”更贴合当前阶段目标。
