# B4+U1 训练结果决策说明

本文不是论文正文，而是基于 `out/` 目录现有真实训练产物，对“为什么当前主线最终方案应选择 B4+U1”给出的技术决策说明。

## 结论先行

当前证据**支持**把 `B4+U1` 作为主线最终训练方案，但这个结论是**有前提的**：

- 如果最终筛选标准遵循项目在 `round4/round5` launch 配置中明确写出的口径，即“以 `masked metrics + control_readiness` 作为最终离线筛查信号”，那么 `B4+U1` 是更合理的主线选择。
- 如果只看 `dense RMSE(global)` 单一指标，那么 `U1` 在最终两轮确认实验中的平均值更低，`B4+U1` 并不是该指标上的最优模型。

因此，最科学、最谨慎的表述不是“B4+U1 全面优于 U1”，而是：

> 在当前实验体系下，`B4+U1` 以更好的 `MAE / masked MAE / final-step / rollout growth / tail error / bias` 离线控制前筛查表现，构成了更合适的主线最终方案；但它并未在 `dense RMSE` 上全面压倒 `U1`。

## 1. 审查范围与证据边界

本次实际检查的直接证据主要来自以下目录：

- `out/train_matrix/pooltest02_s1_8gpu`
- `out/train_matrix/pooltest02_s1_round2_struct40`
- `out/train_matrix/pooltest02_s1_round3_top3_e120`
- `out/train_matrix/pooltest02_s1_round4_controlconfirm_e120`
- `out/train_matrix/pooltest02_s1_round5_finalconfirm_e120`
- `out/ckpts/pooltest02_s1_8gpu`
- `out/ckpts/pooltest02_s1_round2_struct40`
- `out/ckpts/pooltest02_s1_round3_top3_e120`
- `out/ckpts/pooltest02_s1_round4_controlconfirm_e120`
- `configs/launch/pooltest02_s1_8gpu_compare.yaml`
- `configs/launch/pooltest02_s1_round2_struct40.yaml`
- `configs/launch/pooltest02_s1_round3_top3_e120.yaml`
- `configs/launch/pooltest02_s1_round4_controlconfirm_e120.yaml`
- `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml`

本次**没有**把以下内容当作主决策证据：

- `out/imu_*`、`out/pwm`、`out/power_*` 等预处理/传感器产物
- `out/ckpts/pooltest02_s1_lstm_v0` 这类更早期、单次、非当前主线筛选流程中的结果
- `out/ckpts/pooltest02_s1_lstm_smoke` 这类 smoke 产物

### 证据完整性风险

- `round5` 当前工作区中保留了 `summary.csv`、`manifest.yaml`、`logs/*.log` 和 `compare_test/*.png`，但**没有**保留对应的 `out/ckpts/pooltest02_s1_round5_finalconfirm_e120/.../eval_test/metrics.yaml` 与 `component_metrics.csv` 原始评估目录。  
  当前对 `round5` 的判断主要依赖 `summary.csv` 与日志行，因此这部分应标注为“**摘要级证据充分，原始评估目录缺失，若需逐分量复核需人工恢复**”。
- 当前 `out/ckpts/` 一级目录只看到：`pooltest02_s1_8gpu`、`pooltest02_s1_lstm_smoke`、`pooltest02_s1_lstm_v0`、`pooltest02_s1_round2_struct40`、`pooltest02_s1_round3_top3_e120`、`pooltest02_s1_round4_controlconfirm_e120`，没有 `pooltest02_s1_round5_finalconfirm_e120`。
- `round3` 的 `summary.csv/manifest.yaml` 中引用了 `configs/train/generated/pooltest02_s1_round3_top3_e120/*.yaml`，但这些生成配置文件不在当前工作区；不过对应 `resolved_train.yaml` 与 `logs/*.log` 仍然存在，因此**结构与结果仍可核查**。

## 2. 结果谱系梳理

### 2.1 实验阶段与命名规则

当前可重建的主线谱系如下：

| 阶段 | 目录 | 作用 | 当前判断 |
| --- | --- | --- | --- |
| 宽矩阵探索 | `out/train_matrix/pooltest02_s1_8gpu` | 在固定 backbone 下做 baseline / 结构模块 / hidden size 对比 | 用来识别首批候选：`B4` 与 `U1` 脱颖而出 |
| 首轮主线比较 | `out/train_matrix/pooltest02_s1_round2_struct40` | 比较 `B0 / B4 / U1 / B4+U1`，每组 2 seeds，预算缩短为 40 epochs | 用来判断 `B4+U1` 是否优于单独 `B4` 或 `U1` |
| Top3 延长预算 | `out/train_matrix/pooltest02_s1_round3_top3_e120` | 保留 `B0 / U1 / B4+U1`，恢复 120 epochs，并增加 `B4+U1 tuned` 分支 | 用来验证是否需要改变训练超参 |
| 控制前确认 | `out/train_matrix/pooltest02_s1_round4_controlconfirm_e120` | 只比较 `U1` 与 `B4+U1`，扩到 seed2-5 | 用来检验控制前离线筛查是否稳定 |
| 最终确认 | `out/train_matrix/pooltest02_s1_round5_finalconfirm_e120` | 继续只比较 `U1` 与 `B4+U1`，新增 seed6-9 | 用来做最终主线归并判断 |

命名规则可由 `summary.csv`、`manifest.yaml` 和 `resolved_train.yaml` 反推出来：

- `B0`：baseline，不启用额外结构模块
- `B1`：只启用 `thruster_lag`
- `B2`：只启用 `hydro_ssm`
- `B4`：`thruster_lag + hydro_ssm`
- `U1`：只启用 `uncertainty`
- `B4U1`：`thruster_lag + hydro_ssm + uncertainty`
- `default / tuned`：同一结构，不同训练超参
- `e40 / e120`：训练预算
- `control-confirm / final-confirm`：后两轮针对最终候选的确认实验

模块组合证据可直接在以下文件中看到：

- `out/ckpts/pooltest02_s1_round2_struct40/B4_thruster_hydro_seed0_e40/resolved_train.yaml`
- `out/ckpts/pooltest02_s1_round2_struct40/U1_uncertainty_seed0_e40/resolved_train.yaml`
- `out/ckpts/pooltest02_s1_round2_struct40/B4U1_thruster_hydro_uncertainty_seed0_e40/resolved_train.yaml`
- `out/ckpts/pooltest02_s1_round3_top3_e120/B4U1_thruster_hydro_uncertainty_tuned_seed0_e120/resolved_train.yaml`

### 2.2 哪些结果属于主线候选

按当前谱系，可把结果分成三类：

- **主线候选**：`round2` 的 `B4 / U1 / B4+U1`，`round3` 的 `U1 / B4+U1 default / B4+U1 tuned`，`round4-5` 的 `U1 / B4+U1`
- **中间探索**：`pooltest02_s1_8gpu` 中的 `b0_big512`、`b0_deep512x3`、`b1_thruster`、`b2_hydro`
- **辅助或应忽略结果**：`pooltest02_s1_lstm_smoke`、更早期 `pooltest02_s1_lstm_v0` 单次探索

launch 配置本身也给出了明确的决策路径：

- `configs/launch/pooltest02_s1_round3_top3_e120.yaml:5` 写明：因为 `B4` 的 2-seed family average 不如 `U1`，所以在 round3 中被移除。
- `configs/launch/pooltest02_s1_round4_controlconfirm_e120.yaml:2` 写明：round3 已经表明 `B4+U1 default` 是最佳候选。
- `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:2-5` 写明：round5 只保留 round4 缩窄后的候选集，并以 `masked metrics + control_readiness` 作为最终离线筛查信号。

## 3. 主线候选结果汇总

下表只保留与最终选型直接相关的主线节点。  
表中测试指标顺序统一为：`RMSE / MAE / masked RMSE / masked MAE`。

| 实验阶段 | 变体 | seed | 主要模块组合 | 关键验证指标 | 关键测试指标 | 是否主线候选 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round2 | B0 | 0,1 | baseline | `best_val=-2.680099±0.030061` | `0.022016±0.000208 / 0.009111±0.000054 / 0.022206±0.000262 / 0.010322±0.000073` | 参考基线 | 用来衡量结构收益 |
| round2 | B4 | 0,1 | `thruster_lag + hydro_ssm` | `best_val=-2.812793±0.002659` | `0.020501±0.000533 / 0.008594±0.000364 / 0.020810±0.000677 / 0.009488±0.000485` | 是 | 结构增强候选 |
| round2 | U1 | 0,1 | `uncertainty` | `best_val=-2.844976±0.024733` | `0.020101±0.000094 / 0.008484±0.000040 / 0.020556±0.000144 / 0.009128±0.000059` | 是 | 不确定度增强候选 |
| round2 | B4+U1 | 0,1 | `thruster_lag + hydro_ssm + uncertainty` | `best_val=-2.921329±0.003448` | `0.020031±0.000048 / 0.007972±0.000021 / 0.019972±0.000020 / 0.008673±0.000016` | 是 | round2 全部核心 test 指标最佳 |
| round3 | U1 | 0,1 | `uncertainty` | `best_val=-2.844976±0.024733` | `0.020101±0.000094 / 0.008484±0.000040 / 0.020556±0.000144 / 0.009128±0.000059` | 是 | 作为最强 challenger 保留 |
| round3 | B4+U1 default | 0,1 | `thruster_lag + hydro_ssm + uncertainty` | `best_val=-2.921329±0.003448` | `0.020031±0.000048 / 0.007972±0.000021 / 0.019972±0.000020 / 0.008673±0.000016` | 是 | 默认超参分支 |
| round3 | B4+U1 tuned | 0,1 | 同上，外加 `dropout=0.1, lr=5e-4, wd=3e-4` | `best_val=-2.956440±0.022884` | `0.020319±0.000286 / 0.008324±0.000033 / 0.020571±0.000355 / 0.009060±0.000088` | 否 | val 更低，但 test 整体变差 |
| round4 | U1 | 2,3,4,5 | `uncertainty` | `best_val=-2.827396±0.019417` | `0.019727±0.000106 / 0.008380±0.000077 / 0.019974±0.000146 / 0.008964±0.000078` | 是 | 控制前确认 challenger |
| round4 | B4+U1 | 2,3,4,5 | `thruster_lag + hydro_ssm + uncertainty` | `best_val=-2.918174±0.021913` | `0.020038±0.000122 / 0.008085±0.000079 / 0.020022±0.000113 / 0.008794±0.000104` | 是 | dense RMSE 略差，但 MAE 更优 |
| round5 | U1 | 6,7,8,9 | `uncertainty` | `best_val=-2.826292±0.018347` | `0.019613±0.000128 / 0.008363±0.000078 / 0.019844±0.000179 / 0.008907±0.000088` | 是 | 最终 challenger |
| round5 | B4+U1 | 6,7,8,9 | `thruster_lag + hydro_ssm + uncertainty` | `best_val=-2.924287±0.038832` | `0.020134±0.000259 / 0.008115±0.000145 / 0.020117±0.000293 / 0.008837±0.000182` | 是 | 最终 primary 候选 |

### 关键原始文件定位

- round2 汇总行：`out/train_matrix/pooltest02_s1_round2_struct40/summary.csv:2-9`
- round3 汇总行：`out/train_matrix/pooltest02_s1_round3_top3_e120/summary.csv:4-9`
- round4 汇总行：`out/train_matrix/pooltest02_s1_round4_controlconfirm_e120/summary.csv:2-9`
- round5 汇总行：`out/train_matrix/pooltest02_s1_round5_finalconfirm_e120/summary.csv:2-9`

## 4. 多轮训练结果趋势分析

### 4.1 从 baseline 到结构增强的趋势

**客观结果：**

- 在宽矩阵探索阶段，`B4` 与 `U1` 是最先明显拉开与 baseline 距离的两个候选。  
  `pooltest02_s1_8gpu/summary.csv` 中，`B4` 的 `rmse_global=0.019514`，`U1` 的 `rmse_global=0.019560`，都显著低于 baseline 家族均值 `0.021664`。
- `B1`（只加 `thruster_lag`）能带来一定收益，但 `B2`（只加 `hydro_ssm`）单独收益有限。说明结构先验不是“随便加一个模块就行”，而是 `thruster_lag + hydro_ssm` 的组合更有效。

**合理推断：**

- `thruster_lag` 捕捉推进器滞后，`hydro_ssm` 捕捉流体/速度相关动态，二者联合构成了比单模块更强的结构先验；这就是 `B4` 在探索阶段能够成为 primary 候选的原因。

### 4.2 从 B4 / U1 到 B4+U1 的趋势

**客观结果：**

- `round2` 是第一个真正决定“联合结构是否值得保留”的轮次。  
  在 2 seeds family average 上，`B4+U1` 同时优于 `B4` 和 `U1`：
  - 相比 `B4`：`RMSE 0.020501 -> 0.020031`，`MAE 0.008594 -> 0.007972`，`masked RMSE 0.020810 -> 0.019972`，`masked MAE 0.009488 -> 0.008673`
  - 相比 `U1`：`RMSE 0.020101 -> 0.020031`，`MAE 0.008484 -> 0.007972`，`masked RMSE 0.020556 -> 0.019972`，`masked MAE 0.009128 -> 0.008673`
- `B4+U1` 在 `round2` 的 seed 间波动也最小。  
  例如 `MAE` 标准差仅 `0.000021`，明显低于 `B4` 的 `0.000364` 与 `U1` 的 `0.000040`。

**合理推断：**

- `B4+U1` 不是把两个模块简单叠加，而是形成了“结构先验 + 不确定度建模”的互补：  
  `B4` 负责提升动力学表达，`U1` 负责改善误差分布建模，最终在 `round2` 上表现为四个核心 test 指标同时占优。

### 4.3 round3 的意义：不是再选结构，而是排除“需要改超参”的可能

**客观结果：**

- `round3` launch 文件明确写明：`B4` 被移除，因为其 2-seed family average 已经不如 `U1`。  
  这与 `round2` 的实际均值一致。
- `round3` 中 `B4+U1 default` 与 `round2` 的测试指标几乎完全一致，说明单纯把训练预算从 `40` 拉回 `120`，并没有带来新的 test 收益。
- `B4+U1 tuned` 虽然 validation 更低，但 test 更差。  
  代表性证据：
  - `out/train_matrix/pooltest02_s1_round3_top3_e120/logs/b4u1_default_seed0.log:146,153`  
    `best_val=-2.924777`，`RMSE/MAE=0.019983/0.007992`
  - `out/train_matrix/pooltest02_s1_round3_top3_e120/logs/b4u1_tuned_seed0.log:148,155`  
    `best_val=-2.979324`，`RMSE/MAE=0.020033/0.008290`

**合理推断：**

- `round3` 的主要价值不是“让 `B4+U1` 再涨一轮指标”，而是排除了“必须靠更保守的 dropout/lr/wd 才能让 B4+U1 成立”这条路径。  
  也就是说，最终主线应保留 **default B4+U1**，而不是 tuned 分支。

### 4.4 round4 / round5 的趋势：最终对手只剩 U1

**客观结果：**

- `round4` 和 `round5` 已经不再做大规模结构搜索，只比较 `U1` 与 `B4+U1`。这点在 launch 注释中写得很清楚。
- 在最终 8 个确认 seeds（round4 的 2-5，加上 round5 的 6-9）里：
  - `B4+U1` 在 `dense MAE` 上 **8/8 胜出**
  - `B4+U1` 在 `masked MAE` 上 **6/8 胜出**
  - `B4+U1` 在 `dense RMSE` 上 **0/8 胜出**
  - `B4+U1` 在 `masked RMSE` 上 **3/8 胜出**

**解释：**

- `U1` 的优势集中在 `RMSE`，特别是 `dense RMSE`
- `B4+U1` 的优势集中在 `MAE` 和后续控制前更关心的 rollout 稳定性指标

这意味着最终选择标准必须明确，否则会出现“按 RMSE 选是 U1，按 control-oriented screening 选是 B4+U1”的冲突。

## 5. B4+U1 的重点分析

### 5.1 它在所有主线候选中的整体位置

**客观结果：**

- 在 `round2`，`B4+U1` 是唯一一个在 family average 上同时拿下四个核心 test 指标最优的主线候选。
- 在 `round3`，`B4+U1 default` 继续保持优于 `U1` 的 `MAE / masked MAE`，而 `tuned` 分支被排除。
- 在 `round4` 与 `round5`，它不再是 `dense RMSE` 最优，但仍然是被 launch 配置保留为 `primary` 的最终候选，并且在最终离线筛查信号上更强。

### 5.2 它相对其他变体的主要优势

`round5` 是最终最有说服力的一轮，因为它同时保留了更多摘要字段。  
这些字段并不是随意挑的，它们来自 `metrics.yaml["control_readiness"]` 汇总进 `summary.csv` 的结果，映射关系可见：

- `src/uwnav_dynamics/experiment/reporting.py:121-178`
- `docs/evaluation_protocol.md:205-242`

其中 `control_readiness` 的用途被明确限定为“**离线筛查**，不是闭环最终证明”。

在 `round5` family average 上，`B4+U1` 的优势主要体现为：

- `dense MAE` 更低：`0.008115` vs `0.008363`
- `masked MAE` 更低：`0.008837` vs `0.008907`
- `final_step MAE(masked)` 更低：`0.012599` vs `0.012872`
- `tail_abs_p95(masked)` 更低：`0.036391` vs `0.036661`
- `tail_final_step_abs_p95(masked)` 更低：`0.050625` vs `0.051455`
- `rmse_growth(masked)` 更低：`2.1027` vs `2.2524`
- `mae_growth(masked)` 更低：`3.6815` vs `3.8863`
- `worst_abs_bias(masked)` 更低：`0.003898` vs `0.004106`

按 seed 对位比较，`B4+U1` 在 round5 上：

- `best_val`：**4/4 胜出**
- `final_step_mae_global_masked`：**4/4 胜出**
- `tail_abs_p95_masked`：**4/4 胜出**
- `tail_final_step_abs_p95_masked`：**4/4 胜出**
- `rmse_growth_masked`：**4/4 胜出**
- `mae_growth_masked`：**4/4 胜出**

这组证据非常关键，因为 `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:5` 已经明确把 `masked metrics + control_readiness` 定义为最终离线筛查信号。

### 5.3 这些优势是否稳定

**客观结果：**

- 如果用“在最终目标信号上是否重复胜出”来定义稳定性，那么 `B4+U1` 的优势是稳定的。  
  尤其在 round5 的 `final-step / tail / growth` 这组 control-oriented 指标上，呈现接近全胜的格局。
- 如果用“seed 方差是否更小”来定义稳定性，那么 `B4+U1` 不能说全面更稳。  
  例如 round5 中，它的 `dense RMSE` 标准差是 `0.000259`，大于 `U1` 的 `0.000128`。

**合理推断：**

- `B4+U1` 的“稳定”更接近于“**在最终离线控制筛查指标上稳定偏优**”，而不是“所有误差指标方差都更小”。

### 5.4 它的不足

这里必须明确写出来，否则结论会失真。

**客观结果：**

- `B4+U1` 在 `round4` 和 `round5` 的 `dense RMSE` family average 上都不如 `U1`
  - round4：`0.020038` vs `0.019727`
  - round5：`0.020134` vs `0.019613`
- `masked RMSE` 的 family average 也没有稳定领先
  - round4：`0.020022` vs `0.019974`
  - round5：`0.020117` vs `0.019844`

这说明：

- 如果最终论文或答辩只拿 `RMSE` 一个指标讲故事，那么 `B4+U1` 的证据链会不完整
- 选择 `B4+U1` 的逻辑必须建立在“控制前离线筛查目标优先”这一前提上

### 5.5 过拟合与训练不稳定性

`B4+U1` 还有一个很重要的训练现象：

- 在 `round5`，`B4+U1` 的 best epoch 平均是 `4.75`
- 同轮 `U1` 的 best epoch 平均是 `10.75`

代表性日志：

- `out/train_matrix/pooltest02_s1_round5_finalconfirm_e120/logs/b4u1_seed7.log:46-54`
- `out/train_matrix/pooltest02_s1_round5_finalconfirm_e120/logs/u1_seed7.log:59-67`

**客观结果：**

- `B4+U1` 收敛更快，也更早达到最优验证点
- 之后 validation 会明显恶化，说明它对训练预算更敏感

**合理推断：**

- 如果最终采用 `B4+U1`，必须同时保留 `ReduceLROnPlateau + early stopping`
- 不建议因为“还有 120 epochs”就延长训练；这个模型的有效最佳点往往很早

## 6. 最终决策建议

### 6.1 当前已有结果是否支持将 B4+U1 作为最终训练方案

**支持，但要带限定语。**

### 6.2 支持理由

- `round2` 证明了 `B4+U1` 不是偶然组合，而是在 `B4 / U1 / B0` 同台对比下，唯一能同时拿下四个核心 test 指标 family average 最优的组合。
- `round3` 证明默认超参的 `B4+U1` 已经足够，额外调参分支虽然能把 `val_loss` 拉得更低，但不能带来更好的 test 表现，因此主线应固定为 `default B4+U1`。
- `round4` 与 `round5` 明确把最终对手缩到 `U1`，而项目自己的 launch 注释又明确把 `masked metrics + control_readiness` 作为最终离线筛查信号；在这个信号下，`B4+U1` 的 `MAE / final-step / tail / growth / bias` 表现更强。
- `round5` 中 `B4+U1` 对 `U1` 在 `best_val` 上 4/4 全胜，在 `final_step_mae_global_masked`、`tail_abs_p95_masked`、`tail_final_step_abs_p95_masked`、`rmse_growth_masked`、`mae_growth_masked` 上也基本是 4/4 全胜，这条证据链比“单看一个 global RMSE”更贴近控制前筛查目标。

### 6.3 还缺少什么证据

以下内容仍建议标为“需人工确认”：

- `round5` 的原始 `eval_test/metrics.yaml`、`component_metrics.csv` 目录在当前工作区缺失。  
  如果后续需要做附录级别的逐分量验证，或要重画 final-confirm 的数值表，应先恢复 `out/ckpts/pooltest02_s1_round5_finalconfirm_e120/...` 原始评估目录。
- 当前尚无闭环控制或 controller-in-the-loop 结果。  
  `control_readiness` 只说明“值得进入下一步控制验证”，不能单独证明闭环最优。

### 6.4 当前最科学、最谨慎的表述

建议最终表述为：

> 基于 `round2 -> round3 -> round4 -> round5` 的连续筛选结果，当前最合理的主线最终训练方案是 `B4+U1`。  
> 其依据不是“所有误差指标都全面胜出”，而是：在项目已明确采用的 `masked metrics + control_readiness` 离线筛查框架下，`B4+U1` 持续表现出更优的 `MAE`、更低的 rollout 末步误差、更慢的误差增长和更小的最坏偏差，因此更适合作为控制前主线模型。  
> 同时需要保留一条注记：`U1` 在 `dense RMSE` 上仍然更强，因此 `B4+U1` 的胜出是“面向控制前筛查目标的胜出”，而不是“对所有指标的绝对统治”。

## 7. 口头说明版

### 一句话结论

如果最终目标是“选一个更适合进入控制前验证的主线模型”，而不是“只选 dense RMSE 最低的模型”，那么现有结果应选 `B4+U1`。

### 可直接口头复述的指标表

下表用 `round5` family average 概括最终差异：

| 指标 | U1 | B4+U1 | 更优方 | 说明 |
| --- | --- | --- | --- | --- |
| `RMSE(global)` | `0.019613` | `0.020134` | U1 | 纯 dense RMSE，U1 更低 |
| `MAE(global)` | `0.008363` | `0.008115` | B4+U1 | 8/8 confirm seeds 上 B4+U1 都更低 |
| `RMSE(masked)` | `0.019844` | `0.020117` | U1 | masked RMSE 仍是 U1 略优 |
| `MAE(masked)` | `0.008907` | `0.008837` | B4+U1 | 更贴近最终离线筛查目标 |
| `final_step MAE(masked)` | `0.012872` | `0.012599` | B4+U1 | 末步误差更小 |
| `tail_abs_p95(masked)` | `0.036661` | `0.036391` | B4+U1 | 尾部误差更低 |
| `rmse_growth(masked)` | `2.2524` | `2.1027` | B4+U1 | horizon 增长更慢 |
| `worst_abs_bias(masked)` | `0.004106` | `0.003898` | B4+U1 | 最坏偏差更小 |

### 最后一句

`B4+U1` 的核心优势不是“平均 RMSE 最低”，而是“在控制前更关键的 masked / tail / growth / bias 指标上更像一个可进入下一阶段验证的模型”。这就是当前应把它作为主线最终方案的主要理由。
