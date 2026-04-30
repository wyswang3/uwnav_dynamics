<!--
文档名称：论文图谱升级方案

文档职责：
面向当前小论文写作与项目工程收敛，
把“仓库里已有很多单点图”升级为“围绕论文证据链组织的一组主图与对比图”。

主要内容：
1. 定义保留/降级/新增的图类型。
2. 说明每张主图依赖哪些模块、数据产物与脚本。
3. 给出最小可执行命令、验证标准与风险点。

备注：
- 本文档只约束论文图谱与工程落地，不替代论文正文。
- 图面风格以 `src/uwnav_dynamics/viz/style/sci_style.py` 为唯一真源。
-->

# 论文图谱升级方案

## 1. 阶段目标

当前项目需要从“单模块局部绘图”升级到“面向论文结构的证据图谱”。

当前升级目标：

1. 保留真正支撑文章主线的图。
2. 降级或废弃不能独立支撑论点的单点图。
3. 新增跨模块对比图，服务 `prediction validity / ablation / route comparison / rollout readiness` 四条结果主线。

## 2. 图类型分级

### 2.1 保留为主文候选

| 图类型 | 文章作用 | 主要来源 |
| --- | --- | --- |
| Experimental platform | 证明真实池试平台与数据来源 | 现有照片资产 |
| Representative multi-source observations | 证明 PWM / IMU / DVL / power 已同步且可用于状态构造 | `imu_plot.py` / `dvl_plots.py` / `pwm_preprocess_and_plot.py` / `power_plots.py` |
| Transition architecture | 解释模型结构与状态转移语义 | 现有示意图脚本 |
| Horizon-wise error growth | 证明误差随 horizon 的增长规律 | `plot_model_compare.py` |
| Representative short-horizon prediction | 证明局部轨迹跟踪质量 | `plot_prediction_trace.py` / `plot_rollout_samples.py` |
| Ablation summary | 证明结构模块贡献 | `plot_paper_ablation_summary.py` |
| Masked vs Kalman route comparison | 证明两条 state-construction route 的取舍 | `plot_model_compare.py` + `plot_replay_compare.py` + route-specific artifacts |
| Long-horizon rollout / replay compare | 证明 transition solver 的长线稳定性 | `plot_replay_compare.py` / `plot_long_horizon_summary.py` |

### 2.2 降级为 QA / 附录候选

| 图类型 | 原因 |
| --- | --- |
| current-only 8 电机面板图 | 只能说明局部通道波形，不足以单独支撑文章中的“电推进响应”论点 |
| 单独 IMU / DVL / PWM 原始图 | 适合作为审查或补充材料，不应替代同步多源证据图 |
| 粗流程图式 actuation chain / Kalman pipeline 小图 | 信息密度低，容易和主文方法图重复 |

### 2.3 当前不建议进入主文

| 图类型 | 原因 |
| --- | --- |
| 只展示单一时间点或极短局部片段的点状图 | 缺少时序和趋势证据，无法独立支撑控制前预测叙事 |
| 与当前项目主风格冲突的旧稿图 | 容易造成视觉语言割裂，且不少图的单位/窗口尚未最终核实 |

## 3. 图与模块映射

### 3.1 多源观测主图

阶段目标：

把原先分散的 `IMU / DVL / PWM / current-only power` 图，收口成同一时间窗下的多源同步 excerpt。

最小实现：

- PWM：复用 [apps/tools/pwm_preprocess_and_plot.py](/home/wys/uwnav_dynamics/apps/tools/pwm_preprocess_and_plot.py)
- IMU：复用 [src/uwnav_dynamics/viz/plots/imu_plot.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/plots/imu_plot.py)
- DVL：复用 [src/uwnav_dynamics/viz/plots/dvl_plots.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/plots/dvl_plots.py)
- Power：升级 [src/uwnav_dynamics/viz/plots/power_plots.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/plots/power_plots.py) 为“总功率 + 8 电机功率”

运行命令：

```bash
PYTHONPATH=src python apps/dev/test_power_plots.py \
  -y configs/dataset/pooltest02.yaml \
  --rel-time \
  --window-mode peak_total_power \
  --window-s 60
```

验证标准：

- 产物输出 `power_sync_overview_8motors.png`
- 顶部为总功率，底部为 `T1..T8` 八路功率
- 纵轴单位统一为 `W`
- 图面符合项目当前白底、无网格、`Times New Roman` 主风格

风险点：

- 代表性时间窗若自动按功率峰值选取，后续 montage 仍需在 PWM / IMU / DVL 侧复用同一绝对时间窗

### 3.2 Horizon compare

阶段目标：

把“单模型 horizon 图”升级成论文可用的模型对比主图。

最小实现：

- 直接复用 [src/uwnav_dynamics/viz/eval/plot_model_compare.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/eval/plot_model_compare.py)

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.viz.eval.plot_model_compare \
  --eval_dir <eval_dir_a> <eval_dir_b> \
  --label "U1" "B4+U1" \
  --role ablation primary \
  --metric rmse
```

验证标准：

- 输出 `rmse_model_compare_horizon.png`
- `primary / ablation / baseline` 视觉层级正确
- 适用于论文中的 horizon-wise error growth

风险点：

- route comparison 需要 masked route 与 Kalman route 在同一 split / 同一 horizon 下产物齐全

### 3.3 Ablation / route summary

阶段目标：

把纯表格的控制相关指标升级为紧凑的论文 summary 图。

最小实现：

- 使用新增 [src/uwnav_dynamics/viz/eval/plot_paper_ablation_summary.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/eval/plot_paper_ablation_summary.py)
- 直接消费 `summary.csv`、`ranking.csv` 或 `final_selection.csv`

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.viz.eval.plot_paper_ablation_summary \
  --csv <train_matrix_summary.csv> \
  --mode train_masked \
  --label "B1" "B4" "U1" "B4+U1"
```

验证标准：

- 输出 `paper_ablation_summary_train_masked.png`
- 指标至少覆盖 `MAE / final-step / tail / growth / worst bias`
- 归一化坐标可直接支持论文对“越低越好”趋势的解释

风险点：

- 如果输入 CSV 中 `label / role` 不规范，图例与视觉层级会受到影响

### 3.4 Replay / long-horizon compare

阶段目标：

把“模型是否可进入控制前 rollout”从文字判断升级为图形证据。

最小实现：

- 复用 [src/uwnav_dynamics/viz/eval/plot_replay_compare.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/eval/plot_replay_compare.py)
- 结合 [src/uwnav_dynamics/viz/eval/plot_long_horizon_summary.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/eval/plot_long_horizon_summary.py)

运行命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.viz.eval.plot_replay_compare \
  --run_dir <replay_dir_a> <replay_dir_b> \
  --label "Masked route" "KF route"
```

验证标准：

- 输出 replay summary compare 与 long-horizon curves
- 能支撑文章中关于 rollout stability / replay readiness 的论述

风险点：

- replay 图只说明离线状态转移验证，不可直接写成闭环控制已证实

## 4. 当前建议的主文图序列

1. Conceptual overview
2. Experimental platform
3. Representative synchronized multi-source observations
4. Transition architecture
5. Horizon-wise error growth
6. Representative short-horizon prediction
7. Ablation summary
8. Route comparison 或 long-horizon replay compare

## 5. 风格约束

- 代码层唯一真源是 [src/uwnav_dynamics/viz/style/sci_style.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/style/sci_style.py)
- 多行传感器布局真源是 [src/uwnav_dynamics/viz/style/imu_style.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/viz/style/imu_style.py)
- 正式实验数据图默认白底、无背景网格、`Times New Roman`
- 不再按旧稿建议单独引入 `Arial/Helvetica + 细灰网格` 风格

## 6. 当前最小闭环

本轮升级的最小闭环定义为：

1. 功率图从 current-only 升级为论文可用的同步功率主图。
2. 补齐一个论文导向的 ablation / route summary 绘图入口。
3. 把“保留什么 / 废弃什么 / 新增什么”的规则落盘到本文件与论文规划文档中。
4. 把单点/稀疏绘图跳过记录汇总到 bundle 根目录，避免后续人工逐图排查。

当前推荐的一键入口：

```bash
bash scripts/run_paper_results_bundle.sh
```

如需显式指定配置：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.paper_results_bundle \
  -c configs/launch/pooltest02_paper_results_bundle_7gpu_v1.yaml
```

运行完成后，除 `paper_results_bundle_manifest.yaml` 外，还应重点查看：

```text
out/paper_results_bundle/<variant>/plot_warning_summary.yaml
out/paper_results_bundle/<variant>/plot_warning_summary.txt
```

其中会统一汇总：

- 传感器主图中的单点/稀疏序列跳过记录
- 训练示例图中的稀疏绘图记录
- compare 导出目录下被复制过来的 `*.plot_warnings.txt`
