# 实验评估规范

## 1. 目标

本规范用于统一：

- 训练后应保留哪些产物
- 评估应基于哪些 split 与 scaler
- smoke test 与正式实验如何区分
- 结果如何记录，便于复现与论文写作

补充说明：

- 当前训练链已经支持 `train.metric=val_transition_score`
- 因此 `train_summary.yaml` 与 `train_history.csv` 不应只记录 `val_loss`
- 对新的长期拟合主线，应同时记录：
  - `monitor_name`
  - `monitor_value`
  - `best_monitor`
  - `best_val_loss`
  - `val_rmse_global_zspace`
  - `val_mae_global_zspace`
- 兼容历史字段时要注意：
  - `best_val` 仍保留为 legacy 字段
  - 当 `train.metric != val_loss` 时，`best_val` 表示“最佳 monitor 值”，不是“最小 val_loss”
- 训练期新增的 `val_rmse_global_zspace / val_mae_global_zspace` 只表示归一化训练空间中的通用误差基线，
    不应与评估阶段物理量纲下的 `rmse_global / mae_global` 混用
- 自 2026-04-11 起，训练主流程还应自动导出 `train_plots/*`，
  至少包含 loss、monitor、validation error、learning-rate 与 dashboard 图

## 2. 数据 split 规则

评估必须与训练共享同一套数据划分边界。

最低要求：

- 使用同一份 `split_indices.npz`
- 明确当前 `split_strategy`
- 不在评估阶段重新计算 split
- 明确记录当前评估使用的是 `train / val / test` 中哪一个 split

当前仓库的默认策略是 `contiguous_v1`：

- 它按时间顺序切分滑窗样本
- 它不是随机打散窗口后的 permutation split
- 对滑窗任务而言，这一策略属于实验语义，而不是单纯实现细节

## 3. scaler 规则

评估必须加载训练阶段落盘的 scaler。

最低要求：

- `x_scaler.npz` 和 `y_scaler.npz` 来源于训练 run
- 不在评估阶段重新拟合 scaler
- 评估目录中应可追溯其来源 run

## 4. 最小评估产物

一次有效评估至少应产生：

- `metrics.yaml`
- `rmse_by_horizon.csv`
- `mae_by_horizon.csv`
- `pred_samples.npz`
- `component_metrics.csv`
- `pred_context.npz`

PR5 第一阶段后，若启用 mask-aware 评估，同一评估目录还应并行产出：

- `rmse_by_horizon_masked.csv`
- `mae_by_horizon_masked.csv`
- `component_metrics_masked.csv`

为便于调参与数值排障，同一评估目录还应并行保留 z-score 空间辅助产物：

- `rmse_by_horizon_zspace.csv`
- `mae_by_horizon_zspace.csv`
- `rmse_by_horizon_masked_zspace.csv`
- `mae_by_horizon_masked_zspace.csv`
- `pred_samples_zspace.npz`
- `component_metrics_zspace.csv`
- `component_metrics_masked_zspace.csv`

其中：

- `metrics.yaml`、`rmse/mae_by_horizon*.csv`、`pred_samples.npz`、`component_metrics*.csv`
  的主语义以物理量纲为准。
- `*_zspace.*` 只作为辅助调参与排障产物，不作为论文和系统辨识主结论依据。
- `pred_context.npz` 用于给 viz 层补充 `target_mask / sample_index / component metadata`。
- `metrics.yaml["control_readiness"]` 用于记录控制前离线筛查诊断，
  但不应被误解为闭环可用性的最终证明。
- `metrics.yaml["long_horizon_fit"]` 用于记录长期 rollout 拟合能力摘要，
  作为长期数据拟合审计与论文表图复用入口。

这些文件都应由 `src/uwnav_dynamics/eval/evaluate.py` 直接负责生成。

若启用绘图，还应额外保存：

- `plots/*.png`

这些 `plots/*` 属于 CLI / viz orchestration 触发的后处理产物，
而不是数值评估配置契约的一部分。

## 4.1 rollout layout metadata

PR4 后，`metrics.yaml` 额外记录最小 layout metadata，
用于统一 train / eval / viz 对状态布局的解释。

推荐最小结构如下：

```yaml
layout:
  schema_version: state_layout_v1
  execution:
    source: cfg_model.y_in_idx
    y_in_idx: [8, 9, 10, 11, 12, 13, 14, 15, 16]
  semantic:
    source: canonical_acc_gyro_vel_v1
    component_labels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vel_x, vel_y, vel_z]
    group_indices:
      acc: [0, 1, 2]
      gyro: [3, 4, 5]
      vel: [6, 7, 8]
    validated_against_target_cols: true
```

这里必须明确区分两层语义：

- `layout.execution`
  - 只记录 rollout 执行真源
  - 不供 viz 做物理语义分组
- `layout.semantic`
  - 只记录输出组件标签与 group 解释
  - 供指标聚合与 viz 读盘使用

`pred_samples.npz` 与 `pred_samples_zspace.npz` 的样例 schema 保持一致，
均只包含：

- `y_hat`
- `y_true`
- `logvar`

其中：

- `pred_samples.npz`
  - `y_hat / y_true / logvar` 使用物理量纲
- `pred_samples_zspace.npz`
  - `y_hat / y_true / logvar` 使用标准化空间

`pred_context.npz` 最小约定为：

- `target_mask`
- `sample_index`
- `component_labels`
- `component_display_labels`
- `component_units`

## 4.3 supervision metadata 与 dense/masked 并存策略

PR5 第一阶段后，`metrics.yaml` 额外记录最小 supervision metadata，
用于说明当前评估目录是否同时包含 dense 与 masked 指标。

推荐最小结构如下：

```yaml
supervision:
  schema_version: supervision_v1
  dense_metrics:
    present: true
  masked_metrics:
    present: true
    mask_name: target_mask
    raw_mask_source: dvl_mask
    applies_to_groups: [vel]
    group_source: canonical_acc_gyro_vel_v1
    horizon_files:
      rmse: rmse_by_horizon_masked.csv
      mae: mae_by_horizon_masked.csv
```

这里的语义边界必须保持清晰：

- dense metrics
  - 继续沿用已有 `rmse_by_horizon.csv / mae_by_horizon.csv`
  - 语义保持不变，避免破坏 PR3 / PR4 之后的稳定产物
- masked metrics
  - 由运行时 `target_mask` 单独裁决
  - 当前第一阶段主要用于 velocity 稀疏监督

补充边界：

- 对历史 `Vel_state + dvl_mask` 数据集，上述语义保持不变
- 对当前 `KF ctx v2` 这类 dense velocity supervision 数据集，`labels.npz["dvl_mask"]` 可能被有意写成全真
- 这时 masked metrics 在数值上仍然合法，但不再等价于“仅 DVL 命中时刻的速度监督指标”
- 因此如果切到 KF dense target 主线，不能再沿用旧阶段“masked metrics = 稀疏 DVL 监督表现”的解释

注意：

- `metrics.yaml` 只做记录与说明
- 运行时是否计入某个监督元素，必须由 batch 内 `target_mask` 决定
- `metrics.yaml` 不参与 train / eval 主执行路径的 mask 裁决

### 4.3.1 DVL mask artifact 兼容说明

历史与现有数据集构建流程中，
`labels.npz["dvl_mask"]` 可能出现两种等价存储形状：

- `(N, H)`
- `(N, H, 1)`

其中 `H` 为 prediction horizon。

当前训练/评估运行时统一在 supervision mask helper 中做消费端规范化：

- `(N, H)`：直接使用
- `(N, H, 1)`：压缩 singleton 末轴后再广播到 velocity semantic group

这样做的原因是：

- 不改变既有 `labels.npz` artifact 命名与主语义
- 不要求为历史实验重建数据集
- 把兼容逻辑集中在单一 helper，避免 train / eval 各自写一套 shape 特判

除上述两种形状外，其他 `dvl_mask` 形状仍视为非法输入并显式报错。

当前阶段保持：

- `pred_samples.npz` 与 `pred_samples_zspace.npz` 三键 schema 一致
- 不新增 `pred_sample_masks.npz`
- sample-level 有效性信息通过 `pred_context.npz["target_mask"]` 提供给 viz 层

## 4.4 control_readiness 诊断摘要

为避免仅凭全局 `RMSE/MAE` 直接判断“模型已可进入控制”，
当前 `metrics.yaml` 额外记录 `control_readiness` 摘要。

它的用途是：

- 辅助筛查 rollout 末步误差是否过大
- 辅助筛查误差是否随 horizon 快速放大
- 辅助筛查是否存在明显尾部误差与系统偏差

推荐最小结构如下：

```yaml
control_readiness:
  schema_version: control_readiness_v1
  intended_use: offline_screening_for_control
  closed_loop_proof: false
  physical:
    dense:
      final_step:
        rmse_global: 0.0
        mae_global: 0.0
      rollout_growth:
        rmse_last_over_first: 1.0
        mae_last_over_first: 1.0
      tail_error:
        abs_p95_global: 0.0
        abs_p99_global: 0.0
      bias:
        worst_component: acc_x
        worst_abs_bias: 0.0
    masked:
      ...
```

边界说明：

- 该摘要只反映离线 rollout 质量
- 它不能替代 controller-in-the-loop 或闭环仿真验证
- 是否进入后续控制实验，仍需结合任务目标、控制频率与闭环稳定性判断

## 4.5 long_horizon_fit 长期拟合摘要

为避免只盯住一个全局 `RMSE/MAE` 标量，
当前 `metrics.yaml` 还应额外记录 `long_horizon_fit`，
用于固化长期 rollout 的误差面积、尾段均值与增长斜率。

推荐最小结构如下：

```yaml
long_horizon_fit:
  schema_version: long_horizon_fit_v1
  intended_use: offline_long_horizon_rollout_audit
  closed_loop_proof: false
  physical:
    dense:
      late_horizon_fraction: 0.4
      late_horizon_start_step: 7
      rmse_auc_global: 0.0
      mae_auc_global: 0.0
      late_horizon_rmse_global_mean: 0.0
      late_horizon_mae_global_mean: 0.0
      rmse_step_slope: 0.0
      mae_step_slope: 0.0
      final_step_rmse_global: 0.0
      final_step_mae_global: 0.0
    masked:
      ...
```

这里的使用边界必须写清楚：

- `rmse_auc_global / mae_auc_global`
  - 用于表达整个 horizon 上的平均误差面积
- `late_horizon_*_mean`
  - 用于表达预测尾段而不是开头几步的平均误差
- `*_step_slope`
  - 用于表达误差是否随 horizon 持续上升
- `long_horizon_fit`
  - 属于长期离线拟合审计，不是闭环最终证明

## 4.2 legacy artifact fallback

对于 PR4 之前生成、缺少 `layout.semantic` 的旧评估产物，
viz 层采用统一 fallback 规则：

- 显式给出 warning
- 回退到 canonical `acc / gyro / vel` 分组
- 不修改旧 artifact 文件名与 `pred_samples.npz` schema

当前 canonical fallback 为：

- `acc = [0, 1, 2]`
- `gyro = [3, 4, 5]`
- `vel = [6, 7, 8]`

## 5. 实验记录

一次实验最少应保存以下信息：

- train yaml
- `resolved_train.yaml`
- checkpoint
- split / scaler 路径
- 评估输出目录
- 关键命令行参数

建议把这些内容视为论文附录或科研审查的最小材料包。

### 5.1 主落盘位置

若训练 run 目录为：

`run.out_dir/run.variant/`

则评估与绘图主目录通常为：

- 数值评估：`run.out_dir/run.variant/eval_test/` 或 `eval_val/`
- 图片输出：`run.out_dir/run.variant/eval_test/plots/`

例如：

```text
out/ckpts/pooltest02_s1_lstm/B0/
├── train_plots/
│   ├── training_dashboard.png
│   ├── training_loss_curve.png
│   ├── validation_monitor_curve.png
│   ├── validation_error_curve.png
│   └── learning_rate_curve.png
└── eval_test/
    ├── metrics.yaml
    ├── rmse_by_horizon.csv
    ├── rmse_by_horizon_zspace.csv
    ├── component_metrics.csv
    ├── pred_samples.npz
    ├── pred_samples_zspace.npz
    ├── pred_context.npz
    └── plots/
        ├── rmse_horizon_groups.png
        ├── mae_horizon_groups.png
        ├── long_horizon_fit_summary.png
        ├── control_readiness_summary.png
        ├── rollout_sample_000.png
        ├── pred_vs_observed_component_000.png
        └── residual_component_000.png
```

### 5.2 正式命令方式

推荐正式入口：

```bash
PYTHONPATH=src python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

若只想做数值评估，不出图：

```bash
PYTHONPATH=src python -m uwnav_dynamics.eval.evaluate \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --ckpt out/ckpts/pooltest02_s1_lstm/B0/best.pth \
  --split test
```

## 6. smoke test

smoke test 的目标是确认链路健康，而不是给出研究结论。

建议包含：

- import smoke
- CLI `--help`
- 一次最小 train config 加载
- 一组最小 pytest

smoke test 不要求：

- 大规模训练
- 最终指标比较
- 全量绘图

## 7. 正式实验

正式实验至少应满足：

- 固定 train yaml 与 run variant
- 明确记录硬件环境
- 固定 split 与 scaler
- 保留 checkpoint 与评估产物
- 保留可重复执行的命令

## 8. 当前指标边界

当前仓库已稳定支持的指标与产物更偏向：

- RMSE / MAE
- rollout 样例输出
- horizon 级别误差分析

仍在推进中的方向包括：

- 更完整的 mask-aware sample-level 可视化
- 不确定度校准指标
- 更贴近控制性能的评估指标

## 9. 状态求解器 replay 评估协议

当任务目标从“轨迹块预测”升级为“经验型系统状态求解器”时，
必须补充长序列 autoregressive replay 评估，
不能只看单窗口 horizon 指标。

### 9.1 最小 replay 产物

一次有效的 replay 评估至少应产出：

- `metrics.yaml`
- `segment_metrics.csv`
- `step_metrics.csv`
- `component_metrics.csv`
- `pred_samples.npz`
- `resolved_replay.yaml`

推荐目录约定：

```text
run.out_dir/run.variant/replay_<split>/
```

若启用批量 replay compare，还应额外产出：

- `compare_<split>/replay_model_compare.png`
- `compare_<split>/replay_long_horizon_curves.png`

### 9.2 replay 主指标

当前阶段推荐把以下指标作为状态求解器主指标：

- 全局精度：
  - `rmse_global`
  - `mae_global`
- 长期末步精度：
  - `final_step.rmse_global_mean`
  - `final_step.mae_global_mean`
  - `tail_error.final_step_abs_p95_global`
- 长期误差增长：
  - `rollout_growth.rmse_last_over_first_mean`
  - `rollout_growth.rmse_last_over_first_p95`
  - `rollout_growth.mae_last_over_first_mean`
  - `rollout_growth.mae_last_over_first_p95`
- 长线拟合趋势：
  - `long_horizon.rmse_slope`
  - `long_horizon.log_rmse_slope`
- 尾部风险：
  - `tail_error.abs_p95_global`
  - `tail_error.abs_p99_global`
- 阈值失效与生存：
  - `long_horizon.thresholds.rmse`
  - `long_horizon.thresholds.abs_error`
  - `long_horizon.time_to_threshold.rmse.failure_rate`
  - `long_horizon.time_to_threshold.rmse.breach_step_mean`
  - `long_horizon.time_to_threshold.abs_error.failure_rate`
  - `long_horizon.time_to_threshold.abs_error.breach_step_mean`
- 系统偏差：
  - `bias.worst_component`
  - `bias.worst_abs_bias`
- 稳定性与可运行性：
  - `segment_count`
  - `total_steps`
  - `nonfinite_trigger_count`
  - `robustness.finite_pass_rate`

解释边界：

- `rmse_global / mae_global`
  - 说明整体拟合误差
  - 但不能单独代表长时稳定性
- `final_step.*`
  - 更贴近“累计递推到段尾以后还能否保持可用”
- `rollout_growth.*`
  - 用于识别误差是否随递推快速放大
- `long_horizon.*slope`
  - 用于区分“整体误差缓慢增长”与“误差随 step 明显抬升”
- `tail_error.*`
  - 用于识别长尾风险和坏 case
- `time_to_threshold.*`
  - 用于识别“什么时候开始明显失控”
  - 当前同时保留：
    - `failure_rate`：有多少段在本段长度内触发阈值失效
    - `breach_step_mean`：已失效段平均在第几步越阈
- `worst_abs_bias`
  - 用于识别是否存在明显系统偏差
- `nonfinite_trigger_count`
  - 只要大于 0，就不应进入主线候选

### 9.2.1 当前长线阈值默认值

当前 replay 主线默认服务“体速度对 DVL 观测”的统一口径，
因此批量 replay compare 默认采用：

- `rmse_threshold = 0.05`
- `abs_error_threshold = 0.10`

单位均为 `m/s`。

说明：

- 这组默认值是当前速度长期拟合筛选用的工程阈值，不是普适真理；
- 若未来 replay 目标改成其他状态量，必须在 launcher 中显式覆盖阈值，并在 `manifest.yaml` 中留痕。

### 9.2.2 `step_metrics.csv` 语义

`step_metrics.csv` 用于承载长线逐步统计，不要求与训练 horizon artifact 完全同构。

当前最小列约定：

- `step`
- `active_segments`
- `value_count`
- `rmse_global`
- `mae_global`
- `abs_p50_global`
- `abs_p95_global`
- `rmse_survival_rate`
- `abs_survival_rate`
- `rmse_threshold`
- `abs_error_threshold`

其中：

- `rmse_global / mae_global / abs_p95_global`
  - 表示相对 step `k` 上的跨段聚合误差
- `rmse_survival_rate / abs_survival_rate`
  - 表示到 step `k` 为止仍未越阈的 segment 比例
- 该 artifact 主要供 replay compare 图与长线审计使用
  - 不替代 `segment_metrics.csv`

### 9.3 多方案统一汇总与排行

若要比较多个候选模型，
必须使用统一 replay 协议落盘：

```text
work_dir/
  manifest.yaml
  summary.csv
  ranking.csv
  runs/<candidate_name>/...
```

其中：

- `summary.csv`
  - 保存所有候选的原始 replay 主指标与路径
- `ranking.csv`
  - 按统一排行协议输出“谁更好”的排序结果
- `manifest.yaml`
  - 固定本次批量比较使用的 split、最小 segment 长度、保存样例数量与排行协议

对于 `transition_replay_matrix` 的配置文件，还需要额外固定路径解析约定：

- 手写 launcher 配置可以继续使用 repo-root 相对路径
  - 例如 `configs/train/...`、`out/replay_matrix/...`
- 长时长 replay 优先使用秒级字段：
  - `min_seconds: 50`
  - `max_seconds_per_segment: 50`
  - `dt_s: 0.01`
- 不要用 `min_steps: 50` 表达 50 秒；在 100 Hz 数据中这只表示 0.5 秒
- 由 `server_pipeline` 自动生成的 replay matrix 配置，
  允许使用相对“该配置文件所在目录”的 `../..` 路径
- 运行时必须兼容这两类来源
  - 否则把服务器生成配置带回本地或跨目录重放时，
    `train_yaml / ckpt / work_dir` 很容易被错误解析到仓库外部路径，
    导致 replay / ranking 看起来“全部失效”，但实际是配置文件未被正确读取

当前推荐排行协议：

- 先过硬门槛：
  - `segment_count > 0`
  - `nonfinite_trigger_count == 0`
- 再按以下主指标做加权排行，统一采用“越小越好”：
  - `rmse_global`
  - `mae_global`
  - `final_step.rmse_global_mean`
  - `rollout_growth.rmse_last_over_first_p95`
  - `tail_error.abs_p95_global`
  - `long_horizon.time_to_threshold.rmse.failure_rate`
  - `tail_error.abs_p99_global`
  - `long_horizon.time_to_threshold.abs_error.failure_rate`
  - `bias.worst_abs_bias`

注意：

- 该排行协议服务于“离线 replay 筛选”
- 它不是闭环控制最终结论
- 若未来进入 controller replay / simulator loop，
  应继续并行记录时延、循环频率、失败步数等在线指标

### 9.4 当前推荐可视化

当前阶段 replay model selection 至少应并行输出两张图：

1. `replay_model_compare.*`
   - 用于集中查看：
     - `rmse_global`
     - `final_step`
     - `growth`
     - `threshold failure`
   - 当前默认 2x2 四窗，避免单图过密

2. `replay_long_horizon_curves.*`
   - 用于查看随 step 推进的：
     - `rmse_global`
     - `abs_p95_global`
     - `rmse_survival_rate`
     - `abs_survival_rate`
   - 当前默认 2x2 四窗

离线评估默认图包还应包含三张 50s 长时序图：

- `prediction_trace_acc_axes.*`
- `prediction_trace_gyro_axes.*`
- `prediction_trace_vel_axes.*`

每张图只包含 X/Y/Z 三个共享 x 轴子窗，图例放在数据区域之外，
用于替代过去默认输出的大量 0.1s 短窗口样例图。

判断原则：

- summary 图用于快速筛掉明显差的候选
- long-horizon 曲线图用于判断模型是“整体更准”还是“后段更稳”
- 若 summary 排名接近，但 survival 曲线后段明显分叉，应优先相信 survival 结果
