# 实验评估规范

## 1. 目标

本规范用于统一：

- 训练后应保留哪些产物
- 评估应基于哪些 split 与 scaler
- smoke test 与正式实验如何区分
- 结果如何记录，便于复现与论文写作

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

PR5 第一阶段后，若启用 mask-aware 评估，同一评估目录还应并行产出：

- `rmse_by_horizon_masked.csv`
- `mae_by_horizon_masked.csv`

其中前四项是 dense 主产物；
masked 两项属于 PR5 第一阶段新增的并行数值评估产物。
这些文件都应由 `src/uwnav_dynamics/eval/evaluate.py` 直接负责生成。

若启用绘图，还应额外保存：

- `plots/*.png` 或 `plots/*.pdf`

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

`pred_samples.npz` 的 schema 在 PR4 中保持不变，
仍只包含：

- `y_hat`
- `y_true`
- `logvar`

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

注意：

- `metrics.yaml` 只做记录与说明
- 运行时是否计入某个监督元素，必须由 batch 内 `target_mask` 决定
- `metrics.yaml` 不参与 train / eval 主执行路径的 mask 裁决

当前第一阶段保持：

- `pred_samples.npz` 三键 schema 不变
- 不新增 `pred_sample_masks.npz`
- sample-level masked visualization 留到后续 patch 再接入

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
