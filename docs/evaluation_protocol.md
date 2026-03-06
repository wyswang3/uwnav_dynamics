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
- 不在评估阶段重新计算 split
- 明确记录当前评估使用的是 `train / val / test` 中哪一个 split

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

若启用绘图，还应额外保存：

- `plots/*.png` 或 `plots/*.pdf`

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

- mask-aware 指标
- 不确定度校准指标
- 更贴近控制性能的评估指标
