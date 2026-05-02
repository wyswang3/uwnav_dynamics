# 最终选模与论文产物契约 V1

更新时间：2026-05-02

当前最终选型已收口到 `StepBase s11 / step_b0_grouped_tb_seed11`。
具体技术结论、指标和图表位置见
[current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)。
本文档只说明 `final_selection.csv` 与相关 artifact 的契约。

## 1. 目标

本契约用于解决两个问题：

1. 多方案训练、离线评估、replay 排名之后，哪一个候选应被视为当前最终模型；
2. 哪些表格和图目录应被论文正文、附录和技术汇报优先复用。

当前实现的收口思路是：

- 训练矩阵保留 `summary.csv`
- replay 批量评估保留 `ranking.csv`
- server pipeline 额外生成：
  - `final_selection.csv`
  - `paper_artifact_manifest.yaml`

## 2. `final_selection.csv`

位置：

```text
out/server_pipeline/<variant>/final_selection.csv
```

职责：

- 把 `train_matrix/summary.csv` 与 `replay_matrix/ranking.csv` 合并到同一张最终选模表
- 保留训练期指标与 replay 指标，避免最终结论只看单一阶段
- 按 `selection_scope` 标记各自 scope 下的 winner

关键字段：

- `selection_scope`
- `is_scope_winner`
- `selection_pass`
- `overall_rank`
- `overall_rank_score`
- `train_*`
- `replay_*`

注意：

- 不同 `selection_scope` 之间不能直接做全局绝对排行；
- 当前推荐以 `quality_step_v1` 对应 replay scope 的 winner 作为 solver 主候选；
  2026-05-02 的 winner 是 `step_b0_grouped_tb_seed11`。

## 3. 代表性样例产物

### 3.1 eval

位置：

```text
run_dir/eval_test/pred_samples.npz
run_dir/eval_test/pred_context.npz
run_dir/eval_test/pred_sample_manifest.csv
```

当前样例不再是“前 N 个窗口”，而是代表性窗口：

- `best_rmse`
- `median_rmse`
- `worst_rmse`
- `worst_final_step`
- `worst_tail_p95`

### 3.2 replay

位置：

```text
run_dir/replay_test/pred_samples.npz
run_dir/replay_test/pred_context.npz
run_dir/replay_test/pred_sample_manifest.csv
```

当前样例不再是“前 N 个 segment”，而是代表性 segment：

- `best_rmse`
- `median_rmse`
- `worst_rmse`
- `worst_final_step`
- `worst_growth`

## 4. `paper_artifact_manifest.yaml`

位置：

```text
out/server_pipeline/<variant>/paper_artifact_manifest.yaml
```

职责：

- 固化最终建议复用的表格入口
- 固化 train compare / replay compare 图目录
- 标记每个 selection scope 的 winner 对应评估目录与 replay 目录

主要块：

- `tables.phase_status_csv`
- `tables.final_selection_csv`
- `tables.train_matrix_summaries`
- `tables.replay_rankings`
- `figures.train_compare_dirs`
- `figures.replay_compare_dirs`
- `scope_winners`

## 5. 推荐复用顺序

论文与技术汇报推荐按下列顺序取图和表：

1. `final_selection.csv`
2. `paper_artifact_manifest.yaml`
3. `train_matrix/*/summary.csv`
4. `replay_matrix/*/ranking.csv`
5. winner 对应的 `eval_test/plots/*`
6. winner 对应的 `replay_test/pred_sample_manifest.csv` 与样例图

## 6. 当前边界

- `final_selection.csv` 是离线筛选真源，不等价于闭环最终证明；
- 代表性样例用于增强图包代表性，不替代全量统计；
- 若后续更换 solver 主线或增加新 replay scope，应同步扩展本契约。
