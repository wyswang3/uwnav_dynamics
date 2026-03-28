# 工程升级路线

## 1. 目标

本路线图只讨论工程实现层面的升级，不讨论更长期的数学建模扩展。
当前优先级来自“先确保结果可信，再逐步提升研究表达能力”的原则。

## 2. PR2：split / scaler 单一真源（已完成）

### 状态

- 已完成，收口日期：2026-03-06

### 目标

- 统一 train / eval 对 split 与 scaler 的读取路径
- 避免重新计算导致的数据泄漏或边界不一致

### 影响模块

- `src/uwnav_dynamics/train/data.py`
- `src/uwnav_dynamics/train/data_pipeline.py`
- `src/uwnav_dynamics/dataset/split.py`
- `src/uwnav_dynamics/dataset/normalize.py`
- `src/uwnav_dynamics/eval/evaluate.py`

### 验收标准

- 训练与评估使用同一份 `split_indices.npz`
- scaler 只由 train split 拟合
- 相关 smoke test 与无泄漏测试通过

### 已完成内容

- 将 canonical split 语义明确为 `contiguous_v1`
- 在 `split_indices.npz` 中写入 `split_strategy` 元数据
- 训练阶段创建 run-scoped split / scaler artifact，评估阶段只复用
- 将 `split_strategy` 写入 `resolved_train.yaml`
- 增加 split contiguous / no-leak / eval artifact reuse 自动化测试

## 3. PR4：rollout 索引契约

### 状态

- 已完成，收口日期：2026-03-06

### 目标

- 去除 rollout 相关硬编码索引
- 统一 train / eval / viz 对状态布局的解释

### 影响模块

- `src/uwnav_dynamics/models/utils/rollout.py`
- `src/uwnav_dynamics/models/utils/execution_layout.py`
- `src/uwnav_dynamics/models/utils/semantic_output_layout.py`
- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/eval/evaluate.py`
- `src/uwnav_dynamics/viz/eval/*`

### 验收标准

- rollout 执行路径只依赖 `cfg_model.y_in_idx`
- 非默认列顺序测试通过
- train / eval 对相同权重和输入给出一致 rollout 结果
- `metrics.yaml` 写出最小 layout metadata
- 旧 artifact 缺少 layout metadata 时，viz 统一 warning + fallback

### 已完成内容

- 新增 execution layout helper，统一 `y0` 提取与索引合法性校验
- 新增 semantic output layout helper，统一组件标签、group 语义与 legacy fallback
- 训练 loss path 改为使用 `cfg_model.y_in_idx` 提取 `y0`
- `evaluate.py` 改为写出 `metrics.yaml.layout.execution / semantic`
- horizon / rollout / pred-vs-observed / model-compare 四条 viz 路径统一消费 `layout.semantic`
- `pred_samples.npz` schema 保持不变

## 4. PR3：eval-viz 解耦（已完成）

### 状态

- 已完成，收口日期：2026-03-06

### 目标

- 把数值评估与绘图职责拆开
- 让评估结果可在无绘图库依赖或批处理环境中稳定运行

### 影响模块

- `src/uwnav_dynamics/eval/evaluate.py`
- `src/uwnav_dynamics/cli/eval.py`
- `src/uwnav_dynamics/cli/pipeline.py`
- `src/uwnav_dynamics/viz/eval/*`

### 验收标准

- 评估主流程不再直接依赖绘图执行
- `metrics.yaml`、CSV、`pred_samples.npz` 产物保持稳定
- 单独绘图 smoke test 通过

### 已完成内容

- `evaluate.py` 收口为纯数值评估入口，内部 `--plots` 路径改为显式弃用
- `EvalConfig` 只保留数值评估运行时字段，不再承载绘图参数
- `cli/eval.py` 负责正式的 eval -> viz orchestration，并在绘图失败时保留数值 artifact
- `cli/pipeline.py` 改为通过 `cli/eval.py` 完成 train -> eval -> viz 串联
- 增加 CLI 编排、弃用路径与失败保留语义测试

## 5. PR5：mask-aware 训练评估

### 状态

- 第一阶段已完成，收口日期：2026-03-06

### 目标

- 将 DVL 等稀疏观测的有效性显式纳入 loss 和 metric
- 让“缺测”与“零值”在训练语义上区分开

### 影响模块

- `src/uwnav_dynamics/preprocess/build_dataset.py`
- `src/uwnav_dynamics/train/data_pipeline.py`
- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/models/losses/*`
- `src/uwnav_dynamics/eval/evaluate.py`
- `src/uwnav_dynamics/viz/eval/plot_horizon_metrics.py`
- `src/uwnav_dynamics/viz/eval/plot_model_compare.py`
- `src/uwnav_dynamics/supervision_mask.py`

### 验收标准

- 训练 batch 以 `(X, Y, target_mask)` 进入 trainer 主路径
- `gaussian_nll_diag_masked()` 在 mask 全真时与 dense 版本一致
- 评估同时落盘 dense 与 masked horizon artifact
- horizon / model-compare 图在 masked CSV 存在时并行输出 masked 图
- `pred_samples.npz` 三键 schema 保持不变

### 已完成内容

- 新增 `supervision_mask.py`，统一将 `dvl_mask + semantic layout` 构造成 `target_mask`
- `train/data_pipeline.py` 将 `target_mask` 作为 batch 运行时真源接入训练主路径
- `trainer.py` / `run_train.py` 支持 dense 与 masked supervision 两条 loss 路径
- `nll.py` 新增 `gaussian_nll_diag_masked()`，并对“有效元素数为 0”采取 fail-fast
- `evaluate.py` 同时落盘：
  - `rmse_by_horizon.csv`
  - `mae_by_horizon.csv`
  - `rmse_by_horizon_masked.csv`
  - `mae_by_horizon_masked.csv`
- `metrics.yaml` 新增最小 `supervision` metadata
- `plot_horizon_metrics.py` 与 `plot_model_compare.py` 支持 dense / masked 图并行存在
- `plot_rollout_samples.py` 现可从 `pred_context.npz["target_mask"]` 读取 sample-level 有效性，
  并在 rollout 样例图中标出 masked-out 目标位置

### 当前边界

- 第一阶段不新增 `pred_sample_masks.npz`
- 更细粒度的 sample-level masked visualization 仍可继续扩展，但当前主链路已支持
  基于 `pred_context.npz["target_mask"]` 的 rollout 样例标注
- `meta.yaml` 与 `metrics.yaml` 只做记录，不参与运行时 mask 裁决

## 6. 工程执行原则

- 每个 PR 只收敛一个清晰问题
- 优先保留现有 CLI 与 artifact 契约
- 每次升级都需要最小 smoke test
- 设计文档必须与代码一同更新
