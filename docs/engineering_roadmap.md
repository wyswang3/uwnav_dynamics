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

### 说明文档

- [pr2_split_scaler_single_source.md](/home/wys/uwnav_dynamics/docs/pr2_split_scaler_single_source.md)

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

### 目标

- 将 DVL 等稀疏观测的有效性显式纳入 loss 和 metric
- 让“缺测”与“零值”在训练语义上区分开

### 影响模块

- `src/uwnav_dynamics/preprocess/build_dataset.py`
- `src/uwnav_dynamics/train/data_pipeline.py`
- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/models/losses/*`
- `src/uwnav_dynamics/eval/evaluate.py`

### 验收标准

- 数据加载器能输出 mask
- velocity 相关监督支持 masked 指标
- 评估同时提供 dense 与 masked 指标过渡结果

## 6. 工程执行原则

- 每个 PR 只收敛一个清晰问题
- 优先保留现有 CLI 与 artifact 契约
- 每次升级都需要最小 smoke test
- 设计文档必须与代码一同更新
