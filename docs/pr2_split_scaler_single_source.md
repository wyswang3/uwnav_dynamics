# PR2：split / scaler 单一真源

更新时间：2026-03-06

## 1. 问题定义

PR2 的目标是把 train / eval 对数据划分与归一化产物的依赖路径收敛成单一真源，避免以下两类问题：

- 评估阶段重新计算 split，导致 train / eval 边界不一致；
- scaler 在非 train split 上被重新拟合，导致结果不可比或引入泄漏。

对于滑窗数据集，单纯保证“索引不重叠”还不够；如果把时间上相邻的窗口随机打散到不同 split，中间仍可能共享大量时间支撑区间。因此本仓库当前将 split 语义明确收敛为 `contiguous_v1`。

## 2. 本次 patch 范围

### 2.1 canonical split 语义

- `src/uwnav_dynamics/dataset/split.py`
- 新增 `DEFAULT_SPLIT_STRATEGY = "contiguous_v1"`
- `make_split_indices()` 明确为按时间顺序切分的 contiguous split
- `save_split_indices()` 在 `split_indices.npz` 中写入 `split_strategy`
- `load_split_indices()` 对旧 artifact 缺少策略元数据的情况保持兼容，并给出 warning

### 2.2 legacy loader 收口

- `src/uwnav_dynamics/train/data.py`
- 删除本地重复 split 逻辑，统一改为调用 `dataset.split.make_split_indices()`
- 将 `build_loaders()` 标记为 compatibility-only helper，提示正式 train / eval 流程应走 `train.data_pipeline.prepare_train_data()`

### 2.3 run-scoped artifact 复用

- `src/uwnav_dynamics/train/data_pipeline.py`
- 若运行目录下已存在 `split_indices.npz`，则直接加载，不再重算
- scaler 仍只对 train split 拟合；val / test 只做 transform
- 创建新 split 时把 `strategy=contiguous_v1` 打印到日志，便于追踪

### 2.4 实验追溯增强

- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/train/runtime.py`
- `resolved_train.yaml` 的 `_meta` 新增 `split_strategy`
- 这样一次训练的 ckpt、split、scaler 与 split 语义可以在同一快照中追溯

### 2.5 文档契约同步

- `docs/config_contract.md`
- `docs/evaluation_protocol.md`
- 明确 `split_strategy` 是实验语义的一部分，而不是实现细节
- 明确评估必须复用训练阶段落盘的 split / scaler artifact

## 3. 验收证据

以下自动化测试直接覆盖了 PR2 的核心验收点：

- `tests/test_split_no_leak.py::test_split_indices_roundtrip_are_disjoint_and_contiguous`
  验证 split 不重叠、语义为 contiguous、`split_strategy` 元数据成功落盘。
- `tests/test_split_no_leak.py::test_scaler_fit_only_train_subset_and_shared_indices`
  验证 scaler 只由 train split 拟合，val / test 的 transform 不会修改 scaler。
- `tests/test_split_no_leak.py::test_eval_reuses_train_split_and_scaler_artifacts`
  验证评估阶段复用训练阶段生成的 `split_indices.npz`、`x_scaler.npz`、`y_scaler.npz`，不会重写这些 artifact。
- `tests/test_train_runtime.py::test_runtime_helpers_capture_effective_config`
  验证 `resolved_train.yaml` 正确记录 `split_strategy`。

## 4. 已完成后的语义边界

- 当前 canonical split 是 `contiguous_v1`，因此 `seed` 仅保留 API 兼容意义，不再影响切分结果。
- 旧版 `split_indices.npz` 若没有 `split_strategy` 字段，当前实现仍允许加载，但会给出 warning。
- PR2 只解决 split / scaler 单一真源问题，不涉及 rollout 索引契约、eval-viz 解耦或 mask-aware 训练。

## 5. 相关文档

- `docs/project_status.md`
- `docs/engineering_roadmap.md`
- `docs/config_contract.md`
- `docs/evaluation_protocol.md`
