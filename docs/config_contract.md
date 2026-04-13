# 配置契约说明

## 1. 配置系统总览

本仓库的训练与评估都依赖同一套配置契约。目标不是“让 YAML 尽量宽松”，而是让一次实验的模型结构、运行参数与产物路径都能被明确追溯。

当前主链路如下：

`train yaml -> canonical parser -> TrainYamlConfig -> CLI override -> runtime reconcile -> resolved_train.yaml -> eval`

其中：
- `configs/train/*.yaml` 是研究者直接编辑的实验定义；
- `src/uwnav_dynamics/train/config.py` 负责将 YAML 严格解析为强类型配置；
- `src/uwnav_dynamics/train/runtime.py` 负责运行期 override 与最终快照落盘；
- `src/uwnav_dynamics/eval/config.py` 只补评估运行时字段，不重新解释模型结构。

这里的“评估运行时字段”仅指数值评估本身需要的参数，例如：
- checkpoint 路径
- 评估 split
- batch size
- split / scaler artifact 路径

绘图相关参数不属于 `EvalConfig` 的职责范围。
它们应由 `cli/eval.py`、`cli/pipeline.py` 与 `viz/*` 共同编排，
而不是重新塞回数值评估配置契约。

## 2. train / eval 配置一致性的设计原则

### 2.1 单一真源

训练侧的 train yaml 是模型结构的唯一真源。评估侧必须复用训练侧解析结果，而不是维护另一份“近似等价”的 model builder。

这样设计的原因是：
- checkpoint 能够 `load_state_dict(strict=True)`，并不代表结构语义完全一致；
- blocks 开关、索引切片、额外 head、输入替换策略等都属于模型拓扑的一部分；
- 若 train 与 eval 各自解释 YAML，最容易出现“能运行、但结果已偏离训练语义”的隐性错误。

### 2.2 strict schema

`train/config.py` 默认拒绝未知字段。对科研代码而言，这种策略比宽松容错更合适，因为：
- schema 漂移应在配置加载阶段暴露；
- 错误越早失败，越容易定位；
- 对未来开源、复现实验和论文审查更友好。

## 3. canonical parser 的作用

`src/uwnav_dynamics/train/config.py` 的职责是：
- 把原始 YAML 解析为 `TrainYamlConfig`；
- 对 `run/data/model/rollout/loss/train` 建立统一、强类型、可校验的契约；
- 对 blocks 配置执行显式解析，避免默认值和半写配置造成 silent bug；
- 为 train 和 eval 提供共同的模型配置对象。

它不负责的事情包括：
- 运行期设备回落；
- CLI override 的最终裁决；
- split/scaler 产物是否已存在；
- 评估输出目录等评估运行时参数。

这些逻辑分别由 runtime 或 eval 侧处理。

其中需要额外强调的是：
- `eval/config.py` 只负责数值评估运行时；
- `cli/eval.py` / `cli/pipeline.py` 负责是否进入 viz 阶段的 orchestration；
- `viz/eval/*` 负责从已落盘 artifact 读盘绘图。

## 3.1 rollout layout contract 的分层

PR4 之后，仓库对 rollout layout contract 做了明确分层：

- `execution layout contract`
  - 唯一执行真源是 `cfg_model.y_in_idx`
  - 只负责：从 `X` 的最后一个历史时刻提取 `y0`
  - 只服务于 train / eval rollout 数值路径
- `semantic output layout contract`
  - 只负责：9 维输出对应的组件标签与 `acc / gyro / vel` 分组语义
  - 只服务于指标聚合、`metrics.yaml` metadata 与 viz 读盘解释

这两个 contract 必须保持分层清晰：

- train / eval 不得把物理语义分组反向用于执行索引裁决
- viz 不得读取 `cfg_model.y_in_idx` 来推断画图分组
- `target_cols` 不得成为主执行路径的第二真源

当前实现中：

- `src/uwnav_dynamics/models/utils/execution_layout.py`
  负责 execution layout 校验与 `y0` 提取
- `src/uwnav_dynamics/models/utils/semantic_output_layout.py`
  负责 semantic metadata、分组解释与旧 artifact fallback

## 3.2 `target_cols` 的职责边界

`labels.npz` 与 `meta.yaml` 中的 `target_cols` 仍然保留，
但它在 PR4 后只承担以下职责：

- 作为 semantic output layout 的校验旁证
- 作为 artifact 审计信息的辅助来源
- 帮助确认当前数据集的监督目标顺序没有漂移

它不承担以下职责：

- 不裁决 `y0` 的执行索引
- 不替代 `cfg_model.y_in_idx`
- 不成为 viz 主逻辑的主消费契约

## 3.3 dense supervision 与 masked supervision 的边界

PR5 第一阶段之后，仓库需要明确区分两类“监督存在”：

- `dense supervision`
  - 指 `Y` 中数值上完整存在的监督张量
  - 例如经过 forward-fill 后的 `Vel*_state_mps` 仍会保留在 `Y` 中
- `masked supervision`
  - 指训练 / 评估在运行时真正计入 loss 或 metric 的有效监督元素
  - 是否有效必须由 batch 中的 `target_mask` 决定

这里必须强调：

- forward-fill 后的 velocity target 仍可保留为数值对齐结果
- 但它不再自动等同于“有效监督”
- velocity 维是否计入 loss / metric，只能由 `target_mask` 裁决

PR5 当前 runtime mask contract 为：

- 训练阶段：
  - `labels.npz["dvl_mask"]` 作为原始 sparse supervision 来源之一
  - `train/data_pipeline.py` 基于 `dvl_mask + semantic output layout` 构造 `target_mask`
  - DataLoader batch 以 `(X, Y, target_mask)` 形式传入 trainer
- 评估阶段：
  - `evaluate.py` 同样从 `labels.npz["dvl_mask"]` 构造 `target_mask`
  - dense / masked 指标共同基于同一批 `y_hat / y_true / target_mask`

边界要求：

- `target_mask` 是 train / eval 运行时唯一的 mask 执行真源
- `meta.yaml` 与 `metrics.yaml` 只做记录、审计与可视化说明
- `meta.yaml` / `metrics.yaml` 不参与运行时 mask 裁决

其中：

- `execution layout contract`
  - 继续只负责 rollout 执行索引
  - 与 supervision mask 裁决无关
- `semantic output layout contract`
  - 只负责告诉系统哪些输出维属于 `acc / gyro / vel`
  - 供 `target_mask` 构造、指标分组与 viz 解释使用

## 4. frozen 配置与 override 规则

### 4.1 哪些对象应该 frozen

与模型结构和训练超参相关、且本质上是“纯值对象”的 dataclass 应被冻结，例如：
- `ThrusterLagConfig`
- `HydroSSMConfig`
- `DampingHeadConfig`
- `UncertaintyHeadConfig`
- `S1BlocksConfig`
- `S1PredictorConfig`
- `TrainConfig`

冻结的目的不是增加语法负担，而是明确以下边界：
- 配置对象是声明式契约，不是运行期状态容器；
- train / eval 不应通过 in-place 修改共享配置对象；
- 配置变化必须经由显式 override 或 `replace()`，便于审计与回溯。

### 4.2 哪些对象不应 frozen

以下对象属于运行期状态，不应冻结：
- `nn.Module`
- 优化器、AMP scaler
- DataLoader、Tensor
- 任何与当前 epoch、设备状态、梯度状态相关的对象

### 4.3 override 规则

运行期 override 统一通过 `dataclasses.replace()` 实现，而不直接改写原对象。这样可以保留：
- 原始 YAML 配置的稳定语义；
- CLI 注入参数的可追溯性；
- `resolved_train.yaml` 中“最终执行配置”的完整快照。

## 5. `resolved_train.yaml` 的字段说明与科研用途

`resolved_train.yaml` 是训练阶段的最终快照。它记录的是“实际执行时的配置”，而不是“研究者最初写下的 YAML”。

### 5.1 顶层字段

顶层保留以下 canonical 字段：
- `run`
- `data`
- `model`
- `rollout`
- `loss`
- `train`

这些字段应与训练实际使用的配置一一对应。

### 5.2 `_meta` 字段

`_meta` 当前至少包含：
- `schema_version`: 当前为 `train_resolved_v1`
- `source_yaml`: 原始 train yaml 路径
- `cli_overrides`: 本次训练显式传入的 CLI 覆盖项
- `requested_device`: 用户请求的设备
- `runtime_device`: 实际运行设备
- `run_dir`: 最终运行目录
- `split_indices_path`: 本次训练/评估共享的 split 产物路径
- `split_strategy`: 当前 split 语义版本
- `x_scaler_path`: 输入 scaler 路径
- `y_scaler_path`: 输出 scaler 路径

### 5.3 科研用途

这些信息有三个直接用途：
- 复现实验时可直接核对最终执行配置，而不是推测 CLI 是否覆盖了 YAML；
- 团队交接时可快速定位一次训练对应的 split/scaler/ckpt 产物；
- 未来开源或论文补充材料中，可以更清晰地说明实验环境与配置来源。

当前推荐将 `split_strategy` 视为实验语义的一部分，而不是实现细节。
对于滑窗数据集，仓库当前默认使用 `contiguous_v1`，即按时间顺序切分窗口索引；
这样做是为了降低相邻窗口跨 split 带来的时间泄漏风险。

## 6. `out_dir / variant` 契约及反例

### 6.1 正确契约

运行目录的 canonical 约定是：

`run_dir = run.out_dir / run.variant`

因此：
- `run.out_dir` 只应表示实验根目录；
- `run.variant` 只应表示当前变体名称；
- 程序负责把二者拼成最终运行目录。

### 6.2 反例

以下写法会导致重复嵌套：

```yaml
run:
  out_dir: out/ckpts/pooltest02/B0_baseline
  variant: B0_baseline
```

它会生成：

```text
out/ckpts/pooltest02/B0_baseline/B0_baseline
```

这会给 checkpoint、split、scaler、eval 输出的定位带来额外歧义。

### 6.3 建议写法

应改为：

```yaml
run:
  out_dir: out/ckpts/pooltest02
  variant: B0_baseline
```

这样路径契约简洁、稳定，也更适合后续自动化管线与审查。

### 6.4 launcher 相对路径锚点

对于 `train_matrix / server_pipeline / transition_replay_matrix` 这类 launcher 配置，
当前路径解析约定需要额外固定为：

- 普通仓库内路径
  - 例如 `configs/...`、`out/...`、`data/...`
  - 继续按 repo-root 相对路径解释
- 显式包含 `..` 的路径
  - 视为“相对当前配置文件所在目录”
  - 典型用途是：
    - 把 launcher 配置放到临时目录或服务器工作目录下
    - 仍希望 `work_dir / source_summary_csv / replay_out_dir / generated train yaml run.out_dir`
      跟着该配置目录一起迁移

这样做的目的有两个：

- 避免把原本想写到配置旁边的产物错误展开到系统根目录或仓库外层目录
- 避免 server 侧自动生成配置带回本地后，因为工作目录不同而出现“文件找不到 / 产物写错位置”
