# 项目当前状态

更新时间：2026-03-06

## 1. 当前阶段摘要

项目当前处于“工程主链路收口与科研化整理”阶段。

当前目标不是继续扩展功能面，而是先把以下基础打牢：

- 多频异步数据链路可以稳定跑通
- train / eval / artifact 契约保持一致
- 关键冒烟测试可在仓库根目录直接执行
- 文档能够准确说明当前实现与未来路线

## 2. 当前 baseline

当前 baseline 形态：

- 主时间轴：100 Hz
- 输入：`PWM 8 + IMU 6 + Vel_state 3 + Power 8`
- 输出：`Acc 3 + Gyro 3 + Vel_state 3`
- 模型：`S1Predictor`
- rollout：`y_hat = y0 + cumsum(dY)`
- loss：对角高斯 NLL baseline

## 3. 当前验证结果

截至当前文档版本，最重要的回归结果是：

- `PR6` 后，`pytest -q` 的最小收集与对齐边界测试恢复
- `PR1` 后，train / eval 配置 parity 已由自动化测试覆盖
- `PR2` 后，train / eval 对 split / scaler artifact 的共享与复用已由自动化测试覆盖
- `PR3` 后，数值评估与绘图编排职责已拆开，CLI 级编排与失败保留语义已有测试覆盖
- `PR4` 后，train / eval / viz 的状态布局解释已统一收口为 execution / semantic 两层 contract
- `PR5` 第一阶段后，mask-aware supervision 已进入训练 loss 与 eval masked metrics 主路径
- `resolved_train.yaml` 已能记录训练最终执行配置、关键 artifact 路径与 `split_strategy`

建议把以下类型的测试视为当前最小健康信号：

- import smoke
- CLI `--help`
- layout / runtime config tests
- eval config parity tests
- split / scaler no-leak tests

## 4. 已完成升级

### 4.1 PR6：对齐边界与 pytest 冒烟修复

已完成内容：

- 修复多频对齐中的浮点时间边界问题
- 恢复 `pytest` 对正式测试目录的稳定收集
- 增加最小对齐回归测试

### 4.2 PR1：配置契约收口

已完成内容：

- 明确 `train/config.py` 为 canonical parser
- eval 侧复用训练侧完整配置解析结果
- 冻结纯配置 dataclass
- 扩展 `resolved_train.yaml`
- 增加 train / eval parity test

### 4.3 PR2：split / scaler 单一真源

已完成内容：

- 明确 `contiguous_v1` 为当前 canonical split 语义
- `split_indices.npz` 写入 `split_strategy` 元数据
- 训练运行目录内的 split / scaler artifact 由 train 创建后供 eval 复用
- legacy `train.data.build_loaders()` 复用 canonical split builder，避免语义漂移
- `resolved_train.yaml` 记录 `split_strategy`
- 增加 split contiguous / no-leak / eval artifact reuse 测试

详细说明见：

- [pr2_split_scaler_single_source.md](/home/wys/uwnav_dynamics/docs/pr2_split_scaler_single_source.md)

### 4.4 PR3：eval-viz 解耦

已完成内容：

- `evaluate.py` 收敛为纯数值评估与 artifact 落盘，不再直接执行绘图
- `EvalConfig` 收缩为数值评估运行时配置，不再承载 plot runtime fields
- `cli/eval.py` 成为正式用户入口，负责串联“数值评估 -> viz 出图”
- `cli/pipeline.py` 通过 `cli/eval.py` 实现 `train -> eval -> viz` 组合调用
- `evaluate.py --plots` 改为显式弃用，提示用户迁移到 CLI 入口
- 增加 CLI 编排、显式弃用与“绘图失败但数值 artifact 保留”测试

### 4.5 PR5：mask-aware 训练评估第一阶段

已完成内容：

- 新增 `target_mask` 运行时链路，训练 batch 现在可携带 `(X, Y, target_mask)`
- DVL velocity 稀疏监督已通过 masked NLL 进入训练主路径
- 评估目录同时保留 dense 与 masked horizon artifact
- horizon / model-compare 图在 masked CSV 存在时可并行输出 masked 图
- `pred_samples.npz` 仍保持 `y_hat / y_true / logvar` 三键 schema

### 4.6 PR5 兼容性热修复：`dvl_mask` shape 对齐

线上训练暴露的问题：

- 真实 `labels.npz["dvl_mask"]` 可能保存为 `(N, H, 1)`
- 但 PR5 初版 supervision mask helper 只接受 `(N, H)`
- 这会在 `prepare_train_data()` 构造 `target_mask` 时触发 shape mismatch

本次修复方法：

- 不修改训练/评估主流程
- 不修改 `labels.npz` 主 artifact 命名与语义
- 在 `src/uwnav_dynamics/supervision_mask.py` 中集中兼容：
  - `(N, H)`
  - `(N, H, 1)`
- 对除上述两类之外的非法 shape 继续显式报错

这样做的原因是：

- 历史数据集无需重建
- train / eval 共用同一 helper，避免兼容逻辑分散
- 回滚面最小，只涉及 mask helper 与相关测试

## 5. 下一步升级顺序

建议按以下顺序推进：

1. `PR5` 后续阶段：sample-level masked visualization 与更细粒度稀疏监督表达

排序原则：

- 先解决训练/评估结果是否一致
- 再解决 rollout 语义是否稳定
- 再拆职责边界
- 最后再推进指标与 loss 的科研化升级

## 6. 当前技术债

当前仍需重点跟踪的技术债：

- sample-level masked visualization 尚未进入正式 artifact
- 评估和绘图的更细粒度 masked 表达仍可继续完善
- 历史 `split_indices.npz` 可能缺少 `split_strategy` 元数据，当前仅通过 warning 做兼容提示
- 理论文档需持续跟进工程真实状态

## 7. 新人建议入口

建议阅读顺序：

1. [README.md](/home/wys/uwnav_dynamics/README.md)
2. [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
3. [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)
4. [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)
