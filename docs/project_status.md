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

## 5. 下一步升级顺序

建议按以下顺序推进：

1. `PR4`：rollout 索引契约
2. `PR5`：mask-aware 训练评估

排序原则：

- 先解决训练/评估结果是否一致
- 再解决 rollout 语义是否稳定
- 再拆职责边界
- 最后再推进指标与 loss 的科研化升级

## 6. 当前技术债

当前仍需重点跟踪的技术债：

- rollout 索引仍存在历史硬编码风险
- DVL 稀疏监督尚未正式进入 mask-aware 训练
- 评估和绘图的职责边界仍不够清晰
- 历史 `split_indices.npz` 可能缺少 `split_strategy` 元数据，当前仅通过 warning 做兼容提示
- 理论文档需持续跟进工程真实状态

## 7. 新人建议入口

建议阅读顺序：

1. [README.md](/home/wys/uwnav_dynamics/README.md)
2. [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
3. [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)
4. [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)
