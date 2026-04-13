# 理论文档说明

## 1. 文档目标

`docs/math/` 用于存放可直接整合进论文、技术报告或内部研究说明的 LaTeX 文档。

这里的目标不是复述代码实现细节，而是：

- 给出项目采用的核心符号系统
- 把原始数据、预处理、因果融合到训练建模的链路写成可论文复用的方法章节
- 说明水下机器人动力学建模问题的理论背景
- 描述当前神经网络动力学模型的抽象形式
- 记录训练目标与未来控制接口的数学表达

当前主问题定义已经收口为：

```text
在因果状态代理量空间中学习短时状态转移，
并验证该模型能否承担经验型状态求解器职责。
```

## 2. 当前实现 vs 理论文档

当前工程实现已经落地的部分：

- 多频异步日志到 100 Hz 主时间轴的对齐链路
- IMU 预处理与论文风格原始/处理图
- KF / ESKF 状态代理量构造链
- 基于窗口输入的短时多步预测模型
- `pred_len=1` 的单步状态转移实验分支
- `dY + rollout` 形式的未来状态生成
- 可选的结构先验 blocks
- 异方差输出头
- 训练后状态求解器封装与长序列 replay 验证

当前理论文档中既包含“已实现内容”，也保留“下一步目标表达”。

应这样理解：

- 若某节明确对应当前代码，就可直接视为当前方法说明
- 若某节描述统一 `step(u_t, dt)` 接口、控制接入或稳定性约束，
  通常属于“已明确方向、尚未完全工程化”的目标态表达
- 当前最值得优先对照的数学主线，不再是“更长 horizon 的拟合能力”，
  而是“因果状态代理量 + 一步状态转移 + replay 递推可用性”

## 3. 当前章节结构

当前主文档 `main.tex` 由以下章节组成：

- `notation.tex`
- `preprocessing_pipeline.tex`
- `dynamics_model.tex`
- `network_structure.tex`
- `training_objective.tex`
- `transition_solver_feasibility.tex`
- `control_integration.tex`
- `control_interface.tex`

## 4. 推荐阅读顺序

若目标是快速理解当前工程主线，建议按这个顺序读：

1. `notation.tex`
2. `preprocessing_pipeline.tex`
3. `dynamics_model.tex`
4. `training_objective.tex`
5. `transition_solver_feasibility.tex`
6. `control_interface.tex`

## 5. 当前章节状态

与当前代码最直接对应的是：

- `notation.tex`
- `preprocessing_pipeline.tex`
- `dynamics_model.tex`
- `network_structure.tex`
- `transition_solver_feasibility.tex`

与当前实验主线部分对应、仍在推进的是：

- `training_objective.tex`
- `control_interface.tex`
- `control_integration.tex`

更偏“目标接口与后续理论收口”的是：

- `control_interface.tex`
- `control_integration.tex`

这些章节应理解为：

- 数学问题定义已经明确
- 工程壳层与统一 solver API 仍在继续收口
- 不应把其表述成“仓库已经具备完整闭环控制接口”

## 6. 当前理论边界

当前数学文档明确不主张以下叙述：

- 神经网络已经替代完整高保真水动力学模型
- 离线 rollout 结果已经等价于闭环稳定性证明
- 当前仓库已经具备完整生产级 `step(u_t, dt)` 控制接口

当前更稳妥的表述应是：

- 代理状态构造已经足够支持状态转移建模
- 单步状态转移分支已成为当前更贴近 solver 的主线
- replay 验证是从离线拟合走向控制接入前的必要证据层

## 7. 编译方式

在 `docs/math/` 目录下可使用：

```bash
latexmk -xelatex main.tex
```

若只做最小验证，也可使用：

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp main.tex
```

## 8. 维护原则

- 理论文档应优先服务论文与技术报告，不直接夹带实现细节注释
- 若工程语义发生变化，应同步更新数学符号与问题定义
- 若某节仍属规划态，应在文中写清楚，不得伪装成“已实现事实”
