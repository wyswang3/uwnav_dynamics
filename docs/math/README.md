# 理论文档说明

## 1. 文档目标

`docs/math/` 用于存放可直接整合进论文、技术报告或内部研究说明的 LaTeX 文档。

这里的目标不是复述代码实现细节，而是：

- 给出项目采用的核心符号系统
- 把原始数据、预处理、因果融合到训练建模的链路写成可论文复用的方法章节
- 说明水下机器人动力学建模问题的理论背景
- 描述当前神经网络动力学模型的抽象形式
- 记录训练目标与未来控制接口的数学表达

## 2. 当前实现 vs 理论文档

当前工程实现已经落地的部分：

- 多频异步日志到 100 Hz 主时间轴的对齐链路
- IMU 预处理与论文风格原始/处理图
- KF / ESKF 状态代理量构造链
- 基于窗口输入的短时多步预测模型
- `dY + rollout` 形式的未来状态生成
- 可选的结构先验 blocks
- 异方差输出头

当前理论文档中既包含“已实现内容”，也保留“下一步目标表达”。

应这样理解：

- 若某节明确对应当前代码，就可直接视为当前方法说明
- 若某节描述 mask-aware 稀疏监督或控制接口，则属于后续建模方向的正式化表达

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

## 4. 已实现章节

与当前代码最直接对应的是：

- `notation.tex`
- `preprocessing_pipeline.tex`
- `network_structure.tex`

部分对应、仍在推进的是：

- `training_objective.tex`
- `control_integration.tex`

## 5. 编译方式

在 `docs/math/` 目录下可使用：

```bash
latexmk -xelatex main.tex
```

若只做最小验证，也可使用：

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp main.tex
```

## 6. 维护原则

- 理论文档应优先服务论文与技术报告，不直接夹带实现细节注释
- 若工程语义发生变化，应同步更新数学符号与问题定义
- 若某节仍属规划态，应在文中写清楚，不得伪装成“已实现事实”
