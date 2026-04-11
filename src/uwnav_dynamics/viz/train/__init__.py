"""
模块名称：训练阶段可视化入口

模块职责：
统一收口训练阶段的历史曲线绘图与训练摘要可视化，
供训练主流程与手动审计脚本复用。

主要功能：
1. 从 `train_history.csv` 读取 epoch 级训练历史。
2. 导出训练损失、验证监控、学习率与总览 dashboard 图。
3. 保持训练图与评估图共享同一套科研绘图风格。

数据流：
train_history.csv / train_summary.yaml
    ↓
viz.train.plot_training_history
    ↓
train_plots/*.png|pdf

依赖模块：
- uwnav_dynamics.viz.train.plot_training_history

备注：
- 本包只负责训练阶段图表，不参与训练数值计算。
"""

