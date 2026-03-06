# uwnav_dynamics

Control-Oriented Data-Driven Dynamics Modeling for Underwater Robots

## 项目简介

`uwnav_dynamics` 面向水下机器人动力学辨识与短时预测控制准备。
项目当前聚焦一条可复现、可审查的工程链路：从 PWM、IMU、DVL、Power 日志出发，
构建多频异步数据集，训练短时多步动力学预测模型，并输出可用于后续 MPC / 控制评估的离线结果。

当前数据频率：

- PWM: 100 Hz
- IMU: 100 Hz
- DVL: 10 Hz
- Power: 5 Hz

## 研究目标

本项目要解决的不是“完整水动力参数闭式辨识”，而是一个更面向控制的研究问题：

- 在多频异步观测条件下构建稳定的数据驱动动力学模型
- 以推进器输入和传感器历史窗口预测未来短时状态响应
- 在训练与评估阶段保留稀疏监督、不确定度与 rollout 语义
- 为未来 MPC / 仿真 / 闭环验证提供统一接口

## 当前实现范围

当前仓库已落地的主流水线是：

```text
Raw Logs
-> Sensor Preprocess
-> Multi-rate Alignment
-> Sliding Window Dataset
-> Model Training
-> Evaluation
-> Visualization
```

工程现状：

- 主时间轴为 100 Hz
- 当前主模型为 `S1Predictor`（LSTM backbone + optional blocks）
- rollout 语义为 `y_hat = y0 + cumsum(dY)`
- 已完成配置契约收口与 train/eval 配置一致性修复
- 已恢复多频对齐边界的 pytest 冒烟能力

不在当前实现范围内的内容：

- 闭环 MPC 控制器实现
- 完整 6-DOF 水动力参数辨识
- mask-aware 稀疏监督训练的正式落地版本
- 评估可视化完全解耦后的最终工程形态

## 当前升级状态

已完成：

- `PR6`：多频对齐边界修复与 pytest 冒烟恢复
- `PR1`：配置契约收口、canonical parser、resolved config 快照

下一步按顺序推进：

- `PR2`：split / scaler 单一真源
- `PR4`：rollout 索引契约
- `PR3`：eval-viz 解耦
- `PR5`：mask-aware 训练评估

详细说明见：

- [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
- [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
- [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)
- [modeling_roadmap.md](/home/wys/uwnav_dynamics/docs/modeling_roadmap.md)

## 快速上手

在仓库根目录：

```bash
export PYTHONPATH=src
```

最短文档路径：

1. 先读 [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
2. 再读 [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
3. 然后按 [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md) 跑通命令链

最小训练命令示例：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --device cpu
```

最小评估命令示例：

```bash
python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --split test
```

## 文档导航

- 当前实现说明：[ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
- 项目状态：[project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
- 工程路线：[engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)
- 建模路线：[modeling_roadmap.md](/home/wys/uwnav_dynamics/docs/modeling_roadmap.md)
- 评估规范：[evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)
- 配置契约：[config_contract.md](/home/wys/uwnav_dynamics/docs/config_contract.md)
- 数据契约：[dataset_spec.md](/home/wys/uwnav_dynamics/docs/design/dataset_spec.md)
- 绘图规范：[plot_style_guide.md](/home/wys/uwnav_dynamics/docs/design/plot_style_guide.md)
- 理论文档入口：[docs/math/README.md](/home/wys/uwnav_dynamics/docs/math/README.md)
- 仓库索引：[repo_index.md](/home/wys/uwnav_dynamics/docs/repo_index.md)
- 目录树：[repo_tree.txt](/home/wys/uwnav_dynamics/repo_tree.txt)

## 复现实验

建议把一次实验的最小复现单元理解为：

- 原始 train yaml
- `resolved_train.yaml`
- `best.pth` / `last.pth`
- `split_indices.npz`
- `x_scaler.npz`
- `y_scaler.npz`
- `metrics.yaml`

正式说明见 [evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)。

## 参与开发

推荐开发入口：

1. 阅读 [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md) 了解当前阶段
2. 阅读 [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md) 选择待推进 PR
3. 修改前先核对 [config_contract.md](/home/wys/uwnav_dynamics/docs/config_contract.md) 与 [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
4. 提交前至少运行 smoke test，并同步更新相应文档

## License

本仓库当前许可证为 `AGPL-3.0-or-later`，与 [LICENSE](/home/wys/uwnav_dynamics/LICENSE) 保持一致。

## Citation

仓库包含 [CITATION.cff](/home/wys/uwnav_dynamics/CITATION.cff)。若你在学术工作中使用本项目，可参考以下 BibTeX：

```bibtex
@software{uwnav_dynamics_2026,
  title   = {uwnav_dynamics: Control-Oriented Data-Driven Dynamics Modeling for Underwater Robots},
  author  = {Wang, YuShu},
  year    = {2026},
  url     = {https://github.com/wyswang3/uwnav_dynamics},
  license = {AGPL-3.0-or-later}
}
```
