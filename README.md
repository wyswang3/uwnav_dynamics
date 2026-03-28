# uwnav_dynamics

面向水下机器人的数据驱动状态转移求解器训练仓库。

## 当前目标

项目当前不再把重点放在 controller 壳层或历史验证图上，而是聚焦一条更直接的主线：

- 用 `KF / ESKF` 融合预处理统一多源异步观测
- 生成低噪声、统一时间轴的 `acc / gyro / vel / attitude` 状态代理量
- 训练 `S1Predictor` 学习“控制输入如何驱动系统状态演化”
- 以长期 rollout 拟合精度为核心标准筛选模型
- 最终得到一个可长期使用、误差处于允许范围内的状态转移求解器

## 当前主流水线

```text
Raw Logs
-> IMU preprocess
-> Multi-rate alignment
-> KF / ESKF fusion
-> Sliding-window dataset
-> S1Predictor training
-> Offline evaluation
-> Long-replay figures
```

当前默认数据频率：

- PWM: 100 Hz
- IMU: 100 Hz
- DVL: 10 Hz
- Power: 5 Hz

## 当前训练契约

当前训练主线已经切到 KF 融合状态代理量：

- 输入：`29` 维  
  `PWM 8 + AccKf 3 + GyroKf 3 + VelKf 3 + AttCtx 4 + Power 8`
- 输出：`9` 维  
  `AccKf 3 + GyroKf 3 + VelKf 3`
- 主模型：`S1Predictor`
- rollout 契约：`y_hat = y0 + cumsum(dY)`
- 主损失：`transition_balance`
- 训练期监控指标：`val_transition_score`

这里的 `val_transition_score` 不再直接读 `logvar`，而是偏向长期状态误差与尾部 horizon 误差，用来避免模型通过放大不确定度掩盖长期 rollout 漂移。

## 当前先不要直接开训

当前训练主线虽然已经切到 KF 融合状态代理量，但正式重训前还要先修三处问题：

1. `KF / ESKF` 初始化仍有未来 DVL 泄漏风险
2. `x_scaler / y_scaler` 双 scaler 可能破坏 rollout 状态转移语义
3. 当前模型仍偏“历史窗 -> 固定未来块输出”，还不是严格的一步状态转移算子

因此，当前仓库最正确的状态是：

- 可以继续整理、修补和验证训练链
- 暂不建议直接启动正式 8 卡矩阵

## 快速开始

在仓库根目录：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

最短阅读顺序：

1. `docs/handover_guide.md`
2. `docs/handover_kf_training_server_v2.md`
3. `docs/快捷命令行.md`
4. `ARCHITECTURE.md`

最短执行顺序：

1. 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

2. 构建 KF 数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

3. 单卡训练 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

4. 8 卡训练矩阵

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml
```

## 结果图约束

最终图包面向论文和技术汇报复用，统一遵循：

- 不要图表标题
- 只保留坐标轴标题、图例和单位
- 字体统一 `Times New Roman`
- 固定导出比例，优先 `4:3`
- 坐标轴自适应
- 配套 `figure_notes.md` 说明图片证明对象、run 来源和解释边界

## 文档入口

- 交接入口：`docs/handover_guide.md`
- 服务器迁移：`docs/handover_kf_training_server_v2.md`
- 操作说明书：`docs/快捷命令行.md`
- 系统架构：`ARCHITECTURE.md`
- 项目状态：`docs/project_status.md`
- 设计说明：`docs/design/kf_fusion_preprocess_training_v2.md`
- 评估规范：`docs/evaluation_protocol.md`
- 文件索引：`docs/repo_index.md`

## 当前不再作为主线的内容

当前不再以这些方向作为入口层叙事：

- 历史 round4 / round5 controller 相关结论
- 旧的 transition validation 壳层文档
- 只追求 `val_loss` 的选模口径
- 把前向填充速度代理量继续当成主监督真源

这些材料如果仍然存在，只作为历史参考，不再代表当前训练主线。

## 开发要求

- 修改训练、预处理或评估逻辑后，必须同步更新文档
- 新代码文件必须带中文模块说明
- 提交前至少运行必要 pytest

推荐最小自检：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_transition_balance_design.py \
  tests/test_trainer_controls.py \
  tests/test_kf_ctx_training_config_v2.py
```

## License

当前许可证为 `AGPL-3.0-or-later`，见 `LICENSE`。
