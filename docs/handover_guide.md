# 项目交接指南

更新时间：2026-03-28

## 1. 当前接手时先知道什么

当前项目的主线已经明确切换为：

**`KF / ESKF 融合预处理 -> 长期拟合训练 -> 8 卡筛选 -> 长时拟合图证据`**

当前不要再从旧的 controller / transition validation 叙事切入。  
当前最重要的问题是：

- 状态代理量是否足够干净
- 训练目标是否真的约束长期 rollout
- 哪组模型在长期拟合上更稳

## 2. 先读哪几份文档

建议顺序：

1. [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
2. [kf_fusion_preprocess_training_v2.md](/home/wys/uwnav_dynamics/docs/design/kf_fusion_preprocess_training_v2.md)
3. [快捷命令行.md](/home/wys/uwnav_dynamics/docs/快捷命令行.md)
4. [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)
5. [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)

如果只是要尽快迁移到服务器，前 3 份足够。

## 2.1 下次接手的 5 分钟恢复顺序

在仓库根目录执行：

```bash
cd /home/wys/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
export PYTHONPATH=src
```

然后按这个顺序恢复上下文：

1. 先看本文件第 `3.1` 节，确认三处未闭合阻塞。
2. 再看 `docs/handover_kf_training_server_v2.md` 第 `4.2` 节，确认服务器上的正确执行顺序。
3. 再看 `docs/快捷命令行.md`，直接复用命令模板。
4. 如果要继续修代码，优先从 `kf_eskf.py`、`data_pipeline.py`、`s1_predictor.py` 三处开始。

## 3. 当前已经落地的事实

当前可以直接依赖的事实：

- IMU `transform -> gravity -> bias -> filter` 预处理链已经稳定。
- `KF / ESKF` 融合模块已经落地到 `src/uwnav_dynamics/preprocess/fusion/`。
- 新的 KF 融合基础表已经可以生成。
- `kf_ctx_v2` 数据集契约已经固定为 `29` 维输入、`9` 维目标。
- 训练损失已经支持 `transition_balance`。
- 训练期 best ckpt 已经支持用 `val_transition_score` 而不是单纯 `val_loss` 选模。
- 8 卡训练矩阵已经准备好，但当前不建议在修完三处阻塞前直接启动。

## 3.1 当前必须先修的三处问题

在最近一次状态转移训练审查后，当前还有三处关键阻塞没有闭合：

1. `KF / ESKF` 融合初始化仍可能泄漏未来速度信息。  
   代码位置：`src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`  
   当前 `_select_initial_velocity()` 会取“整段序列第一条有效 DVL”，
   这会把未来观测带回序列起点。

2. rollout 训练当前仍存在 `X / Y` 双 scaler 语义错位风险。  
   代码位置：`src/uwnav_dynamics/train/data_pipeline.py`、`src/uwnav_dynamics/train/run_train.py`  
   当前 `x_scaler` 与 `y_scaler` 分别独立拟合，但训练时 `y0` 从 `X` 取，
   `Y` 从 `y_scaler` 空间比较；如果共享状态维的统计量不一致，
   则状态转移 loss 的物理语义会被破坏。

3. 当前模型仍是“历史窗 -> 固定未来块输出”预测器，
   还不是严格的 `x_{t+1} = f(x_t, u_t)` 一步状态转移算子。  
   代码位置：`configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`、`configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`、`src/uwnav_dynamics/models/nets/s1_predictor.py`

因此，当前不要直接进入正式 8 卡训练。

## 4. 当前最短工作流

仓库根目录：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

1. 自检

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_transition_balance_design.py \
  tests/test_trainer_controls.py \
  tests/test_kf_ctx_training_config_v2.py
```

2. 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

3. 构建数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

4. 单卡训练 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

5. 8 卡训练矩阵

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml
```

## 5. 当前最该盯的产物

训练前：

- `out/train/2026-01-10_pooltest02_train_base_kf_v2.csv`
- `data/processed/2026-01-10_pooltest02_s1_kf_ctx_v2/`

训练后：

- `run_dir/resolved_train.yaml`
- `run_dir/train_history.csv`
- `run_dir/train_summary.yaml`
- `run_dir/best.pth`
- `run_dir/eval_test/metrics.yaml`

矩阵后：

- `out/train_matrix/<variant>/summary.csv`

## 6. 当前选模口径

当前不再只看 `val_loss`。

训练期：

- ckpt 选择和 early stopping 优先按 `train.metric`
- 当前推荐值是 `val_transition_score`

run 级筛选：

- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- `acc / gyro / vel` 组误差

## 7. 现在不要先做什么

当前不建议先做：

- 恢复旧的 controller 验证文档链
- 直接扩成 12 维主输出协议
- 先画大量图再看数值
- 把 KF 输出当成“绝对真值”来表述

## 8. 当前下一步

下一步最合理的顺序是：

1. 先修复融合初始化泄漏
2. 再修复共享状态维的 scaler 语义
3. 把训练任务收口成更接近 `x_{t+1}=f(x_t,u_t)` 的一步状态转移形式
4. 之后再重建数据集、做单卡 smoke、启动 8 卡矩阵

如果后续要继续扩展，应优先扩训练与评估链，而不是重新打开旧阶段的验证壳层。
