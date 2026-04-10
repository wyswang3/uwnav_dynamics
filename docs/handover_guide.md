# 项目交接指南

更新时间：2026-04-10

## 1. 当前接手时先知道什么

当前项目主线已经从“先跑长时拟合矩阵”进一步收口为：

**`因果 KF 状态代理量 -> 共享状态维 scaler 收口 -> 一步状态转移求解器升级`**

当前不要再从旧的 controller / transition validation 壳层切入。  
当前最重要的问题已经变成：

- 当前状态代理量链是否满足严格因果
- rollout 训练的数值语义是否一致
- 下一阶段如何把模型收口成更接近 `x_{t+1}=f(x_t,u_t,c_t)` 的一步算子

## 2. 先读哪几份文档

建议顺序：

1. [README.md](/home/wys/uwnav_dynamics/docs/README.md)
2. [transition_solver_phase1_upgrade.md](/home/wys/uwnav_dynamics/docs/design/transition_solver_phase1_upgrade.md)
3. [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
4. [kf_fusion_preprocess_training_v2.md](/home/wys/uwnav_dynamics/docs/design/kf_fusion_preprocess_training_v2.md)
5. [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
6. [quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)

如果只是要尽快恢复本地上下文，前 3 份足够。

## 2.1 下次接手的 5 分钟恢复顺序

在仓库根目录执行：

```bash
cd /home/wys/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
export PYTHONPATH=src
```

然后按这个顺序恢复上下文：

1. 先看本文件第 `3.1` 节，确认 Phase 1 已完成内容与剩余阻塞。
2. 再看 `docs/design/transition_solver_phase1_upgrade.md` 第 `5` 节，确认本地最小执行顺序。
3. 再看 `docs/handover_kf_training_server_v2.md` 第 `4.2` 节，确认服务器上的执行顺序。
4. 需要查命令时，去 `docs/reference/quick_commands.md`，不要把它当主交接文档。
5. 如果要继续修代码，优先从 `data_pipeline.py`、`s1_predictor.py`、`run_train.py` 三处开始。

## 3. 当前已经落地的事实

当前可以直接依赖的事实：

- IMU `transform -> gravity -> bias -> filter` 预处理链稳定。
- `KF / ESKF` 融合模块已经落地到 `src/uwnav_dynamics/preprocess/fusion/`。
- `kf_ctx_v2` 数据集契约已经固定为 `29` 维输入、`9` 维目标。
- 训练损失已经支持 `transition_balance`。
- 训练期 best ckpt 已经支持按 `val_transition_score` 选模。
- Phase 1 已修复：
  - 融合初始化未来 DVL 泄漏
  - 共享状态维 `X/Y` scaler 语义错位
- Phase 2 预备配置已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml`
  - `configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- Phase 2 单步分支已新增：
  - `configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml`
  - `configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`

## 3.1 当前阶段结论

Phase 1 已完成并收口了前两个基础阻塞：

1. `KF / ESKF` 融合初始化已改为严格因果 warm-start  
   代码位置：`src/uwnav_dynamics/preprocess/fusion/kf_eskf.py`

2. rollout 训练的共享状态维 scaler 已收口为单一统计量  
   代码位置：`src/uwnav_dynamics/train/data_pipeline.py`

当前剩余主阻塞：

3. 模型仍是“历史窗 -> 固定未来块输出”预测器，
   还不是严格的一步状态转移算子。  
   代码位置：`src/uwnav_dynamics/models/nets/s1_predictor.py`

因此，当前可以恢复单卡 smoke 与新一轮数据重建，
但还不建议把项目表述成“已具备闭环求解器”。

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
  tests/test_train_data_pipeline_nan_sanitization.py \
  tests/test_transition_balance_design.py \
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

如果要测试质量上下文版输入链，改用：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

如果要测试单步状态转移版数据集，改用：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

4. 单卡训练 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

质量上下文版 smoke：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml
```

单步状态转移版 smoke：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
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

1. 用 Phase 1 修复后的代码重建融合基础表与数据集
2. 先对比 v2、quality v3 与 quality step v1 的单卡 smoke
3. 评估是否把一步状态转移分支转正为主线
4. 之后再重开 8 卡矩阵与 top2 图包

如果后续要继续扩展，应优先扩训练与评估链、求解器接口与最小 replay 验证，
而不是重新打开旧阶段的大型验证壳层。
