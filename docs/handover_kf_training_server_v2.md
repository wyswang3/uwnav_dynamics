# KF 融合训练服务器迁移交接文档

更新时间：2026-03-28  
当前工作分支：`feature/kf-preprocess-training-v1`

## 1. 迁移目标

服务器上的目标不是恢复旧的 controller 验证流程，而是尽快把当前主线跑通：

1. 生成新的 KF 融合基础表
2. 构建新的 KF 数据集
3. 单卡 smoke 验证训练链
4. 启动 8 卡矩阵
5. 对 top2 输出长期拟合图包

注意：

- 当前文档里原本的“直接上服务器训练”顺序已经需要暂缓。
- 在最近一次训练链审查后，当前还应先修三处关键问题，再开始正式重训。

## 2. 当前主线摘要

当前主线已经固定为：

- 因果 `KF / ESKF` 融合预处理
- `29` 维输入、`9` 维主监督
- `S1Predictor + grouped head + transition_balance`
- 训练期按 `val_transition_score` 选 best ckpt
- run 级按长期 rollout 相关指标筛选

但当前仍未闭合的阻塞是：

1. `KF / ESKF` 初始化速度存在未来 DVL 泄漏风险。
2. `x_scaler / y_scaler` 双 scaler 可能破坏 rollout 状态转移语义。
3. 当前模型还是 `hist -> future block` 预测器，不是严格的一步状态转移算子。

## 3. 需要同步到服务器的核心路径

代码与配置：

- `src/uwnav_dynamics/`
- `configs/fusion/pooltest02_kf_eskf_v2.yaml`
- `configs/dataset/pooltest02_s1_kf_ctx_v2.yaml`
- `configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml`
- `configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml`

文档：

- `README.md`
- `docs/handover_guide.md`
- `docs/handover_kf_training_server_v2.md`
- `docs/快捷命令行.md`
- `docs/design/kf_fusion_preprocess_training_v2.md`

## 3.1 下次进入服务器后的最小恢复动作

```bash
cd /path/to/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
git log --oneline -3
export PYTHONPATH=src
```

然后先做两件事：

1. 读 `docs/handover_guide.md` 第 `3.1` 节
2. 跑本页第 `4.1` 节自检命令

## 4. 服务器上的最短执行顺序

```bash
cd /path/to/uwnav_dynamics
git switch feature/kf-preprocess-training-v1
export PYTHONPATH=src
```

### 4.1 配置与代码自检

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kf_eskf_fusion.py \
  tests/test_transition_balance_design.py \
  tests/test_trainer_controls.py \
  tests/test_kf_ctx_training_config_v2.py
```

### 4.2 当前更推荐的顺序

先修以下三件事：

1. `kf_eskf.py` 的初始化泄漏
2. 共享状态维的 scaler 契约
3. 一步状态转移训练配置

只有这三项收口后，再继续下面的正式训练命令。

### 4.3 生成融合基础表

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

### 4.4 构建数据集

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_v2.yaml
```

### 4.5 单卡 smoke

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_transition_balance_v2.yaml
```

### 4.6 8 卡矩阵

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_ctx_8gpu_v1.yaml
```

## 5. 迁移后先检查什么

融合基础表：

- `AccKf / GyroKf / VelKf` 列全 finite
- 没有 NaN 注入训练表

数据集：

- `features.npz["X"]` 形状正确
- `labels.npz["Y"]` 形状正确
- `labels.npz["dvl_mask"]` 均值应为 `1.0`

训练：

- `train_summary.yaml` 中能看到：
  - `monitor_name: val_transition_score`
  - `best_monitor`
  - `best_val_loss`
- `train_history.csv` 中应有：
  - `val_loss`
  - `monitor_value`

矩阵：

- `summary.csv` 正常产出
- 每个 run 都有 `train_summary.yaml` 和 `eval_test/metrics.yaml`

## 6. 当前筛选标准

训练期：

- 先看 `val_transition_score`
- 再看 `selected_val_loss / best_val_loss`

run 级：

- `final_step`
- `rollout_growth`
- `tail_error`
- `worst_abs_bias`
- `acc / gyro / vel` 分组误差

## 7. 图包约束

图包仍沿用统一风格：

- 无标题
- 只保留坐标轴标题、图例和单位
- `Times New Roman`
- 固定 `4:3`
- 配套 `figure_notes.md`

## 8. 当前不要做的事

- 不要先恢复旧的 controller 图壳层
- 不要先改 12 维主输出协议
- 不要只靠 `val_loss` 决定模型
- 不要把 KF 输出写成高保真物理真值
