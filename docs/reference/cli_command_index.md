# 命令行入口索引

更新时间：2026-04-13

## 1. 目的

本文档只做一件事：

- 统一列出当前仓库可直接执行的主要命令行入口
- 说明每个入口解决什么问题
- 给出最小调用模板，方便快速查找

说明：

- 如果你要按当前主线直接执行实验，优先看 [quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)。
- 如果你要确认“某个 CLI 入口到底叫什么、接什么配置、写到哪里”，优先看本文。

## 2. 通用前置

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

路径约定：

- `configs/...`、`out/...`、`data/...` 这类路径默认按仓库根目录解释。
- 如果 launcher 配置里的路径显式包含 `..`，则按“该配置文件所在目录”解释。
- 这样可以避免把产物误写到系统根目录或错误的工作目录。

## 3. 预处理入口

### 3.1 时间对齐

用途：

- 把 IMU / DVL / PWM / Power 等多源数据对齐到统一时间轴

命令：

```bash
python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml
```

### 3.2 KF / ESKF 融合

用途：

- 在对齐基础表上生成因果 KF 状态代理量

命令：

```bash
python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml
```

### 3.3 数据集构建

用途：

- 从 `train_base.csv / train_base_kf_v2.csv` 构建 `features.npz / labels.npz / meta.yaml`

命令：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_step_v1.yaml
```

### 3.4 数据 QA

用途：

- 对基础表或数据集 coverage / mask / 缺测情况做快速检查

命令：

```bash
python -m uwnav_dynamics.preprocess.cli_qa \
  --help
```

## 4. 单次训练与评估

### 4.1 单次训练

用途：

- 跑一个 train yaml

命令：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

常用覆盖项：

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --device cuda:0 \
  --epochs 5 \
  --batch_size 1024
```

### 4.2 数值评估与出图

用途：

- 读取训练产物，执行 `eval_test / eval_val / eval_train`
- 可选串联绘图

命令：

```bash
python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

### 4.3 训练后一步到位流水线

用途：

- 从单个 train yaml 出发，串起 train + eval + plots

命令：

```bash
python -m uwnav_dynamics.cli.pipeline \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml
```

## 5. 长序列验证入口

### 5.1 单模型 replay

用途：

- 把训练后的网络当作状态转移求解器做长序列 autoregressive replay

命令：

```bash
python -m uwnav_dynamics.cli.transition_replay \
  -y configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml \
  --split test \
  --min_steps 50
```

### 5.2 多模型 replay matrix

用途：

- 对多个候选模型按统一 replay 协议批量比较

命令：

```bash
python -m uwnav_dynamics.cli.transition_replay_matrix \
  -c configs/launch/replay_matrix_example.yaml
```

主要产物：

```text
work_dir/
  manifest.yaml
  summary.csv
  ranking.csv
  runs/<candidate_name>/
```

## 6. 矩阵训练与总控入口

### 6.1 train matrix

用途：

- 同一基础 train yaml 下物化多个变体
- 按 GPU 池并发执行 `train -> eval -> compare`

命令：

```bash
python -m uwnav_dynamics.cli.train_matrix \
  -c configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml
```

### 6.2 server pipeline

用途：

- 一次性编排 `fusion -> dataset -> smoke -> train_matrix -> replay_matrix`

命令：

```bash
python -m uwnav_dynamics.cli.server_pipeline \
  -c configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml
```

主要产物：

```text
out/server_pipeline/<variant>/
  manifest.yaml
  phase_status.csv
  generated_replay_matrix/*.yaml
  final_selection.csv
```

## 7. baseline 与辅助入口

### 7.1 baseline runner

用途：

- 用统一 split / scaler / eval 契约跑 baseline 对照

命令：

```bash
python -m uwnav_dynamics.cli.baseline_runner --help
```

## 8. 当前最常用的配置入口

建议优先记住这几份：

- 单步主线训练：`configs/train/pooltest02_s1_kf_ctx_quality_step_transition_v1.yaml`
- 多步质量主线训练：`configs/train/pooltest02_s1_kf_ctx_quality_transition_v3.yaml`
- 单步 8 卡矩阵：`configs/launch/pooltest02_s1_kf_quality_step_8gpu_v2.yaml`
- 多步 8 卡矩阵：`configs/launch/pooltest02_s1_kf_quality_8gpu_v2.yaml`
- 8 卡全流程总控：`configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml`

## 9. 相关文档

- [quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)
- [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)
- [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
- [config_contract.md](/home/wys/uwnav_dynamics/docs/config_contract.md)
- [evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)
