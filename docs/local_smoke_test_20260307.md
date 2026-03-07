# 本地深度学习全流程 Smoke Test 记录

更新时间：2026-03-07

## 1. 文档目标

本文档记录一次在本地电脑上执行的最小深度学习工程通路验证，目标不是追求模型精度，而是低成本确认以下链路可以完整跑通：

1. 数据对齐与基础表生成
2. 预处理后 QA 检查
3. 数据集构建
4. 小规模训练
5. 最小评估

本次 smoke test 结论是：

- `align -> cli_qa -> build_dataset -> run_train -> evaluate` 主链路已跑通
- `train_base.csv` 中 `Acc/Gyro` dense target 全 finite
- `labels.npz["Y"]` 全 finite
- 训练可完成至少 1 个 epoch，loss 为 finite
- 评估可完成最小 `test` split 数值评估并写出指标 artifact
- 评估阶段曾暴露“输入 `X` 的稀疏 power NaN 未清洗”问题，已在正式评估入口修复

## 2. 本次采用的 smoke test 策略

由于本地机器算力有限，本次测试采用“数据和训练同时缩小”的方案：

- 数据侧：
  - 先用正式 `align` 生成完整 [train_base.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base.csv)
  - 再从中截取一段连续时间片，生成 [train_base_smoke.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base_smoke.csv)
  - 保持 `hist_len=100`、`pred_len=10`、输入列和目标列语义不变
- 训练侧：
  - 单独新增 smoke train 配置
  - `device=cpu`
  - `epochs=1`
  - `batch_size=64`
  - `num_workers=0`
  - `rnn_hidden=64`
  - `rnn_layers=1`
- 评估侧：
  - 复用正式 [evaluate.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/evaluate.py)
  - 只跑 `test` split
  - `batch_size=64`

本次新增的 smoke 配置文件：

- [pooltest02_s1_smoke.yaml](/home/wys/uwnav_dynamics/configs/dataset/pooltest02_s1_smoke.yaml#L1)
- [pooltest02_s1_lstm_smoke.yaml](/home/wys/uwnav_dynamics/configs/train/pooltest02_s1_lstm_smoke.yaml#L1)

## 3. 入口与产物说明

### 3.1 正式入口

- 对齐入口：[cli_align.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/align/cli_align.py)
- QA 入口：[cli_qa.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/cli_qa.py)
- 数据集构建入口：[build_dataset.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/build_dataset.py)
- 训练入口：[run_train.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/run_train.py)
- 评估入口：[evaluate.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/evaluate.py)

### 3.2 关键产物

- 完整训练基础表：[train_base.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base.csv)
- smoke 训练基础表：[train_base_smoke.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base_smoke.csv)
- smoke dataset：
  - [features.npz](/home/wys/uwnav_dynamics/data/processed/2026-01-10_pooltest02_s1_smoke/features.npz)
  - [labels.npz](/home/wys/uwnav_dynamics/data/processed/2026-01-10_pooltest02_s1_smoke/labels.npz)
  - [meta.yaml](/home/wys/uwnav_dynamics/data/processed/2026-01-10_pooltest02_s1_smoke/meta.yaml)
- smoke 训练产物：
  - [resolved_train.yaml](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/resolved_train.yaml)
  - [split_indices.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/split_indices.npz)
  - [x_scaler.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/scalers/x_scaler.npz)
  - [y_scaler.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/scalers/y_scaler.npz)
  - [best.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/best.pth)
  - [last.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/last.pth)
- smoke 评估产物：
  - [metrics.yaml](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/eval_test/metrics.yaml)
  - [pred_samples.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/eval_test/pred_samples.npz)

## 4. 具体执行步骤

### 4.1 重新生成完整 `train_base.csv`

命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.align.cli_align -y configs/align/pooltest02.yaml
```

说明：

- 使用正式对齐配置生成完整 `train_base.csv`
- 对齐阶段会执行主时间轴选择、IMU dense 插值、DVL attach、Power attach
- 对齐结束后会自动触发 train-base QA

关键输出：

- 主时间轴：`t=[1013.417, 2193.881]`
- 行数：`115170`
- `Acc/Gyro` dense 插值 `out_nonfinite=0`
- DVL attach：`N_main_with_dvl=23820`
- Power attach：`N_main_with_power=106843`
- QA：`status=PASS`

### 4.2 从完整 `train_base.csv` 截取 smoke 时间片

本次没有新增脚本，而是直接从完整基础表中截取一段连续数据：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

src = Path('out/train/2026-01-10_pooltest02_train_base.csv')
dst = Path('out/train/2026-01-10_pooltest02_train_base_smoke.csv')
start = 15000
n_rows = 2000

df = pd.read_csv(src)
sub = df.iloc[start:start+n_rows].copy()
sub.to_csv(dst, index=False)
print(len(sub), sub["dvl_mask"].mean(), sub["power_mask"].mean())
PY
```

本次选择该片段的原因：

- `2000` 行足够构造 `hist_len=100`、`pred_len=10` 的滑窗
- `dvl_mask` 覆盖率约 `0.2385`
- `power_mask` 覆盖率约 `0.9820`
- 相比最开头时间段，这一片段更适合覆盖 velocity 稀疏监督和 power 辅助输入

### 4.3 对 smoke base 表执行 QA

命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.cli_qa --dataset-yaml configs/dataset/pooltest02_s1_smoke.yaml
```

说明：

- 使用独立 smoke dataset yaml
- QA 规则复用 [qa.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/preprocess/qa.py)
- 不做任何自动修复，只做检查与 summary 输出

结果：

- `status=PASS`
- 行数：`2000`
- 时间列 `t_s`：全 finite，严格递增
- `dvl_mask` 覆盖率：`0.238500`
- `power_mask` 覆盖率：`0.982000`
- `Acc/Gyro` 6 个 dense target：`nonfinite=0`

### 4.4 构建 smoke dataset

命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset -y configs/dataset/pooltest02_s1_smoke.yaml
```

说明：

- 读取 smoke `train_base.csv`
- 调用 `_add_state_velocity_cols()`
- 再次执行 base-csv QA
- 生成滑窗数据集并落盘 `features.npz / labels.npz / meta.yaml`

结果：

- `X shape = (1891, 100, 25)`
- `Y shape = (1891, 10, 9)`
- 窗口数计算正确：
  - `2000 - (100 + 10) + 1 = 1891`
- DVL 和 power 的历史窗 / 预测窗 mask 均正常生成

### 4.5 执行 smoke training

命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.train.run_train -y configs/train/pooltest02_s1_lstm_smoke.yaml
```

说明：

- 复用正式训练入口，不改训练契约
- 使用 smoke train yaml，在 CPU 上跑 1 个 epoch
- 训练阶段会自动：
  - 生成 split
  - 拟合 scaler
  - 构建 DataLoader
  - 保存 best/last checkpoint

结果：

- run 目录：[cpu_smoke](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke)
- split：
  - `train=1323`
  - `val=283`
  - `test=285`
- 训练侧输入 `X` 的非有限值清洗：
  - train：`20112`
  - val：`2984`
  - test：`3776`
- 第 1 个 epoch：
  - `train_loss=0.037102`
  - `val_loss=0.028638`
- checkpoint 正常写出：
  - [best.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/best.pth)
  - [last.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/last.pth)

### 4.6 执行 smoke evaluate

命令：

```bash
PYTHONPATH=src python -m uwnav_dynamics.eval.evaluate \
  -y configs/train/pooltest02_s1_lstm_smoke.yaml \
  --ckpt out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/best.pth \
  --split test \
  --device cpu \
  --batch_size 64 \
  --save_samples 32
```

说明：

- 复用训练阶段生成的 split/scaler/checkpoint
- 在 `test` split 上执行正式数值评估
- 输出 `metrics.yaml`、按 horizon 的 CSV 和 `pred_samples.npz`

最终结果：

- `n_eval=285`
- `RMSE(global)=0.338631`
- `MAE(global)=0.204056`
- `RMSE(global_masked)=0.384478`
- `MAE(global_masked)=0.242482`

## 5. QA 测试报告

### 5.1 完整 `train_base.csv`

文件：[train_base.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base.csv)

检查结果：

- 行数：`115170`
- 时间列 `t_s`：
  - 全 finite
  - 严格递增
  - `t_min=1013.417458`
  - `t_max=2193.881127`
  - `dt_min=0.002137890`
  - `dt_med=0.010213749`
  - `dt_max=0.033102882`
- `Acc/Gyro` 6 列 dense target：全部 finite
- `dvl_mask` 覆盖率：`0.206825`
- `power_mask` 覆盖率：`0.927698`

关键列统计：

- `AccX_body_mps2`：mean `-0.192458`，std `0.687196`
- `AccY_body_mps2`：mean `-0.183749`，std `0.656504`
- `AccZ_body_mps2`：mean `0.002816`，std `0.043400`
- `GyroX_body_rad_s`：mean `0.001050`，std `0.059615`
- `GyroY_body_rad_s`：mean `0.001960`，std `0.078171`
- `GyroZ_body_rad_s`：mean `-0.022617`，std `0.142761`

### 5.2 smoke `train_base.csv`

文件：[train_base_smoke.csv](/home/wys/uwnav_dynamics/out/train/2026-01-10_pooltest02_train_base_smoke.csv)

检查结果：

- 行数：`2000`
- 时间列严格递增
- `Acc/Gyro` 6 列 dense target：`nonfinite=0`
- `dvl_ratio=0.238500`
- `power_ratio=0.982000`

补充说明：

- 本次 smoke 时间片中的 `VelB*_body_mps` 虽然在 `dvl_mask=1` 的位置存在观测，但观测值基本都为 `0`
- 因此 `VelX_state_mps / VelY_state_mps / VelZ_state_mps` 在该片段中几乎全为 `0`
- 这不影响“流程可运行”验证，但不适合用来判断速度建模效果

### 5.3 `features.npz` 与 `labels.npz`

文件：

- [features.npz](/home/wys/uwnav_dynamics/data/processed/2026-01-10_pooltest02_s1_smoke/features.npz)
- [labels.npz](/home/wys/uwnav_dynamics/data/processed/2026-01-10_pooltest02_s1_smoke/labels.npz)

统计结果：

- `features.npz`
  - `X.shape=(1891, 100, 25)`
  - `nonfinite=26872`
  - 所有非有限值均来自 `P0_W..P7_W`
  - 每个 power 输入列 `nonfinite=3359`
- `labels.npz`
  - `Y.shape=(1891, 10, 9)`
  - `Y` 全 finite
  - `dvl_mask_ratio=0.238604`
  - `power_mask_ratio=0.982020`

结论：

- `Y` 契约正常，dense target 没有 NaN/Inf
- `X` 中保留的非有限值属于稀疏 power 辅助输入，不是标签错误
- 训练和评估消费端必须对这类输入做一致的非有限值清洗

## 6. 训练结果

训练入口：[run_train.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/run_train.py)

训练结果：

- 训练成功完成 1 个 epoch
- `train_loss=0.037102`
- `val_loss=0.028638`
- loss 保持 finite
- `resolved_train.yaml`、split、scaler、checkpoint 均已正常生成

训练产物检查：

- [resolved_train.yaml](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/resolved_train.yaml)：存在
- [split_indices.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/split_indices.npz)：存在
- [x_scaler.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/scalers/x_scaler.npz)：存在
- [y_scaler.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/scalers/y_scaler.npz)：存在
- [best.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/best.pth)：存在
- [last.pth](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/last.pth)：存在

## 7. 评估结果

评估入口：[evaluate.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/evaluate.py)

### 7.1 初次评估暴露的问题

第一次执行 `evaluate` 时，程序没有崩溃，但输出的指标是：

- `RMSE(global)=nan`
- `MAE(global)=nan`

根因定位：

- 训练消费端已经在 [data_pipeline.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/train/data_pipeline.py#L221) 对 scaler 后仍保留的输入 NaN 做清洗
- 评估侧此前没有对应逻辑
- `features.npz` 中 `P0_W..P7_W` 的稀疏 NaN 被直接送入模型
- 导致 `y_hat` 和 `logvar` 全部变成 NaN

### 7.2 本次最小正式修复

修复位置：

- [evaluate.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/evaluate.py#L87)
- [evaluate.py](/home/wys/uwnav_dynamics/src/uwnav_dynamics/eval/evaluate.py#L271)

修复内容：

- 新增 `_sanitize_scaled_inputs_for_eval()`
- 在评估阶段对 scaler 后的输入 `X` 执行：
  - `NaN -> 0.0`
  - `+Inf/-Inf -> 0.0`
- 语义与训练侧保持一致：`0.0` 对应 z-score 后的 train 均值

对应回归测试：

- [test_eval_input_sanitization.py](/home/wys/uwnav_dynamics/tests/test_eval_input_sanitization.py#L134)

测试命令：

```bash
PYTHONPATH=src pytest -q tests/test_eval_input_sanitization.py
```

结果：

- `1 passed`

### 7.3 修复后的评估结果

最终评估结果：

- `split=test`
- `n_eval=285`
- `RMSE(global)=0.338631`
- `MAE(global)=0.204056`
- `RMSE(global_masked)=0.384478`
- `MAE(global_masked)=0.242482`

评估 artifact：

- [metrics.yaml](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/eval_test/metrics.yaml)
- [pred_samples.npz](/home/wys/uwnav_dynamics/out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/eval_test/pred_samples.npz)

## 8. 执行结果总结

### 8.1 成功项

- `align`：成功
- `cli_qa`：成功
- `build_dataset`：成功
- `run_train`：成功
- `evaluate`：修复后成功

### 8.2 失败项与处理

本次流程中的唯一失败点是评估阶段的 NaN 输入未清洗问题。

处理方式：

- 不改训练契约
- 不改评估指标定义
- 不回写 dataset artifact
- 仅在评估消费端补齐与训练一致的输入清洗

这是一个最小补丁，且已经有独立测试覆盖。

## 9. 服务器复现步骤

若要在服务器上执行相同流程，建议按以下顺序：

### 9.1 获取代码并切换分支

```bash
git clone <repo_url>
cd uwnav_dynamics
git checkout <your_branch>
```

### 9.2 准备预处理产物

如果服务器上还没有 IMU/PWM/DVL/Power 预处理结果，先补齐对应产物，再执行：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.align.cli_align -y configs/align/pooltest02.yaml
```

### 9.3 执行 QA

正式数据：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.cli_qa --dataset-yaml configs/dataset/pooltest02_s1.yaml
```

smoke 数据：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.cli_qa --dataset-yaml configs/dataset/pooltest02_s1_smoke.yaml
```

### 9.4 构建 dataset

正式数据：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset -y configs/dataset/pooltest02_s1.yaml
```

smoke 数据：

```bash
PYTHONPATH=src python -m uwnav_dynamics.preprocess.build_dataset -y configs/dataset/pooltest02_s1_smoke.yaml
```

### 9.5 执行训练

本地 smoke：

```bash
PYTHONPATH=src python -m uwnav_dynamics.train.run_train -y configs/train/pooltest02_s1_lstm_smoke.yaml
```

服务器正式训练：

```bash
PYTHONPATH=src python -m uwnav_dynamics.train.run_train -y configs/train/pooltest02_s1_lstm_v0.yaml
```

### 9.6 执行评估

smoke 评估：

```bash
PYTHONPATH=src python -m uwnav_dynamics.eval.evaluate \
  -y configs/train/pooltest02_s1_lstm_smoke.yaml \
  --ckpt out/ckpts/pooltest02_s1_lstm_smoke/cpu_smoke/best.pth \
  --split test \
  --device cpu \
  --batch_size 64
```

正式评估：

```bash
PYTHONPATH=src python -m uwnav_dynamics.eval.evaluate \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --ckpt <best_ckpt_path> \
  --split test \
  --device cuda \
  --batch_size 512
```

### 9.7 记录建议

建议在服务器上为每次运行至少记录：

- `align` 输出行数、DVL/Power 覆盖率、QA 结果
- `build_dataset` 输出 `X/Y shape`
- `run_train` 输出 `train_loss / val_loss`
- `evaluate` 输出 `RMSE / MAE`
- run 目录中的 `resolved_train.yaml / split / scaler / best.pth / metrics.yaml`

## 10. 后续行动建议

### 10.1 在服务器上继续训练

本地 smoke test 只用于验证工程链路。若要继续科研训练，建议：

- 使用完整 [pooltest02_s1.yaml](/home/wys/uwnav_dynamics/configs/dataset/pooltest02_s1.yaml)
- 使用正式 [pooltest02_s1_lstm_v0.yaml](/home/wys/uwnav_dynamics/configs/train/pooltest02_s1_lstm_v0.yaml)
- 训练时恢复更大的：
  - batch size
  - hidden dim
  - epochs
  - CUDA 设备

### 10.2 后续优化点

建议继续跟踪以下问题：

1. smoke 时间片的速度观测较弱
   - 适合工程通路验证
   - 不适合评估速度建模质量
2. `features.npz` 中的 power 输入仍以稀疏 NaN 表达
   - 当前训练和评估消费端已兼容
   - 若后续希望 dataset artifact 本身全 finite，需要单独设计 power 输入补值策略
3. 服务器正式训练前，建议先在目标数据集上重新执行一次 `cli_qa`
   - 确认 `Acc/Gyro` dense target 全 finite
   - 确认时间列严格递增
   - 确认 `dvl_mask / power_mask` 覆盖率符合预期

## 11. 结论

截至 2026-03-07，本仓库的本地最小深度学习主链路已经验证通过：

- 预处理对齐可生成合格的 `train_base.csv`
- QA 可在 dataset build 前拦截坏数据
- smoke dataset 可正常构建
- 训练可完成至少 1 个 epoch，loss finite
- 评估可完成最小数值评估并写出 artifact
- 评估侧输入 NaN 不一致问题已被正式修复并补充回归测试

因此可以认为：

当前仓库已经具备“本地工程通路验证”和“服务器正式训练前预检”的基础能力。
