# 新人交接最短指南（handover guide）

本文给出从 0 到可视化结果的最短命令链，基于当前仓库已有 CLI 与配置。

## 0) 环境前提
在仓库根目录执行：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src
```

建议先确认关键配置存在：
- `configs/align/pooltest02.yaml`
- `configs/dataset/pooltest02_s1.yaml`
- `configs/train/pooltest02_s1_lstm_v0.yaml`

## 1) 对齐（align）
将 IMU/PWM/DVL/Power 对齐到主时间轴训练表：

```bash
python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml
```

产物（按配置）：`out/train/2026-01-10_pooltest02_train_base.csv`

## 2) 构建数据集（build_dataset）
从训练基础表构建滑窗数据集：

```bash
python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1.yaml
```

典型产物目录（按配置）：`data/processed/2026-01-10_pooltest02_s1/`

## 3) 训练（cli/train.py）

```bash
python -m uwnav_dynamics.cli.train \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --device cpu
```

说明：
- 若有 CUDA，可改 `--device cuda`。
- 训练会在 `run.out_dir/variant` 下写出 `best.pth` / `last.pth`。

## 4) 评估 + 绘图（cli/eval.py）

```bash
python -m uwnav_dynamics.cli.eval \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --split test \
  --plots \
  --plot_fmt png
```

产物（评估目录）通常包含：
- `metrics.yaml`
- `rmse_by_horizon.csv`
- `mae_by_horizon.csv`
- `pred_samples.npz`
- `plots/*.png`

## 5) 一条命令跑训练+评估+绘图（cli/pipeline.py）
如果想快速全流程回归，可直接：

```bash
python -m uwnav_dynamics.cli.pipeline \
  -y configs/train/pooltest02_s1_lstm_v0.yaml \
  --device cpu \
  --eval_split test \
  --plots \
  --plot_fmt png
```

## 6) 常见最小排查
- 看配置是否指向了存在的数据路径：`configs/dataset/*.yaml`、`configs/align/*.yaml`
- 看训练数据是否生成：`features.npz`、`labels.npz`
- 看 checkpoint 是否写出：`best.pth` / `last.pth`
- 看评估目录是否有 `metrics.yaml` 与 `plots/`
