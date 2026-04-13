# 数据与 Git 管理规则

更新时间：2026-04-13

## 1. 目标

当前仓库的数据治理目标只有两条：

1. 原始数据要能随仓库一起交付与复核。
2. 处理后数据与运行产物不进 git，统一按配置和算法在服务器重建。

## 2. 目录职责

仓库当前使用如下分层：

```text
data/
  raw/
  interim/
  processed/
  splits/

out/
replay_matrix/
runs/
```

各目录职责如下：

- `data/raw/`：原始日志底稿，允许进入 git
- `data/interim/`：中间对齐、辅助分析、离线导航或临时转换产物，不进 git
- `data/processed/`：训练直接消费的数据集产物，不进 git
- `data/splits/`：外部统一 split 索引或临时划分产物，不进 git
- `out/`：训练、评估、绘图、日志、矩阵结果，不进 git
- `replay_matrix/`：本地临时回收或人工整理的 replay 汇总目录，不进 git
- `runs/`：运行期目录，不进 git

## 3. Git 入库规则

当前仓库的 git 规则收口为：

- 必须提交：
  - `data/raw/<dataset_id>/` 下被当前配置实际引用的原始日志
  - `configs/dataset/*.yaml`
  - 与数据重建相关的设计文档与命令文档
- 不提交：
  - `data/interim/**`
  - `data/processed/**`
  - `data/splits/**`
  - `out/**`
  - `replay_matrix/**`
  - `runs/**`

这意味着：

- 原始数据是仓库的可复现实验输入
- 处理后数据是可重建 artifact，不是版本控制真源
- replay 结果若需要长期引用，应统一整理到 `out/replay_matrix/<variant>/`
- 本地手工回收、比对或临时汇总目录保持在仓库忽略区内

## 4. `raw/` 目录边界

`data/raw/` 只允许放原始底稿，不允许混放派生产物。

禁止放入 `raw/` 的内容包括：

- `_aligned.csv`
- `_aligned.png`
- `plots/`
- `offline_nav_outputs/`
- 任何根据原始日志二次导出的中间表、截图或分析目录

这类内容应放入：

- `data/interim/`
- `out/`

## 5. 当前重建契约

当前主线默认依赖：

1. `data/raw/<dataset_id>/...` 中的原始传感器日志
2. `configs/dataset/*.yaml` 中的文件选择与时间基配置
3. `configs/align/*.yaml` 与 `configs/fusion/*.yaml`
4. `configs/train/*.yaml`

典型重建顺序：

```bash
cd /home/wys/uwnav_dynamics
export PYTHONPATH=src

python -m uwnav_dynamics.preprocess.align.cli_align \
  -y configs/align/pooltest02.yaml

python -m uwnav_dynamics.preprocess.fusion.cli_fuse_train_base \
  -y configs/fusion/pooltest02_kf_eskf_v2.yaml

python -m uwnav_dynamics.preprocess.build_dataset \
  -y configs/dataset/pooltest02_s1_kf_ctx_quality_v3.yaml
```

完整命令参考：

- [reference/quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)
- [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)

## 5.1 Replay 结果目录约定

当前 replay 相关目录职责统一为：

- `out/replay_matrix/<variant>/`：正式运行产物目录，作为评估与汇报引用路径
- `replay_matrix/<variant>/`：本地临时回收、副本整理或人工比对目录，不进 git

若需要把服务器侧 replay 结果带回仓库，应优先整理到 `out/replay_matrix/` 契约下；
不要把临时汇总目录直接放在根目录等待入库。

## 6. 当前建议的最小审计信息

每个 `dataset_id` 至少应能回答下面四个问题：

1. 原始文件名是什么。
2. 当前配置实际选择了哪几个文件。
3. 这些文件通过什么命令被重建成 `train_base` 与 `processed dataset`。
4. 训练 run 使用了哪个 `data_dir` 与哪份配置。

当前仓库已经具备第 2、3、4 点的主链路信息：

- `configs/dataset/*.yaml`
- `resolved_train.yaml`
- `docs/reference/quick_commands.md`

后续若数据集继续增多，建议为每个 `dataset_id` 增加独立 manifest，记录：

- 文件清单
- 文件大小
- `sha256`
- 采集说明
- 对应配置文件

## 7. 当前已执行的目录清理

本次已把以下明显的派生产物从 `raw/` 挪出：

- `pwm_log_20260110_193725_aligned.csv`
- `pwm_log_20260110_193725_aligned.png`
- `offline_nav_outputs/`

它们现在位于 `data/interim/`，与 `raw/` 的职责分开。
