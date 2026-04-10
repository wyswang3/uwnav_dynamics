# 数据契约说明

## 1. 文档目标

本文档定义 `uwnav_dynamics` 当前工程使用的数据契约。
它关注的是“数据如何被组织、解释和消费”，而不是具体实验结果。

适用范围：

- 原始日志组织
- 时间戳优先级
- 传感器字段与单位约定
- 多频对齐的工程假设
- 数据集产物约定

## 2. 数据根目录

仓库当前默认使用如下数据分层：

```text
data/
  raw/
  interim/
  processed/
  splits/
```

推荐语义：

- `raw/`：原始日志，只读保存，作为 git 入库真源
- `interim/`：中间预处理与对齐产物，可重建，不入 git
- `processed/`：训练直接消费的数据集产物，可重建，不入 git
- `splits/`：若需要外部统一 split 索引，可在此保存，默认不入 git

更具体的 git 管理与重建规则见
[data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)。

## 3. 日志命名规则

建议每次实验数据使用统一 `dataset_id`：

```text
YYYY-MM-DD_<experiment_name>
```

例如：

- `2026-01-10_pooltest02`

同一次实验下的传感器文件建议按目录隔离：

```text
raw/<dataset_id>/imu/
raw/<dataset_id>/dvl/
raw/<dataset_id>/logs/
raw/<dataset_id>/volt/
```

## 4. 传感器角色与频率

当前项目默认处理：

- PWM：100 Hz，推进器输入命令
- IMU：100 Hz，高频主观测
- DVL：10 Hz，速度相关低频观测
- Power：5 Hz，低频辅助特征

该频率关系决定了后续对齐与监督策略：

- PWM / IMU 面向主时间轴密集对齐
- DVL 保持稀疏语义
- Power 以低频 hold-last 辅助方式接入

## 5. 时间戳优先级

对于包含多种时间列的 CSV，推荐优先级为：

1. `MonoNS`
2. `MonoS`
3. `EstNS`
4. `EstS`

基本原则：

- 对齐与裁剪优先使用单调时钟
- `Est*` 更适合作为对外时间参考，而不是主对齐依据
- 若使用秒单位列，读入后应统一转换为纳秒或浮点秒的单一内部表达

## 6. 字段与单位

最低要求：

- 列名应具有稳定语义，不依赖外部口头解释
- 单位必须在配置、脚本或文档中有明确约定
- 若原始日志单位不一致，必须在预处理阶段归一化

当前最重要的语义约束包括：

- PWM 通道顺序固定
- IMU 加速度与角速度单位明确
- DVL 速度列应与坐标系转换策略绑定说明
- Power 特征应明确是功率、电流还是衍生量

## 7. 多频对齐假设

当前对齐假设如下：

- 主时间轴为 100 Hz
- PWM 以 hold-last 接入主轴
- IMU 作为高频主观测对齐到主轴
- DVL 只在命中位置写值，并保留 mask 语义
- Power 低频 hold-last 到主轴

这意味着：

- 不应把 DVL 强行插值为密集真值
- 缺失观测与真实数值很小必须区分
- 后续 mask-aware 训练评估应直接继承该对齐语义

## 8. 数据集产物

当前训练数据集的典型产物包括：

- `features.npz`
- `labels.npz`
- `meta.yaml`

一次训练 run 还会额外关联：

- `split_indices.npz`
- `x_scaler.npz`
- `y_scaler.npz`

## 9. 与工程配置的关系

本数据契约主要通过以下配置文件参与工程：

- `configs/dataset/*.yaml`
- `configs/align/*.yaml`
- `configs/train/*.yaml`

相关实现入口包括：

- `src/uwnav_dynamics/preprocess/align/`
- `src/uwnav_dynamics/preprocess/build_dataset.py`
- `src/uwnav_dynamics/train/data_pipeline.py`

## 10. 后续扩展方向

未来如果继续扩展数据契约，优先级建议为：

1. 明确 DVL / Power 的 mask 语义
2. 把观测字段 schema 与单位定义写成可校验规则
3. 为正式开源准备更标准的数据字典与样例数据说明
