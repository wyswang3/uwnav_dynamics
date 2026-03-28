# 第 4.3 节数据与训练流程专项审查（B4+U1）

> 历史说明：本文档记录的是旧阶段 `B4+U1 / Vel_state / dvl_mask 稀疏速度监督` 方案，
> 不是当前 `KF / ESKF 融合 + VelKf dense supervision` 主线的实现说明。
> 当前训练主线请优先参考 `docs/design/kf_fusion_preprocess_training_v2.md`。

## 1. 审查结论先行

围绕“当前主线实验（B4+U1）使用的数据预处理流程、样本构造与输入输出定义、真实采用的训练配置”三项问题，当前代码与配置可以归纳为以下三点：

1. 当前主线数据链路是：`IMU 原始日志 -> IMU 预处理 -> 与 PWM / DVL / Power 对齐成 train_base.csv -> 构造 Vel_state 状态代理量 -> 滑动窗口 -> 训练时再做 split/scaler/mask`。其中 IMU 在对齐阶段采用**线性插值**，PWM 采用 **hold-last**，Power 采用 **hold-last + max_dt gating**，DVL 采用**最近邻 sparse attach**。
2. 当前样本定义不是“单步状态预测”，而是：用最近 `hist_len=100` 步历史，预测未来 `pred_len=10` 步、每步 `9` 维的观测状态代理量轨迹。输入 `X` 为 `(100,25)`，目标 `Y` 为 `(10,9)`。其中速度 3 维在张量里是**稠密状态代理量**，但在损失里只在 `dvl_mask=True` 的未来时刻参与监督。
3. 当前论文应写入的正式主线训练配置，不应直接抄基础 `configs/train/pooltest02_s1_lstm_v0.yaml`，而应写成：`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml` 的公共覆盖配置，加上 `configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed{6,7,8,9}.yaml` 的 B4+U1 结构开关。该配置使用 `AdamW + ReduceLROnPlateau + early stopping`，单卡单进程训练、8 卡并发矩阵调度，不是 DDP。

> 重要审计提醒  
> 代码注释与配置文本多次把 DVL 描述为“BI 10 Hz 稀疏 attach”。但当前主线实际使用的 `out/dvl_proc/dvl_nav_state_tb_20260110_193538_proc.csv` 不含 `Src/kind/used` 过滤字段，而 `aligner.py` 只有在这些字段存在时才会重新筛选 BI 行。因此，若论文要写成“当前训练严格使用 BI 10 Hz DVL 速度监督”，这一句需要标注“需人工确认”。

## 2. 审查对象与证据边界

本报告直接审查了以下真源：

- 对齐配置：`configs/align/pooltest02.yaml`
- 原始数据选择配置：`configs/dataset/pooltest02.yaml`
- 数据集构建配置：`configs/dataset/pooltest02_s1.yaml`
- 基础训练配置：`configs/train/pooltest02_s1_lstm_v0.yaml`
- 正式主线 launch：`configs/launch/pooltest02_s1_round3_top3_e120.yaml`、`configs/launch/pooltest02_s1_round4_controlconfirm_e120.yaml`、`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml`
- 正式主线生成训练配置：`configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed6.yaml`
- 预处理与训练代码：`src/uwnav_dynamics/preprocess/align/aligner.py`、`src/uwnav_dynamics/preprocess/build_dataset.py`、`src/uwnav_dynamics/preprocess/sliding_window.py`、`src/uwnav_dynamics/train/data_pipeline.py`、`src/uwnav_dynamics/dataset/split.py`、`src/uwnav_dynamics/train/run_train.py`、`src/uwnav_dynamics/cli/train_matrix.py`
- 当前数据产物与训练矩阵旁证：`data/processed/2026-01-10_pooltest02_s1/meta.yaml`、`features.npz`、`labels.npz`，以及 `out/train_matrix/pooltest02_s1_round5_finalconfirm_e120/{manifest.yaml,summary.csv,logs/*.log}`

本报告没有分析模型优劣，也不讨论 B4+U1 相比其他结构为什么更优；只回答当前主线**实际上如何做数据处理、如何构样本、如何训练**。

## 3. 多源传感器数据概况

### 3.1 当前主线涉及的数据源

| 数据源 | 当前配置/文件 | 训练中实际使用字段 | 频率旁证 | 在辨识任务中的角色 | 审计说明 |
| --- | --- | --- | --- | --- | --- |
| IMU | `configs/dataset/pooltest02.yaml` 选择 `imu/min_imu_tb_20260110_193348.csv`；对齐阶段使用 `out/imu_proc/min_imu_tb_20260110_193348_proc.csv` | 原始字段来自 `AccX/Y/Z`、`GyroX/Y/Z`、`AngX/Y/Z`；训练实际消费 `AccX_body_mps2..AccZ_body_mps2`、`GyroX_body_rad_s..GyroZ_body_rad_s` | 原始文件 `MonoNS` 中位步长约 `0.010214 s`，约 `97.9 Hz` | 稠密输入；同时是 9 维输出中的前 6 维稠密监督量 | 训练用的是**预处理后的 IMU 体坐标线加速度与角速度**，不是原始 `g/deg/s` |
| PWM 日志 | `configs/dataset/pooltest02.yaml` 选择 `logs/pwm_log_20260110_193725.csv`；对齐阶段使用 `out/pwm/..._cmd_aligned.csv` | 原始字段 `t_s`、`ch1_cmd..ch8_cmd`；读入器还支持 `ch*_applied`，但当前训练输入只使用 `ch*_cmd` | 原始/对齐文件中位步长约 `0.01 s`，约 `100 Hz` | 稠密控制输入，只进入 `X`，不进入 `Y` | PWM 时间轴不是直接原始绝对时间，而是通过 `ests0_pwm/ts0_pwm` 映射到统一 `EstS` 体系 |
| DVL | `configs/dataset/pooltest02.yaml` 选择 `dvl/dvl_nav_state_tb_20260110_193538.csv`；对齐阶段使用 `out/dvl_proc/dvl_nav_state_tb_20260110_193538_proc.csv` | 原始 CSV 的关键字段是 `Src`、`Vx_body(m_s)`、`Vy_body(m_s)`、`Vz_body(m_s)`、`Vu_enu(m_s)`、`Depth(m)`；训练基表里实际使用 `VelBx_body_mps..VelBz_body_mps`、`dvl_mask/has_dvl`，之后再构成 `VelX_state_mps..VelZ_state_mps` | 原始 `Src=BI` 子集的中位步长约 `0.09927 s`，约 `10.07 Hz`；但当前 `dvl_proc` 文件整体中位步长约 `0.00208 s` | 既是速度状态代理量的来源，也是速度维监督掩码 `dvl_mask` 的来源 | **不是直接把原始 DVL 稠密送入网络**；而是先构造 `Vel_state` 代理量，再通过 `dvl_mask` 决定速度损失何时有效 |
| 电机功率 | `configs/dataset/pooltest02.yaml` 选择 `volt/motor_data_20260110_193455.csv`；对齐阶段使用 `out/aux_power/motor_data_20260110_193455_power8.csv` | 原始字段是 `CH0..CH15` 和时间列；训练实际消费 `P0_W..P7_W` | 原始/对齐文件中位步长约 `0.2093 s`，约 `4.78 Hz` | 辅助输入，只进入 `X`，不进入 `Y`，也不参与 target mask | 当前 `P0_W..P7_W` 是**派生功率特征**，不是直接采样到的每路真实功率列 |

### 3.2 代码实现描述与论文建议表述

| 项目 | 代码实现描述 | 论文建议表述 |
| --- | --- | --- |
| IMU | 训练使用 `Acc*_body_mps2` 与 `Gyro*_body_rad_s`，它们由 IMU 原始日志经过坐标/单位统一、重力补偿、零偏估计和滤波后得到 | “IMU 提供经预处理的体坐标线加速度与角速度，用作稠密观测输入与稠密监督量” |
| DVL | 训练并不直接把原始 `Vx_body(m_s)` 作为稠密输入，而是先在 `train_base.csv` 中 attach 为 `VelB*`，再前向填充成 `Vel*_state_mps` | “DVL 主要用于构造速度状态代理量，并通过可用性掩码提供稀疏速度监督” |
| Power | `P0_W..P7_W` 由 `power_reader.py` 按固定 `12 V`、电流放大倍数 `40`、电机通道重排 `MOTOR_PERM` 计算得到 | “电机功率特征由电流日志按固定母线电压近似换算得到，可作为辅助输入” |

## 4. 数据预处理流程

### 4.1 IMU 预处理

当前训练使用的 IMU 不是原始 CSV 直接入模，而是先走完整 IMU 预处理管线：

1. `transform.py` 把原始 IMU 从 RFU 转为 FRD，把加速度从 `g` 转为 `m/s²`，把陀螺从 `deg/s` 转为 `rad/s`，把姿态角转为 `rad` 并统一 yaw 角范围。
2. `gravity.py` 根据 `roll/pitch/yaw` 计算体坐标重力分量，并从加速度测量中扣除重力，得到线加速度。
3. `bias.py` 用起始 `20 s` 静止窗估计加速度/角速度零偏，最小样本数为 `200`。
4. `filter.py` 对线加速度、角速度和 yaw 做去毛刺与一阶 IIR 低通。当前默认参数是：加速度截止频率 `5 Hz`、角速度 `8 Hz`、yaw `2 Hz`；毛刺阈值分别为 `3.0 m/s²`、`1.0 rad/s`、`0.5 rad`。

因此，论文如果要写 IMU 预处理，建议写成“坐标统一 + 单位统一 + 重力补偿 + 零偏校正 + 去毛刺/低通滤波”，而不是只写“做了简单对齐”。

### 4.2 主时间轴定义

当前主时间轴由 `aligner.py` 按如下规则确定：

1. 先取 IMU 与 PWM 时间范围的交集。
2. 若配置的 `dt_main_s` 与 IMU 中位步长足够接近（相对误差不超过 20%），则**直接复用 IMU 时间戳**。
3. 只有在不接近时，才回退为均匀时间网格。

当前 `configs/align/pooltest02.yaml` 设置的是 `dt_main_s=0.01`、`t_margin_s=0.0`。但当前实际 `out/train/2026-01-10_pooltest02_train_base.csv` 的统计量是：

```text
rows 115170
t0 1013.417457841  t1 2193.881126997
dt_med 0.010213748999831296
dt_min 0.0021378899996307155
dt_max 0.03310288199986644
```

这说明：

- 名义主步长是 `0.01 s`；
- 当前实际数据集**复用了近似 100 Hz 的 IMU 时间戳**；
- 因而主时间轴并不是“严格 100 Hz 均匀网格”。

论文建议表述：

- 可以写“主时间轴名义步长为 `0.01 s`，以 IMU 时间轴为主”；
- 不建议写成“所有信号被严格重采样到等间隔 100 Hz 均匀网格”。

### 4.3 不同频率数据如何统一到主时间轴

| 数据 | 对齐方式 | 代码依据 | 当前语义 |
| --- | --- | --- | --- |
| IMU | 线性插值到主时间轴 | `aligner.py` 中 `_interp_dense_to_main()` 与 `build_training_table_imu_main()` | 稠密连续观测；对齐后要求全 finite |
| PWM | `hold-last` | `aligner.py` 中 `_sample_last_before()` | 控制输入在每个主时间点使用最近历史命令 |
| DVL | 最近邻 sparse attach，允许时间差 `dvl_max_dt_s=0.10` | `aligner.py` 中 `_attach_sparse_to_main()` | 无命中则保留 `NaN`，并写 `dvl_mask/has_dvl` |
| Power | `hold-last + max_dt gating`，允许时间差 `power_max_dt_s=0.25` | `aligner.py` 中 `_sample_last_before_with_max_dt()` | 命中失败则置 `NaN` 并写 `power_mask` |

这里需要特别指出两个实现细节：

1. 虽然 `aligner.py` 顶部注释提到过“分箱平均”，但当前真正执行 IMU 对齐的是**线性插值**，不是 bin average。
2. Power 在 `train_base.csv` 中允许保留 `NaN`；这些 `NaN` 不会在 dataset build 阶段被清洗，而是在训练消费端先按 train split 拟合 scaler，再把 z-score 后的非有限值置为 `0.0`。

### 4.4 DVL 如何进入训练流程

DVL 是当前第 4.3 节最容易写错的部分。当前真实链路可以分为三层：

1. **对齐层**  
   `aligner.py` 把 DVL 预处理文件的 `VelBx_body_mps..VelBz_body_mps` attach 到主时间轴，未命中位置保留 `NaN`，并输出 `dvl_mask` 与 `has_dvl`。
2. **状态代理量层**  
   `build_dataset.py` 用 `VelBx_body_mps..VelBz_body_mps` 构造 `VelX_state_mps..VelZ_state_mps`：有 DVL 时直接取该值；无 DVL 时前向填充；若序列开头尚无 DVL，则初始化为 `0.0`。
3. **训练监督层**  
   `train/data_pipeline.py` 并不直接把原始 `dvl_mask` 当成最终 loss mask，而是先恢复 9 维语义布局，再把 `dvl_mask` 广播到 velocity 语义组，生成 `target_mask:(N,H,9)`。结果是：
   `target_mask[:,:,0:6] = True`，`target_mask[:,:,6:9] = dvl_mask`。

因此，当前主线中 DVL 的角色不是单选题，而是：

- 不是“直接原始输入”；
- 不是“只做稀疏监督”；
- 而是“**先作为速度状态代理量来源进入输入与目标，再作为速度维监督掩码控制损失有效性**”。

### 4.5 稠密输入、稠密监督、稀疏监督的区分

| 量 | 是否进入 `X` | 是否进入 `Y` | 是否始终参与 loss | 说明 |
| --- | --- | --- | --- | --- |
| PWM 8 维 | 是 | 否 | 否 | 纯控制输入 |
| IMU Acc/Gyro 6 维 | 是 | 是 | 是 | 稠密输入，也是稠密监督 |
| `Vel*_state_mps` 3 维 | 是 | 是 | 否 | 张量内是稠密状态代理量，但只有 `dvl_mask=True` 时参与速度监督 |
| `P0_W..P7_W` 8 维 | 是 | 否 | 否 | 辅助输入；缺测时 runtime 置为 z-score 空间 0 |
| `dvl_mask/has_dvl` | 否 | 否 | 间接参与 | 不直接入模，只用于构造 `target_mask` |
| `power_mask` | 否 | 否 | 否 | 当前只作为辅助 metadata，不参与 `target_mask` |

### 4.6 当前 DVL 实现中的关键不确定点

从原始 DVL 文件与当前 `dvl_proc_csv` 的现场核对看：

```text
raw_dvl_BI rows 24375 dt_med 0.09927131399990685 hz_med 10.0734034808982
dvl_proc rows 146250 dt_med 0.002077516000099422 hz_med 481.34406664119257
```

原始 DVL 文件含多种 `Src`：

```text
src_counts_top {'BI': 24375, 'BS': 24375, 'BE': 24375, 'SA': 24375, 'BD': 24375, 'TS': 24374, 'T0': 1}
```

而当前 `dvl_proc_csv`：

- 不再保留 `Src`；
- 不包含 `kind` 或 `used` 列；
- `valid` 列全为 `0`；
- `aligner.py` 也就无法在对齐阶段重新筛出 BI。

结合当前 `train_base.csv` 的统计：

```text
dvl_mask_ratio 0.20682469393071112
```

可得出审计结论：

- “代码注释的意图”是把 DVL 当作低频稀疏速度观测；
- “当前主线真实实现”是直接消费了 `dvl_proc_csv` 中的所有已处理速度行，再通过 `dvl_mask` 控制命中位置；
- 若论文要严格表述为“使用 BI 10 Hz DVL 速度”，应标注“需人工确认”。

论文建议表述：

- 可以写“DVL 速度通过对齐后速度列和对应可用性掩码进入训练”；
- 不建议在未人工复核前写成“对齐阶段显式仅保留 BI 10 Hz 速度真值”。

## 5. 样本构造与输入输出定义

### 5.1 基本窗口参数

当前数据集构建参数由 `configs/dataset/pooltest02_s1.yaml` 与 `data/processed/2026-01-10_pooltest02_s1/meta.yaml` 共同确认：

- 主时间轴名义步长：`0.01 s`
- 历史窗口长度 `hist_len=100`
- 预测长度 `pred_len=10`
- 滑窗步长 `stride=1`
- `drop_incomplete=true`
- `valid_mask_col=null`，即**滑窗阶段不因 DVL 稀疏而丢窗**

因此，窗口切片规则是：

```text
X_i = base[i : i+100, input_cols]
Y_i = base[i+100 : i+110, target_cols]
```

若基础表行数为 `N_base=115170`，则完整窗口数为：

```text
N_win = 115170 - (100 + 10) + 1 = 115061
```

这与当前 `meta.yaml` 中的 `n_windows: 115061` 一致。

### 5.2 单个样本的输入张量 `X`

当前 `X` 的语义是 `(hist_len, din) = (100, 25)`，列顺序固定为：

1. PWM 8 维：`ch1_cmd..ch8_cmd`
2. IMU 6 维：`AccX/Y/Z_body_mps2`、`GyroX/Y/Z_body_rad_s`
3. 速度状态代理量 3 维：`VelX/Y/Z_state_mps`
4. 功率辅助量 8 维：`P0_W..P7_W`

因此：

- `din=25`
- `u_in_idx = [0..7]` 对应 PWM 控制输入
- `y_in_idx = [8..16]` 对应输入张量中最后一个历史时刻的 9 维观测状态代理量

### 5.3 单个样本的目标张量 `Y`

当前 `Y` 的语义是 `(pred_len, dout) = (10, 9)`，列顺序固定为：

1. `AccX/Y/Z_body_mps2`
2. `GyroX/Y/Z_body_rad_s`
3. `VelX/Y/Z_state_mps`

因此：

- `dout=9`
- `Y` 不是“下一时刻单个状态”，而是**未来 10 步 9 维观测状态代理量轨迹**

更准确地说，当前论文应把它表述为：

```text
[a_x, a_y, a_z, ω_x, ω_y, ω_z, v_x^state, v_y^state, v_z^state]_{k+1:k+10}
```

其中：

- 前 6 维是 IMU 观测量；
- 后 3 维是由 DVL 体速度经前向填充得到的速度状态代理量；
- 速度 3 维虽然在 `Y` 中是逐时刻都存在的稠密数值，但损失只在对应未来时刻 `dvl_mask=True` 时才激活。

### 5.4 “未来 10 步 9 维观测状态代理量轨迹”的准确含义

当前代码不是直接预测绝对 `Y`，而是：

1. 从 `X` 最后一个历史时刻用 `y_in_idx=[8..16]` 提取 `y0`；
2. 网络输出未来 10 步的状态增量 `dY`；
3. 通过 `rollout_from_delta()` 执行 `y_hat = y0 + cumsum(dY)`。

因此，所谓“未来 10 步 9 维观测状态代理量轨迹”，准确含义是：

- 初值：历史窗最后一帧的 9 维观测状态代理量；
- 预测对象：其后连续 10 个未来时刻的 9 维增量累积结果；
- 语义上属于“多步观测状态代理量 rollout”，而不是“单步回归”。

### 5.5 真实稠密监督量与稀疏监督量

| 监督组 | 张量维度 | 在 `Y` 中是否稠密存在 | 在 loss 中是否稠密有效 | 备注 |
| --- | --- | --- | --- | --- |
| Acc 3 维 | `Y[:,:,0:3]` | 是 | 是 | 稠密监督 |
| Gyro 3 维 | `Y[:,:,3:6]` | 是 | 是 | 稠密监督 |
| Vel_state 3 维 | `Y[:,:,6:9]` | 是 | 否 | 稠密存储、稀疏监督 |

这一定义非常关键。当前实现里“速度量”同时具有两层属性：

- **作为张量内容时**：它是由 DVL 前向填充得到的稠密状态代理量；
- **作为监督语义时**：它只在 `dvl_mask=True` 的未来时刻参与 loss。

因此论文不应写成“速度标签是完整稠密真值”，也不应写成“速度完全不进入 `Y`”；二者都不准确。

### 5.6 速度分量与 DVL 掩码之间的关系

训练时 `target_mask` 的构造逻辑是：

```text
target_mask[:,:,0:6] = True
target_mask[:,:,6:9] = dvl_mask
```

这意味着：

- `dvl_mask` 只控制 velocity semantic group；
- `acc/gyro` 两组不受 DVL 掩码影响；
- `power_mask` 当前不进入 `target_mask`。

换言之，当前主线不是“整条 `Y` 都 masked”，而是**只有速度监督被 DVL 可用性裁决**。

### 5.7 训练/验证/测试集划分与 leakage 防护

当前训练主链路使用 `contiguous_purged_v2`，不是随机打散，也不是简单窗口级 contiguous split。

真实规则是：

1. 先在**原始行轴**上按 `train=0.70 / val=0.15 / test=0.15` 划出三段；
2. 只保留“完整 `L+H=110` 步跨度完全落在本段内部”的窗口；
3. 所有跨边界窗口都会 purge 掉。

当前数据集按此规则得到：

```text
train 80510
val 17166
test 17167
dropped 218
train_raw_end 80619
val_raw_end 97894
```

因此，论文应写“连续时间段划分并对跨边界窗口做 purge”，而不是“随机划分窗口样本”。

另一个容易写错的点是：`seed` 不参与 split 随机化。当前 `seed` 只影响模型初始化、DataLoader shuffle 与训练随机性，不改变 split 边界。

## 6. 正式主线训练配置

### 6.1 基础默认配置

基础默认配置来自 `configs/train/pooltest02_s1_lstm_v0.yaml`。它本质上是一个 **B0 baseline + 本地 smoke 默认值**：

- `run.amp: false`
- `data.batch_size: 256`
- `data.pin_memory: false`
- `train.epochs: 3`
- `model.blocks.*.enabled: false`
- `variant: B0_baseline`

因此，这份文件不能直接当作论文中的主线训练设置。

### 6.2 正式主线覆盖配置

当前正式主线训练配置应从两层覆盖后确定：

第一层，`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml` 的公共覆盖：

- `run.device: cuda`
- `run.amp: true`
- `run.out_dir: out/ckpts/pooltest02_s1_round5_finalconfirm_e120`
- `data.data_dir: data/processed/2026-01-10_pooltest02_s1`
- `data.batch_size: 512`
- `data.num_workers: 4`
- `data.pin_memory: true`
- `train.epochs: 120`
- `launcher.gpus: [0,1,2,3,4,5,6,7]`
- `launcher.max_parallel: 8`
- `launcher.run_eval: true`
- `launcher.eval_batch_size: 1024`

第二层，B4+U1 运行项的结构覆盖：

- `thruster_lag.enabled: true`
- `hydro_ssm.enabled: true`
- `uncertainty.enabled: true`
- `damping.enabled: false`
- 仅种子和 variant 名在 `seed6~9` 间变化

### 6.3 论文中应写入的真实主线训练设置

若论文第 4.3.3 需要写“正式主线训练配置”，当前应写成如下口径：

| 项目 | 当前应写入论文的配置 |
| --- | --- |
| 正式配置组 | `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml` + `configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed{6,7,8,9}.yaml` |
| 训练设备 | `cuda` 单卡单进程；8 卡并发矩阵调度 |
| 并行方式 | 8 个独立训练任务并发，每个任务通过 `CUDA_VISIBLE_DEVICES` 绑定一张 GPU；**不是 DDP** |
| 优化器 | `AdamW` |
| 学习率 | `1e-3` |
| 权重衰减 | `1e-4` |
| 梯度裁剪 | `1.0` |
| 调度器 | `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)` |
| batch size | 训练 `512`；评估 `1024` |
| 最大 epoch | `120` |
| eval 频率 | 每 epoch 一次 |
| early stopping | `patience=12`, `min_delta=1e-4` |
| AMP | 启用 |
| blocks | `ThrusterLag + HydroSSM + UncertaintyHead` 开；`DampingHead` 关 |
| DVL 辅助头 | 未启用 |

需要特别区分“配置上限”和“实际运行行为”：

- 配置上限是 `120 epoch`；
- 但当前 round5 的 B4+U1 四个正式 run 都触发了 early stopping，`epochs_ran` 分别为 `17/17/17/16`；
- 这四个 run 的 `final_lr` 都降到了 `2.5e-4`。

因此，若论文要描述训练过程，更准确的写法是：

- “最大训练轮数为 120，配合 `ReduceLROnPlateau` 与 early stopping；在当前正式主线 runs 中，B4+U1 实际于第 16~17 轮停止。”

### 6.4 关于“8 卡 4090 服务器”的写法

当前代码、manifest 与日志可以确认：

- 使用了 `gpus=[0..7]`
- `max_parallel=8`
- 单个任务只占用一张卡

但当前代码与配置**不能证明 GPU 型号就是 4090**。因此建议：

- 若服务器型号信息来自人工提供，可以写“训练在 8 卡 GPU 服务器上完成（型号信息来自实验平台记录）”；
- 若只基于代码侧证据，应写“代码可确认采用 8 卡并发训练矩阵，但 GPU 具体型号需人工确认”。

## 7. 面向论文写作的提炼结果

### 7.1 适合写入 4.3.1 的“数据预处理与辨识任务定义”摘要

当前主线实验首先对 IMU 原始日志执行坐标与单位统一、重力补偿、零偏估计以及去毛刺/低通滤波，得到体坐标系下的线加速度与角速度序列；随后以名义 `0.01 s` 步长、实际复用 IMU 时间轴的主时间轴为基准，将 PWM 命令采用零阶保持对齐，将 DVL 速度采用最近邻稀疏写入，并将电机功率采用低频零阶保持加时间窗 gating 的方式附着到同一时间轴上。在此基础上，利用 DVL 速度列前向填充构造速度状态代理量 `Vel_state`，与 IMU 观测量共同组成 9 维观测状态代理量。辨识任务据此定义为：输入最近 100 步的控制量、IMU 观测量、速度状态代理量和功率辅助量，预测未来 10 步 9 维观测状态代理量轨迹，其中加速度与角速度为稠密监督，速度分量仅在 DVL 可用时刻通过掩码参与稀疏监督。

### 7.2 适合写入 4.3.3 的“正式训练配置说明”摘要

正式主线训练配置采用 `pooltest02_s1_round5_finalconfirm_e120` 实验矩阵中的 B4+U1 分支：在 `S1Predictor` 主干上同时启用 `ThrusterLag`、`HydroSSM` 与 `UncertaintyHead`，关闭 `DampingHead` 与 DVL 辅助观测头；优化器为 `AdamW`，初始学习率 `1e-3`，权重衰减 `1e-4`，梯度裁剪阈值 `1.0`，学习率调度为 `ReduceLROnPlateau`，训练最大轮数为 `120`，并使用 `patience=12` 的 early stopping。训练在 CUDA 环境下进行，启用 AMP，单个 run 的 batch size 为 `512`，整个主线筛选过程采用 8 张 GPU 的并发矩阵调度方式，但每个模型训练任务仍为单卡单进程而非 DDP。

### 7.3 论文中容易写错的点

1. 不要把 `configs/train/pooltest02_s1_lstm_v0.yaml` 直接当作正式主线训练配置。它是基础默认值；正式主线是 round5 launch 覆盖后的 B4+U1 生成配置。
2. 不要把 `train.epochs: 120` 写成“实际都训练了 120 轮”。当前正式 B4+U1 runs 实际在第 `16~17` 轮就 early stop 了。
3. 不要把当前任务写成“单步预测”。`pred_len=10`，真实任务是未来 10 步序列预测。
4. 不要把 `Vel_state` 写成“完整稠密真速度标签”。它是由 DVL 前向填充得到的速度状态代理量，速度损失只在 `dvl_mask=True` 时生效。
5. 不要把 DVL 简单写成“只作为输入”或“只作为监督”。当前实现同时使用了 DVL 派生的状态代理量和 DVL 可用性掩码。
6. 不要把 `power_mask` 写成训练损失掩码。当前 `power_mask` 只做辅助 metadata，真正进入 `target_mask` 的是 `dvl_mask`。
7. 不要把 processed 数据集写成“已经标准化完成”。`output.normalize: standard` 在 dataset build 阶段并未真正执行，标准化被延后到训练 split 上拟合 scaler。
8. 不要把主时间轴写成“严格均匀 100 Hz”。当前实现名义步长为 `0.01 s`，但实际复用了接近 100 Hz 的 IMU 时间戳。
9. 不要在未人工复核前写“当前训练严格只使用 BI 10 Hz DVL 速度”。当前配置与现有 `dvl_proc` 产物之间存在语义不完全闭合的问题，应标为“需人工确认”。
10. 不要把 8 卡并发实验矩阵写成 DDP 训练。当前是 8 个独立单卡任务并发运行。

## 8. 关键证据索引

- IMU 预处理顺序：`src/uwnav_dynamics/preprocess/imu/pipeline.py`
- IMU 线性插值到主时间轴：`src/uwnav_dynamics/preprocess/align/aligner.py`
- PWM `hold-last`：`src/uwnav_dynamics/preprocess/align/aligner.py`
- DVL sparse attach 与 `dvl_mask`：`src/uwnav_dynamics/preprocess/align/aligner.py`
- `Vel_state` 前向填充与初始零填充：`src/uwnav_dynamics/preprocess/build_dataset.py`
- 滑窗切片规则：`src/uwnav_dynamics/preprocess/sliding_window.py`
- runtime `target_mask` 只作用于速度语义组：`src/uwnav_dynamics/supervision_mask.py`
- purged split：`src/uwnav_dynamics/dataset/split.py` 与 `src/uwnav_dynamics/train/data_pipeline.py`
- 正式主线 launch 与 8 卡矩阵：`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml`、`src/uwnav_dynamics/cli/train_matrix.py`
- 当前正式主线生成配置：`configs/train/generated/pooltest02_s1_round5_finalconfirm_e120/b4u1_seed6.yaml`
- 当前正式主线运行旁证：`out/train_matrix/pooltest02_s1_round5_finalconfirm_e120/{manifest.yaml,summary.csv,logs/b4u1_seed6.log}`
