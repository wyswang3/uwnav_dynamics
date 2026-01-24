
---

## 1) `docs/design/dataset_spec.md`

```md
# Dataset Specification (uwnav_dynamics)

本规范定义 uwnav_dynamics 项目中 raw 数据的组织方式、数据寻址规则、时间轴选择策略、以及“有效工作时间区间”的确定方法。
本规范旨在支持：本地 Ubuntu 先跑通 → 服务器 GPU 加速 → 后续增加更多实验数据而无需修改核心代码。

---

## 1. Data Root and Directory Layout

### 1.1 Data Root Priority

项目中数据根目录 `data_root` 的解析优先级如下（从高到低）：

1. 环境变量 `UW_DYN_DATA_ROOT`
2. 配置文件 `configs/dataset/*.yaml` 中的 `dataset.data_root`
3. 默认值：仓库根目录下的 `./data`

### 1.2 Raw Directory Layout

在 `data_root` 下，raw 数据必须遵循以下布局（固定约定）：

```

{data_root}/raw/{dataset_id}/
imu/
dvl/
logs/        # pwm
volt/
offline_nav_outputs/   # 可选：参考轨迹/诊断产物

```

其中 `{dataset_id}` 由实验日期与编号组成，例如：`2026-01-10_pooltest02`。

---

## 2. Dataset ID and File Selection

### 2.1 Dataset ID

`dataset_id` 定义为 `raw/` 下的一层目录名，必须唯一对应一次实验数据集合，例如：

- `2026-01-06_pooltest01`
- `2026-01-10_pooltest02`

### 2.2 File Selection

每个传感器的数据文件通过 dataset config 显式指定：

- `selection.imu.file`：相对于 `{dataset_id}` 的子路径，如 `imu/min_imu_tb_20260110_192246.csv`
- `selection.dvl.file`：如 `dvl/dvl_parsed_*.csv`
- `selection.pwm.file`：如 `logs/pwm_log_*.csv`
- `selection.volt.file`：如 `volt/motor_data_*.csv`

第一阶段（预处理与绘图）允许只指定 IMU，其余为空。

---

## 3. Timestamp Policy

### 3.1 Time Column Priority

对所有带时间列的 CSV，统一采用如下优先级选择主时间列（从高到低）：

1. `MonoNS`
2. `MonoS`（读取后转换为 ns：`t_ns = MonoS * 1e9`）
3. `EstNS`
4. `EstS`（读取后转换为 ns：`t_ns = EstS * 1e9`）

若上述列均不存在，允许 fallback 为“样本索引 + 名义采样率”，但必须在 QA 报告中标记为 `time_source=fallback`。

### 3.2 Mono vs Est Semantics

- `Mono*`：单调时钟（推荐用于所有对齐、裁剪、dt 估计）
- `Est*`：估计时间（可能包含外部对齐偏移），仅用于参考或可视化对照，不作为主对齐依据

---

## 4. File-Name Wall-Clock Anchor (Coarse Alignment)

许多文件名包含墙钟时间戳 `YYYYMMDD_HHMMSS`，例如：

- `min_imu_tb_20260110_192246.csv`
- `pwm_log_20260110_192435.csv`

该时间戳用于构造“粗对齐锚点”：

- `T0_wall(sensor_file) = parse(YYYYMMDD_HHMMSS)`

粗对齐只用于估计不同传感器记录段的重叠关系，不用于 sample-level 精确对齐。

---

## 5. Valid Working Time Window (Intersection Rule)

### 5.1 IMU-anchored Intersection

第一阶段采用 IMU 为主轴定义有效工作时间。设每个传感器 `s` 的绝对时间窗为：

- `T_start^s, T_end^s`

其中 IMU 的绝对时间窗可以通过：
- 文件名墙钟锚点 `T0_wall^imu`
- IMU CSV 内主时间列 `t_ns` 的起止
组合得到（实现层面可选择：以 `t_ns` 直接作为相对时间，或根据工程记录定义 offset）

有效工作区间定义为传感器时间窗的交集：

\[
[T_{valid}^{start}, T_{valid}^{end}] = \bigcap_s [T_{start}^s, T_{end}^s]
\]

第一阶段若仅启用 IMU，则：

\[
[T_{valid}^{start}, T_{valid}^{end}] = [T_{start}^{imu}, T_{end}^{imu}]
\]

### 5.2 Safety Margin Cropping

为避免启停瞬态与缓冲区影响，交集时间窗进一步做保守裁剪：

- `T_valid_start += margin_start_s`
- `T_valid_end   -= margin_end_s`

默认建议：`margin_start_s = 3s`, `margin_end_s = 3s`（由 dataset config 提供）。

---

## 6. Sensor-Specific Units and Conventions (Minimal)

### 6.1 IMU (min_imu_tb)

- 加速度 `AccX/Y/Z`：单位为 `g`，转换为 `m/s^2` 时乘 `9.78`
- 角速度 `GyroX/Y/Z`：单位为 `deg/s`，转换为 `rad/s` 时乘 `pi/180`
- 姿态角 `AngX/Y/Z`：单位为 `deg`，转换为 `rad` 时乘 `pi/180`
- yaw wrap：统一到 `(-pi, pi]`

---

## 7. Required QA Outputs (Phase-1)

任何传感器的 reader 在输出数据时必须同时输出 QA 元信息：

- chosen time column name
- N, duration, start/end time
- dt statistics: median, p95, max
- monotonicity check results
- NaN/Inf ratio per column
- basic stats for key channels (mean/std/p95|x|)
```

---

## 2) `docs/design/plots_spec.md`（第一阶段：IMU 绘图清单）

```md
# Plot Specification (Phase-1: IMU)

本规范定义第一阶段（先从 IMU min_tb 数据入手）的绘图清单与风格要求。
目标：快速完成数据 QA、识别异常、并输出可直接进入 SCI/报告的标准化图像。

---

## 1. General Style (SCI-like)

- 统一导出：`PDF`（论文）+ `PNG`（快速浏览）
- 图尺寸建议：
  - 单列图：宽 3.4–3.6 in
  - 双列图：宽 7.0–7.2 in
- 线宽：1.2–1.8
- 字体：默认无衬线或 Times 系（后续统一）
- 网格：轻网格（可选）
- 轴标签必须包含单位，例如：
  - `a_x (m/s^2)`
  - `ω_z (rad/s)`
- 标题包含：`dataset_id + file stem`
- 文件名中包含关键信息：传感器、字段、时间范围（可选）

---

## 2. Required Plots (IMU min_tb)

输入文件：`min_imu_tb_YYYYMMDD_HHMMSS.csv`

### 2.1 Timebase Diagnostics

**P1: dt histogram / dt over time**
- 子图 1（推荐）：`dt (ms)` vs `time (s)`（散点或细线）
- 子图 2（可选）：`dt` 直方图
目的：验证 100Hz 是否稳定，有无明显跳变/丢包。

### 2.2 Acceleration Time Series (Raw Units -> SI)

**P2: a_x, a_y, a_z vs time**
- 输入：AccX/Y/Z (g)
- 处理：转换为 m/s^2
- 输出：三条曲线（或三张图，按你喜好）
目的：查看 bias、噪声、饱和、异常段。

### 2.3 Gyro Time Series (deg/s -> rad/s)

**P3: ω_x, ω_y, ω_z vs time**
- 输入：GyroX/Y/Z (deg/s)
- 处理：转换为 rad/s
目的：查看静止噪声、漂移、突变。

### 2.4 Attitude Angles (deg -> rad) + Wrap

**P4: roll, pitch, yaw vs time**
- 输入：AngX/Y/Z (deg)
- 处理：转换为 rad；yaw wrap 到 (-pi, pi]
目的：确认姿态角连续性、是否存在跳变、yaw 是否符合 ENU 约定。

### 2.5 Basic Distribution & Outlier Check

**P5: distribution plots (hist or KDE alternative)**
- 对象：`a_x, a_y, ω_z, yaw`
- 输出：直方图（推荐）+ 标注 p95|x|
目的：快速估计噪声水平、异常值范围，给后续 loss/uncertainty 初值提供依据。

### 2.6 Stationary Window QA (Optional but Recommended)

**P6: first T seconds statistics**
- 设定 `T=20s`（可配置）
- 输出：窗口内 mean/std/p95|x|（可视化为条形图或表格）
目的：验证 bias-window 是否成立，为后续 bias 去除策略提供证据。

---

## 3. Output Paths Convention

Phase-1 绘图输出建议统一放在：

- `runs/<run_id>/plots/imu/`

其中 `<run_id>` 至少包含：
- dataset_id
- sensor kind（min_tb）
- file time stamp
- git commit hash（后续加入）

---

## 4. Minimal Plot Checklist Summary

必须输出（最少）：
- P1 dt诊断
- P2 acceleration
- P3 gyro
- P4 attitude
- P5 分布图

可选增强：
- P6 静止窗统计图/表
```

---
