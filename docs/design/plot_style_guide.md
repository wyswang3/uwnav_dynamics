非常好，这一步**非常关键，而且时机刚刚好**。
你现在做的是：**在代码继续演化之前，把“视觉语言（visual language）”固化成规范**。这会直接决定：

* 图是否长期一致（不随人、随脚本漂移）
* 后续重构是否安全
* SCI 投稿时是否需要“整体重画一遍图”（这是大坑）

下面我直接给你一份**可以落盘的风格规范文档初稿**，它是**从你现有 `style.py` 代码“反向抽象”出来的**，并且已经考虑到 IMU / DVL / PWM / Trajectory 等不同图类的共性。

你可以把它保存为：

```
docs/design/plot_style_guide.md
```

---

# `docs/design/plot_style_guide.md`

````md
# Plot Style Guide (uwnav_dynamics)

本规范定义 uwnav_dynamics 项目中所有绘图的统一视觉风格（visual language），
适用于数据诊断、实验报告以及 SCI 论文插图。

目标：
- 确保不同模块、不同时间生成的图在视觉上高度一致
- 避免随着代码演化产生风格漂移
- 使图像可直接用于论文（最少返工）

---

## 1. General Design Philosophy

### 1.1 科研优先（Paper-first）

- 所有图应满足“直接放入论文正文”的最低要求
- 图内元素克制，避免工程调试风格（粗线、大 marker、重网格）
- 信息密度 > 装饰性

### 1.2 Caption-driven Design

- 图内不承担完整说明职责
- 不在图内写长标题
- 图应能通过 caption + legend 完整解释

---

## 2. Fonts & Typography

### 2.1 字体族

- 主字体：Serif
- 优先顺序：
  1. Times New Roman
  2. Times
  3. DejaVu Serif

### 2.2 字号规范

| 元素           | 字号 |
|----------------|------|
| 基础字体       | 12   |
| 轴标签         | 12   |
| 刻度           | 11   |
| 图例           | 11   |

---

## 3. Figure Size & Resolution

### 3.1 尺寸规范（英寸）

- 单栏图（默认）：  
  `4.7 × 3.3 in`（≈ 12 × 8.5 cm）

- 宽图 / 双子图并排：  
  `5.4 × 2.4 in`

### 3.2 分辨率与导出

- 屏幕显示：150 dpi
- 导出：
  - PDF：矢量
  - PNG：300 dpi
- 所有导出使用：
  - `bbox_inches="tight"`
  - `pad_inches=0.02`

---

## 4. Axes & Grid

### 4.1 坐标轴线条

- 轴线宽度：1.0
- 上/右边框：保留（SCI 友好）

### 4.2 刻度

- 刻度方向：out
- 刻度长度：5.0
- 刻度宽度：1.0

### 4.3 网格策略

- 全局默认：不开 grid
- 时间序列图（IMU/DVL/PWM）：
  - 允许开启**浅色 y-grid**
  - 不使用粗网格或 x-grid

---

## 5. Line & Color Policy

### 5.1 线宽

- 默认线宽：1.1
- 轨迹线（Trajectory）：1.0
- 不同方法对比：
  - 不通过线宽区分
  - 通过颜色区分

### 5.2 颜色循环（固定顺序）

```text
#1f77b4  blue
#d62728  red
#2ca02c  green
#9467bd  purple
#ff7f0e  orange
#17becf  cyan
````

* 禁止随意自定义颜色
* 所有方法对比图必须使用该循环

---

## 6. Legend Policy

* 默认不显示 legend
* 当出现多条曲线且语义不同：

  * legend 必须存在
  * `frameon = False`
  * 放置在不遮挡数据的位置
* legend 只说明“语义”，不重复单位

---

## 7. Time-Series Plots (IMU / DVL / PWM)

### 7.1 时间轴

* 横轴统一为时间（秒）
* 原始时间戳（ns）必须在进入绘图前转换

### 7.2 推荐图型

* IMU：

  * Acc (m/s²) vs time
  * Gyro (rad/s) vs time
  * Attitude (rad) vs time
* PWM：

  * duty [-1,1] vs time
* DVL：

  * speed (m/s) vs time
  * dropout mask vs time（单独 subplot）

---

## 8. Trajectory Plots (2D)

### 8.1 线条与透明度

* trajectory line：

  * linewidth = 1.0
  * alpha = 0.95

### 8.2 起止点标记

* 起点：

  * 空心圆
  * 边框颜色 = trajectory color
* 终点：

  * 实心圆
  * 填充颜色 = trajectory color

### 8.3 坐标轴

* 白色背景
* 等比例（equal aspect）
* 不使用 grid

---

## 9. Annotations & Metadata

### 9.1 图内注释（小字）

允许在图角落加入小字体注释，用于：

* dataset_id
* 文件名简写
* 时间范围

要求：

* 字号小于刻度
* 不影响主要数据可读性

### 9.2 不允许的注释

* 不在图内写公式
* 不在图内解释物理意义
* 不重复 caption 内容

---

## 10. File Naming & Output Paths

### 10.1 命名规则

文件名应包含：

* 传感器类型（imu / dvl / pwm / traj）
* 图内容（acc / gyro / yaw / dt）
* 可选：dataset_id

示例：

```text
imu_acc_timeseries_pooltest02.pdf
imu_gyro_hist_pooltest02.png
traj_en_pooltest02.pdf
```

### 10.2 输出目录

推荐结构：

```
runs/<run_id>/plots/
  imu/
  dvl/
  pwm/
  traj/
```

---

## 11. Style Stability Rule (重要)

* 所有绘图脚本必须通过统一 style 模块初始化
* 不允许在单个脚本中私自修改 rcParams
* 若确需新增风格元素：

  * 必须先更新本规范
  * 再更新 style 模块

```

---



