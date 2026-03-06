# IMU Plot Style Specification

本规范定义 `uwnav_dynamics` 项目中 IMU 相关图像的统一绘图风格。
它是 `docs/design/plot_style_guide.md` 在 IMU 图型上的专项落地版本。

## 1. Design Goal

IMU 图像用于展示：

- 原始或最小预处理后的 IMU 观测
- 噪声、偏置、漂移和异常
- 姿态与运动状态的整体一致性

图像目标不是精确读数，而是快速、可靠地传达物理行为与异常模式。

## 2. Canonical Layout

IMU 图采用固定三行布局：

1. Acceleration
2. Angular rate
3. Attitude angles

所有子图：

- 共享同一时间轴
- 垂直对齐
- 行顺序严格固定

## 3. Canvas and Layout Policy

- 画布尺寸使用 `sci_style.py` 中的 `sensor_3row` preset
- margins 与子图间距固定
- 不依赖 `tight_layout` 或 `constrained_layout`
- 所有字体、线宽、tick 尺寸随 preset 成体系变化

## 4. Axis and Tick Policy

### 4.1 Time Axis

- 所有子图共享 x 轴
- 仅最底部子图显示 x 轴标签 `Time (s)`
- 上层子图不重复设置 xlabel

### 4.2 Y Axis

- 每个子图目标显示约 3 个主刻度
- 优先使用整洁、可读的 “nice ticks”
- 行语义使用 y 轴标签承担，而不是标题

推荐 y 轴标签：

- `Acc (g)` 或 `Acc (m/s^2)`
- `Gyro (deg/s)` 或 `Gyro (rad/s)`
- `Att (deg)`

## 5. Color and Semantic Mapping

- X axis: 固定颜色
- Y axis: 固定颜色
- Z axis: 固定颜色

该映射在所有 IMU 图中保持不变。

## 6. Titles and Legends

### 6.1 Titles

- 默认不使用标题
- 图中物理量名称与单位由 y 轴标签承担

### 6.2 Legends

- legend 只用于说明 X / Y / Z 语义
- 整张图尽量只保留一次 legend
- 使用无边框、克制的最小 legend 样式

## 7. Export Policy

- 默认支持 PNG
- 正式输出应支持 PDF
- 输出路径与文件名应包含 run_id 或 dataset_id 以及图类型标识

## 8. Stability Rule

- 本规范描述的是稳定风格，而非单次实现示例
- 若新增 IMU 图类型，应复用相同布局、刻度和语义映射
- 若确需修改风格，应先更新文档，再更新代码
