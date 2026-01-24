# IMU Plot Style Specification

本规范定义 uwnav_dynamics 项目中 IMU 相关图像的统一绘图风格。
该风格源自已有 offline_nav 项目的成熟实现，并被视为“稳定视觉语言”，
后续新增或重构绘图代码应以本规范为准。

---

## 1. Design Goal

IMU 图像用于展示：
- 原始或最小预处理后的 IMU 观测
- 传感器噪声水平、偏置、漂移与异常
- 姿态与运动状态的整体一致性

图像目标不是精确读数，而是 **快速、可靠地传达物理行为**。

---

## 2. Canonical Layout

### 2.1 Overall Structure

IMU 图采用固定的三行布局（纵向排列）：

1. Acceleration（X / Y / Z）
2. Angular rate（X / Y / Z）
3. Attitude angles（X / Y / Z）

所有子图：
- 共享同一时间轴
- 垂直对齐
- 语义顺序严格固定（不可打乱）

---

## 3. Canvas and Layout Policy

### 3.1 Anchored Canvas

- 画布尺寸固定（非自适应）
- margins 与子图间距固定
- 不依赖 tight_layout 或 constrained_layout

该策略用于确保：
- 不同实验、不同脚本生成的图具有一致比例
- 图像可直接用于论文排版

### 3.2 Typography Scaling

- 所有字体、线宽、刻度尺寸随画布等比例缩放
- 以参考尺寸（ref_in）为基准
- 禁止在子图中单独指定字体大小

---

## 4. Axis and Tick Policy

### 4.1 Time Axis (X)

- 所有子图共享 x 轴
- 仅最底部子图显示 x tick label
- x tick 数量受控（避免拥挤）

时间轴被视为“参考坐标”，不应主导视觉注意力。

### 4.2 Y Axis (Critical)

- 每个子图目标显示约 3 个主刻度
- 优先使用“nice ticks”（整洁、可读的数值）
- 若自动定位失败，退化为 25% / 50% / 75% 轴位置

该策略强调趋势而非精度，提升整体可读性。

---

## 5. Color and Semantic Mapping

- X axis: C0
- Y axis: C1
- Z axis: C2

该映射在所有 IMU 图中保持不变。

颜色用于表达语义，而非区分实验或方法。

---

## 6. Titles and Legends

### 6.1 Titles

- 每个子图有 title
- title 只包含：
  - 物理量名称
  - 单位
- title 中不得重复 X/Y/Z 信息

示例：
- "Acceleration (g)"
- "Angular rate (deg/s)"
- "Attitude (deg)"

### 6.2 Legends

- legend 用于说明 X / Y / Z 语义
- 半透明背景
- 无边框
- 放置在不遮挡主要数据的位置

---

## 7. Export Policy

- 默认导出 PNG（工程浏览）
- 分辨率高于屏幕显示需求（>=300 dpi）
- 输出路径与文件名应包含：
  - run_id 或 dataset_id
  - 图类型标识（imu_raw_9axis 等）

---

## 8. Stability Rule (Important)

- 本规范描述的是“稳定风格”，而非示例
- 若需要新增 IMU 图类型：
  - 应复用相同布局、刻度与语义映射
- 若确需修改风格：
  - 先更新本规范
  - 再修改代码实现
