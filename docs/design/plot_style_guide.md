# 绘图风格规范

## 1. 文档目标

本文档用于统一 `uwnav_dynamics` 的科研绘图输出风格，使图像同时适用于：

- 论文正文与附录
- 组会或答辩 PPT
- 工程调试与实验审查

本规范追求的不是“视觉装饰”，而是：

- 信息最小冗余
- 学术性与可读性兼顾
- 风格稳定、可复现、可批量生成

## 2. 适用范围

当前覆盖的图类包括：

- 传感器诊断图（IMU / DVL / Power）
- 单模型评估图（horizon / rollout / pred-vs-observed）
- 单模型分量审查图（component trace / residual）
- 多模型比较图（优先支持 horizon compare）

未来若新增控制、轨迹或不确定性图，也应继承同一视觉语言。

## 3. 风格真源

代码层的风格真源必须唯一：

- 全局绘图 token、preset、语义配色、视觉层级：`src/uwnav_dynamics/viz/style/sci_style.py`
- 多行传感器图的布局与刻度策略：`src/uwnav_dynamics/viz/style/imu_style.py`

其余 `viz/*` 模块只允许消费这些 style token，不应重新定义独立的全局风格规则。

## 4. Scientific Figure Design Principles

### 4.1 Information Design

- 同一语义只表达一次。
- 默认不使用图标题，优先使用坐标轴标签与图例表达语义。
- 共享 x 轴时，只在底部保留 x 轴标签。
- 单变量子图默认不使用 legend。
- 多子图中尽量只保留一次 legend。
- 缺测或缺字段提示只保留必要的轴内文本，不叠加标题、legend 与说明文字。

### 4.2 Visual Design

- 使用克制、低饱和、稳定的配色，不依赖 Matplotlib 默认 color cycle 作为最终视觉语言。
- 线型层次比颜色更重要，保证灰度打印、投影和压缩后仍可区分。
- 不同图型使用不同尺寸 preset，禁止所有图强行使用同一宽高比。
- 留白、边距、子图间距应固定，不依赖随意 `tight_layout`。
- 默认图应在脱离正文后，仍可通过坐标轴标签、单位与图例被基本理解。

### 4.3 Visual Hierarchy

- 主结果必须一眼可见。
- `primary / proposed` 方法使用最高视觉优先级。
- `baseline` 方法默认弱化，但必须可辨认。
- `ablation` 方法处于中间层级，便于说明部件作用。
- 这些规则应固化为 style token，而不是依赖每次调用手工指定全部样式。

### 4.4 Uncertainty-ready

- 若模型输出 `logvar` 或其他不确定度信息，绘图系统应预留 `mean ± sigma band` 的扩展接口。
- 当前最小实现可以不默认出不确定度图，但代码结构与文档命名应允许无破坏接入。

## 5. 字体与排版

统一要求：

- 整个仓库只使用一套 serif 主字体家族
- label / tick / legend / annotation 有固定字号层级
- 默认无标题，禁止用长标题承载正文解释
- 单位必须进入坐标轴标签

建议默认：

- 正文字体：serif
- y 轴标签：物理量 + 单位
- 图例：简洁且不重复

## 6. Preset 体系

至少保留两类 preset：

- `paper`
  - 默认 preset
  - 用于论文正文、附录与正式报告
  - 更紧凑的画布与字号
- `ppt`
  - 与 `paper` 共用同一视觉语言
  - 只放大字号、线宽、marker 和画布
  - 不引入另一套独立配色体系

在 `paper` preset 下，至少提供以下图型尺寸：

- `single`：单图或单栏图
- `sensor_3row`：IMU / rollout / pred-vs-observed 的 3 行共享 x 轴图
- `sensor_2row`：DVL raw 两行图
- `sensor_4x2`：Power 4×2 面板
- `compare_3row`：多模型 horizon compare
- `component_3x3`：9 分量逐轴对比或残差图

## 7. 坐标轴规则

统一要求：

- 所有坐标轴标签必须包含物理量与单位
- 时间轴命名统一为 `Time (s)` 或 `Prediction horizon (s)`
- 共享 x 轴时，上层子图不再重复设置 xlabel
- 默认关闭网格；若某类图确实需要轻量 grid，必须由 style token 显式打开

多行图的行语义默认由 y 轴标签承担，例如：

- `Acc (g)`
- `Gyro (deg/s)`
- `v_body (m/s)`
- `Depth (m)`

## 8. 图例规则

图例应满足：

- 只解释必要语义
- 尽量只出现一次
- 单变量子图默认无 legend
- 不把正文说明塞进图例

常见语义映射建议：

- `X / Y / Z`：使用稳定颜色映射
- `Acc / Gyro / Vel`：使用稳定颜色映射
- `Observed target / Prediction`：优先用线型区分，颜色只做辅助
- `Primary / Baseline / Ablation`：由 style token 自动赋予视觉层级

## 9. 颜色与线型语义

推荐使用稳定语义映射，而不是每次脚本自动随机选择：

- `X / Y / Z`：固定颜色
- `Acc / Gyro / Vel`：固定颜色
- `Observed target / Prediction`：实线 vs 虚线
- `Primary / Baseline / Ablation`：固定颜色 + 线型 + 线宽 + z-order

默认规则：

- 主结果更深、更粗、更靠上层
- baseline 更浅、更细、更靠下层
- ablation 处于中间层级

## 10. 图类型设计要点

### 10.1 传感器图

- 默认无标题
- 行语义使用 y 轴标签
- 共享 x 轴，仅底部子图显示 `Time (s)`
- 单变量子图不使用 legend

### 10.2 Horizon 图

- 单模型图：一张图中展示 `Acc / Gyro / Vel`
- 多模型对比图：优先使用 3×1 compare 布局，每行一个 group，多模型同轴比较
- 不使用标题，通过 y 轴标签直接表明 `RMSE` 或 `MAE`

### 10.3 Rollout 图

- 优先使用 3×1 共享 x 轴布局
- 行语义使用 `||Acc||`、`||Gyro||`、`||Vel||`
- `Observed target` 与 `Prediction` 使用线型主导区分

### 10.4 Pred-vs-Observed 图

- 第一版默认读取 `pred_samples.npz`
- 其中 `observed` 明确指评估阶段的监督目标 `y_true`
- 它不等同于未经处理的原始传感器输出
- 当前正式支持：
  - `group_norm` mode：3×1 总览图
  - `component` mode：3×3 分量逐轴图

### 10.5 Component Residual 图

- 读取 `pred_samples.npz` 与可选 `pred_context.npz`
- 每个输出分量单独绘制 `prediction - target`
- 若存在 `target_mask`，masked-out 位置应以轻量标记显示
- 风格上沿用 `X/Y/Z` 稳定配色，不额外引入新调色板

### 10.6 Model Compare 图

- 第一版优先支持 horizon compare
- `primary / proposed` 必须视觉主导
- `baseline` 必须弱化但仍可辨认
- `global summary` 仅作为补充，不应抢占主图结论

## 11. Export and Naming Policy

当前正式评估链路默认只保留图片文件：

- `png`：快速浏览、实验审查、结果归档

若未来确需论文排版专用导出，再单独扩展，不应影响当前主产物命名。

命名必须稳定，优先使用以下约定：

- horizon 单模型：
  - `rmse_horizon_groups.png`
  - `mae_horizon_groups.png`
- horizon 多模型比较：
  - `rmse_horizon_compare.png`
  - `mae_horizon_compare.png`
- rollout 样例：
  - `rollout_sample_000.png`
- pred-vs-observed：
  - `pred_vs_observed_group_norm_000.png`
  - `pred_vs_observed_component_000.png`
- component residual：
  - `residual_component_000.png`
- 多模型 compare 主图：
  - `rmse_model_compare_horizon.png`
  - `mae_model_compare_horizon.png`

新增图型不得随意改变已有核心 artifact 名称；若确有必要，必须先更新本规范。

## 12. 与评估 artifact 的关系

当前评估主流程稳定产物定义见 `docs/evaluation_protocol.md`：

- `metrics.yaml`
- `rmse_by_horizon.csv`
- `mae_by_horizon.csv`
- `pred_samples.npz`

绘图层应基于这些稳定 artifact 读盘，不应反向修改其数值语义。

特别说明：

- `pred_samples.npz` 中的 `y_true` 是评估阶段使用的监督目标
- 它可作为 `pred-vs-observed` 第一版中的 `observed`
- 但它不等于原始 IMU / DVL / Volt 传感器原始输出

## 13. 与代码的关系

当前绘图实现主要位于：

- `src/uwnav_dynamics/viz/style/`
- `src/uwnav_dynamics/viz/eval/`
- `src/uwnav_dynamics/viz/plots/`

当代码与本文档不一致时，应优先同步二者，而不是长期容忍风格漂移。
