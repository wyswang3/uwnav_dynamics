# 仓库关键文件索引

生成时间：2026-03-06

说明：以下索引按“入口脚本 / 模型 / 配置 / 解析逻辑 / rollout-evaluate-plot-viz”分组；每条包含 `路径 + 一句话职责 + 可能输入输出`。

## 1) 训练入口脚本（train / cli / scripts）

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/cli/train.py` | 训练 CLI 包装器，解析命令行并转发到 `uwnav_dynamics.train.run_train`。 | 输入：`--yaml` 与可选覆盖参数（device/epochs/batch_size/data_dir）；输出：触发训练进程、终端打印预期 run 目录。 |
| `src/uwnav_dynamics/train/run_train.py` | 训练主入口：加载 YAML 配置、构建数据加载器/模型/损失并调用 `fit`。 | 输入：train YAML、`features.npz/labels.npz`；输出：`best.pth`、`last.pth`（位于 `run.out_dir/variant`），训练日志。 |
| `src/uwnav_dynamics/cli/pipeline.py` | 一键流水线入口，串联“训练 -> 选 ckpt -> 评估（可选画图）”。 | 输入：train YAML 与训练/评估参数；输出：训练 ckpt + 评估产物目录（含 metrics/plots）。 |

## 2) 评估入口脚本

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/cli/eval.py` | 正式评估 CLI 入口，自动选择 checkpoint，并按需编排“数值评估 -> viz 出图”。 | 输入：train YAML、可选 ckpt/split/device/plot 参数；输出：评估目录与可选 `plots/*`。 |
| `src/uwnav_dynamics/eval/evaluate.py` | 数值评估主程序：加载数据与 ckpt，执行 rollout、统计指标并落盘 artifact。 | 输入：train YAML + ckpt + `features.npz/labels.npz`；输出：`metrics.yaml`（含 layout metadata）、`rmse_by_horizon.csv`、`mae_by_horizon.csv`、`pred_samples.npz`。 |

## 3) 数据预处理入口（pipeline / align / build_dataset）

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/preprocess/build_dataset.py` | 从对齐训练表构建滑窗监督数据集，并可做标准化。 | 输入：`configs/dataset/*_s1.yaml` + `out/train/*_train_base.csv`；输出：`features.npz`、`labels.npz`、`meta.yaml`。 |
| `src/uwnav_dynamics/preprocess/align/cli_align.py` | 对齐 CLI 入口，从对齐 YAML 构造 `AlignConfig` 并执行表构建。 | 输入：`configs/align/*.yaml`；输出：`out/train/*_train_base.csv`。 |
| `src/uwnav_dynamics/preprocess/align/aligner.py` | 多传感器时间轴对齐核心实现（IMU/PWM/DVL/Power -> 主时间轴训练表）。 | 输入：IMU 处理后 CSV、PWM CSV、可选 DVL/Power CSV；输出：对齐后的 `DataFrame` 或训练表 CSV。 |
| `src/uwnav_dynamics/preprocess/imu/pipeline.py` | IMU 总管线（transform->gravity->bias->filter）并支持 CSV 入口。 | 输入：原始 IMU CSV 或数组 + `ImuPreprocessConfig`；输出：`*_proc.csv`、`ImuPreprocessDiag`。 |
| `src/uwnav_dynamics/preprocess/dvl/pipeline.py` | DVL 总管线（时间列/速度列选择、单位统一、有效性与附加列导出）。 | 输入：原始 DVL CSV + `DvlPreprocessConfig`；输出：`*_proc.csv`、`DvlPreprocessDiag`。 |
| `src/uwnav_dynamics/preprocess/power/pipeline.py` | 根据 `DatasetSpec` 读取 Volt 日志并生成 8 路功率辅助数据。 | 输入：dataset spec（含 volt 路径）；输出：`out/.../aux_power/*_power8.csv`。 |
| `apps/tools/pwm_preprocess_and_plot.py` | 工具脚本：PWM 对时导出 `cmd` 对齐 CSV，并画 8 通道命令图。 | 输入：dataset YAML + PWM 原始 CSV；输出：`*_cmd_aligned.csv`、`*_cmd_8ch.png`。 |

## 4) 模型定义文件（models/）

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/models/nets/s1_predictor.py` | S1 主模型定义，融合 LSTM backbone 与可选 blocks。 | 输入：`x:(B,L,Din)`；输出：`dY:(B,H,Dout)`、`logvar:(B,H,Dout)`。 |
| `src/uwnav_dynamics/models/blocks/thruster_lag.py` | 推进器输入先验模块，建模 deadzone/saturation/一阶滞后。 | 输入：`u_seq:(B,L,8)`；输出：`u_eff:(B,L,8)`。 |
| `src/uwnav_dynamics/models/blocks/hydro_ssm_cell.py` | Hydro-SSM 稳定递推隐状态模块，建模流体记忆效应。 | 输入：`u_eff_seq` 与 `y_seq`；输出：`h_seq`、`h_last`。 |
| `src/uwnav_dynamics/models/blocks/damping_head.py` | 阻尼先验输出头，生成速度维的显式耗散增量。 | 输入：`y_last:(B,9)`；输出：`dY_damp:(B,H,9)`。 |
| `src/uwnav_dynamics/models/blocks/uncertainty_head.py` | 异方差不确定度头，输出对角 log-variance。 | 输入：`feat:(B,feat_dim)`；输出：`logvar:(B,H,9)`。 |
| `src/uwnav_dynamics/models/utils/execution_layout.py` | rollout 执行布局 helper，校验 `y_in_idx` 并从 `X` 提取 `y0`。 | 输入：`cfg_model.y_in_idx`、`X:(B,L,Din)`；输出：执行层索引 metadata 与 `y0:(B,Dout)`。 |
| `src/uwnav_dynamics/models/utils/semantic_output_layout.py` | 输出语义布局 helper，统一组件标签、`acc/gyro/vel` 分组与 legacy fallback。 | 输入：`metrics.yaml` 或 `target_cols`；输出：semantic layout metadata。 |
| `src/uwnav_dynamics/models/utils/rollout.py` | rollout 工具函数（从 `dY` 累加得到未来状态序列）。 | 输入：`y0` 与 `dY`；输出：`y_hat`。 |
| `src/uwnav_dynamics/models/losses/nll.py` | 对角高斯 NLL 损失定义。 | 输入：`y_hat/y_true/logvar`（同形状）；输出：标量 loss。 |
| `src/uwnav_dynamics/models/blocks/__init__.py` | blocks 统一导出入口。 | 输入：无；输出：模块类与配置类命名空间。 |

## 5) 配置文件（configs/）

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `configs/train/pooltest02_s1_lstm_v0.yaml` | 训练总配置（run/data/model/rollout/loss/optim/train）。 | 输入：被 `train/config.py` 读取；输出：驱动训练、评估路径与超参数。 |
| `configs/dataset/pooltest02.yaml` | 原始数据集规格（传感器文件选择、pwm_timebase、valid_window）。 | 输入：被 `DatasetSpec.load` 读取；输出：解析后的传感器路径与 reader kwargs。 |
| `configs/dataset/pooltest02_s1.yaml` | 数据集构建配置（base_table + sliding_window + output）。 | 输入：被 `build_dataset.py` 读取；输出：决定 `features/labels/meta` 生成方式。 |
| `configs/align/pooltest02.yaml` | 多传感器对齐参数与输入输出路径配置。 | 输入：被 `cli_align.py` 读取；输出：决定训练基础表对齐策略。 |
| `configs/preprocess/imu.yaml` | IMU 预处理策略文档化配置（时间列优先级、列映射、单位与 QA）。 | 输入：当前主要作规范参考；输出：为 IMU 预处理参数提供模板。 |
| `configs/model/s1_u1_hyrossm.yaml` | 模型配置占位文件。 | 输入：当前为空文件；输出：暂无（待补充）。 |
| `configs/dataset/pooltest01.yaml` | 数据集配置占位文件。 | 输入：当前为空文件；输出：暂无（待补充）。 |

## 6) 包含 Config/dataclass/yaml 解析逻辑的关键文件

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/train/config.py` | 严格 schema 的 train YAML 解析器，构建 `TrainYamlConfig` 与各 block 配置。 | 输入：train YAML；输出：`run/data/model/rollout/loss/train` dataclass 配置对象。 |
| `src/uwnav_dynamics/io/dataset_spec.py` | dataset YAML 强类型解析与数据路径解析中枢。 | 输入：`configs/dataset/*.yaml`；输出：`DatasetSpec`、各传感器绝对路径与 kwargs。 |
| `src/uwnav_dynamics/preprocess/build_dataset.py` | 解析 dataset-building YAML 并映射为 `DatasetConfig`。 | 输入：`*_s1.yaml`；输出：滑窗配置与输出目录配置。 |
| `src/uwnav_dynamics/eval/config.py` | 定义 `EvalConfig` 并复用 train YAML 的 canonical parser 结果。 | 输入：train YAML；输出：数值评估运行时配置与共享模型配置。 |
| `src/uwnav_dynamics/preprocess/align/cli_align.py` | 解析 align YAML 并构造 `AlignConfig`。 | 输入：align YAML；输出：对齐调用参数。 |
| `src/uwnav_dynamics/viz/eval/plot_horizon_metrics.py` | 读取评估输出中的 `metrics.yaml` 与 horizon CSV 后生成图。 | 输入：`metrics.yaml` + `rmse/mae_by_horizon.csv`；输出：horizon 曲线图。 |
| `src/uwnav_dynamics/cli/utils.py` | 提供通用 YAML 读取与 ckpt 解析工具。 | 输入：train YAML 或 run_dir；输出：`out_dir/variant`、ckpt 路径。 |
| `src/uwnav_dynamics/preprocess/sliding_window.py` | 定义 `SlidingWindowConfig` 并提供 dict->config 构造函数。 | 输入：DataFrame + 配置字典；输出：`X/Y/t0/idx0` 滑窗结果。 |

## 7) 包含 rollout / evaluate / plot / viz 的关键文件

| 路径 | 一句话职责 | 可能输入输出 |
|---|---|---|
| `src/uwnav_dynamics/eval/evaluate.py` | 评估与 rollout 主流程。 | 输入：数据窗口 + ckpt；输出：metrics/csv/npz 与 layout metadata。 |
| `src/uwnav_dynamics/models/utils/execution_layout.py` | rollout 执行索引 helper。 | 输入：`cfg_model.y_in_idx` 与 `X`；输出：`y0`。 |
| `src/uwnav_dynamics/models/utils/semantic_output_layout.py` | rollout 输出语义 helper。 | 输入：`metrics.yaml` 或 `target_cols`；输出：分组解释与 fallback 结果。 |
| `src/uwnav_dynamics/models/utils/rollout.py` | rollout 纯数值辅助函数集合。 | 输入：`dY` 与初值；输出：未来状态序列。 |
| `src/uwnav_dynamics/viz/eval/plot_horizon_metrics.py` | 画 RMSE/MAE 随预测步长变化曲线。 | 输入：评估目录；输出：`rmse_horizon_*.png/pdf`、`mae_horizon_*.png/pdf`。 |
| `src/uwnav_dynamics/viz/eval/plot_rollout_samples.py` | 画 rollout 样例时域对比图。 | 输入：`pred_samples.npz`；输出：`rollout_sample_*.png/pdf`。 |
| `src/uwnav_dynamics/viz/plots/imu_plot.py` | 原始/预处理 IMU 绘图模块。 | 输入：`ImuFrame` 或 `*_proc.csv`；输出：`imu_raw_9axis.png`、`imu_dt.png`、`imu_proc_3rows.png`。 |
| `src/uwnav_dynamics/viz/plots/dvl_plots.py` | DVL 原始与预处理绘图模块。 | 输入：`DvlFrame` 或 DVL processed CSV；输出：`dvl_vel_BI_BE.png`、`dvl_proc_BI_BE_BD.png`。 |
| `src/uwnav_dynamics/viz/plots/power_plots.py` | 8 电机电流绘图模块。 | 输入：`PowerFrame`；输出：`power_currents_8motors.png`。 |
| `src/uwnav_dynamics/viz/style/sci_style.py` | 全局科学绘图样式与常用绘图辅助函数。 | 输入：matplotlib `Axes/Figure`；输出：统一风格图面。 |
| `src/uwnav_dynamics/viz/style/imu_style.py` | IMU/DVL 图布局与 tick/legend 策略。 | 输入：layout 参数与坐标轴对象；输出：标准化画布与线条样式。 |
| `apps/dev/test_imu_pipeline.py` | 开发脚本：贯通 IMU 读取、统计、预处理与绘图。 | 输入：dataset YAML 与 IMU CSV；输出：`out/imu_stats/*`、`out/imu_plots/*`、`out/imu_proc/*`。 |
| `apps/dev/test_dvl_plots.py` | 开发脚本：DVL 读取、预处理与绘图联调。 | 输入：dataset YAML 与 DVL CSV；输出：`out/dvl_proc/*`、`out/dvl_plots*/*`。 |
| `apps/dev/test_power_plots.py` | 开发脚本：Power 读取、绘图并导出 8 路功率数据。 | 输入：dataset YAML 与 volt CSV；输出：`out/power_plots/*`、`out/aux_power/*_power8.csv`。 |
| `apps/tools/pwm_preprocess_and_plot.py` | 工具脚本：PWM 对齐与 8 通道命令图绘制。 | 输入：dataset YAML 与 PWM CSV；输出：`*_cmd_aligned.csv`、`*_cmd_8ch.png`。 |

## 8) 备注

- `configs/model/s1_u1_hyrossm.yaml` 与 `configs/dataset/pooltest01.yaml` 当前为空（0 字节）。
- `scripts/` 目录当前未发现可执行入口脚本（主要入口位于 `src/uwnav_dynamics/cli` 与 `apps/`）。
