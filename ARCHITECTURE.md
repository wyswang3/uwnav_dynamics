# 项目架构说明（uwnav_dynamics）

## 1. 项目目标
本项目面向水下航行器动力学辨识（system identification）：
- 通过多源传感器历史窗口学习短时动力学映射；
- 以控制输入（PWM）和观测状态（IMU/DVL 等）预测未来状态序列；
- 服务于后续控制、仿真、评估与可解释分析。

当前主实现是基于序列模型的监督学习流水线：
`align -> build_dataset -> train -> eval -> viz`。

## 2. 四源数据与采样频率
项目默认处理 4 类异步数据源：
- PWM（约 100 Hz）：推进器控制输入（`ch1_cmd ... ch8_cmd`）
- IMU（约 100 Hz）：机体加速度/角速度/姿态
- DVL（约 10 Hz）：体速度与深度相关观测，天然稀疏
- Power（约 5 Hz）：电机电压/电流推导的功率辅助量

核心配置入口见：
- `configs/dataset/pooltest02.yaml`
- `configs/align/pooltest02.yaml`
- `configs/dataset/pooltest02_s1.yaml`

## 3. 多频异步融合策略（主轴 100Hz + 稀疏监督）
当前工程策略：以 100 Hz 主时间轴作为训练基准（`dt_main_s = 0.01`）。

对齐阶段（`src/uwnav_dynamics/preprocess/align/aligner.py`）主要做：
- IMU：聚合到主轴（高频主观测）
- PWM：hold-last 映射到主轴（控制输入）
- DVL：按最近邻 attach 到主轴，仅在命中位置写入，其他时刻保留 NaN，并写 `has_dvl` 掩码
- Power：低频 hold-last 到主轴，作为辅助特征

训练侧的语义是“主轴密集输入 + DVL 稀疏监督”。
当前数据构建里还会生成状态速度列（如 `VelX_state_mps`）用于窗口建模。

## 4. 现有管线

### 4.1 对齐（align）
- 入口：`src/uwnav_dynamics/preprocess/align/cli_align.py`
- 产物：`out/train/*_train_base.csv`

### 4.2 数据集构建（build_dataset）
- 入口：`src/uwnav_dynamics/preprocess/build_dataset.py`
- 产物：
  - `features.npz`
  - `labels.npz`
  - `meta.yaml`

### 4.3 训练（train）
- CLI 包装：`src/uwnav_dynamics/cli/train.py`
- 训练主入口：`src/uwnav_dynamics/train/run_train.py`
- 训练器：`src/uwnav_dynamics/train/trainer.py`
- 典型产物：`best.pth`、`last.pth`

### 4.4 评估（eval）
- CLI 包装：`src/uwnav_dynamics/cli/eval.py`
- 评估主入口：`src/uwnav_dynamics/eval/evaluate.py`
- 典型产物：
  - `metrics.yaml`
  - `rmse_by_horizon.csv`
  - `mae_by_horizon.csv`
  - `pred_samples.npz`

### 4.5 可视化（viz）
- 评估可视化：
  - `src/uwnav_dynamics/viz/eval/plot_horizon_metrics.py`
  - `src/uwnav_dynamics/viz/eval/plot_rollout_samples.py`
- 传感器与预处理可视化：`src/uwnav_dynamics/viz/plots/*`

## 5. 当前模型模块（s1_predictor + blocks）
主模型：`src/uwnav_dynamics/models/nets/s1_predictor.py`

当前结构：
- Backbone：LSTM encoder
- Head：输出未来增量 `dY` 与 `logvar`
- Rollout：`y_hat = y0 + cumsum(dY)`（delta-cumsum）

可选 blocks（按配置开关）：
- `ThrusterLag`：推进器输入侧非理想建模（deadzone/saturation/lag）
- `HydroSSMCell`：流体记忆隐状态
- `DampingHead`：速度相关阻尼先验
- `UncertaintyHead`：异方差不确定度头

相关文件：
- `src/uwnav_dynamics/models/blocks/thruster_lag.py`
- `src/uwnav_dynamics/models/blocks/hydro_ssm_cell.py`
- `src/uwnav_dynamics/models/blocks/damping_head.py`
- `src/uwnav_dynamics/models/blocks/uncertainty_head.py`

## 6. 后续升级路线

### 6.1 Mask-aware 训练与评估（优先）
- 将 `has_dvl`/有效性掩码显式并入 loss 与 metric 计算；
- 对稀疏目标维（尤其 DVL 相关速度）执行逐元素 mask 监督，减少伪标签污染。

### 6.2 模型家族扩展（可选）
- 在保持当前 S1 管线不破坏的前提下，引入可切换骨干：
  - Transformer 序列建模分支（更强长依赖建模）；
  - KF/ESKF 融合分支（物理约束 + 学习残差的混合路线）。

### 6.3 工程化与实验管理
- 统一配置层（train/model/data）与实验命名；
- 完善评估基线、可重复命令链与结果落盘规范。
