# uwnav_dynamics
Control-Oriented Data-Driven Dynamics Modeling for Underwater Robots
---

# uwnav_dynamics

**Control-Oriented Data-Driven Dynamics Modeling for Underwater Robots**

---

## 1. 项目背景与目标

### 1.1 背景

在水下机器人控制中，经典的建模路径通常为：

> 控制信号 → 推进器力/力矩 → 刚体 + 水动力学 → 系统运动学

该路径存在以下工程与理论困难：

* 推进器与流体强耦合、强非线性、时变；
* 动力学参数难以准确辨识，且随工况变化；
* 多传感器（IMU / DVL）观测存在漂移、掉测与不确定性；
* 复杂物理模型不利于 MPC / RL 等在线控制方法部署。

### 1.2 项目目标

本项目旨在构建一个：

> **面向控制（control-oriented）的数据驱动水下动力学建模平台**

核心目标不是“参数辨识”，而是：

* 从**控制输入**直接预测**可观测运动响应**；
* 提供**短时多步预测**，可直接服务 MPC / RL；
* 显式输出**预测不确定度**，支持风险敏感控制；
* 模型轻量、稳定、可部署，并支持仿真与实机一致接口。

---

## 2. 系统建模抽象（Problem Formulation）

### 2.1 状态与输出定义（S1）

项目采用**最小但控制充分**的状态集合：

[
x = [v_x,\ v_y,\ \omega_z,\ a_x,\ a_y,\ \psi]
]

其中：

* (v_x, v_y)：平面线速度（DVL 校正）
* (\omega_z)：偏航角速度（IMU）
* (a_x, a_y)：平面线加速度（IMU，重力补偿后）
* (\psi)：偏航角（IMU 内部姿态解算）

统一约定：

* 坐标系：**ENU（East–North–Up）**
* 角度单位：**rad**
* yaw 范围：((-\pi,\pi])

### 2.2 控制导向预测目标

在控制频率 50 Hz 下，模型学习映射：

[
(u_{k-L:k},\ x_k)\ \longrightarrow\ x_{k+1:k+H}
]

* (L)：历史窗口（1–2 s）
* (H)：预测窗口（0.4–0.8 s）
* 输出为未来多步状态序列

---

## 3. 不确定度建模（U1）

由于 IMU 与 DVL 均存在噪声、漂移与掉测，本项目采用**异方差建模（heteroscedastic modeling）**：

* 网络同时预测状态均值 (\hat{x}) 与不确定度 (\hat{\sigma})
* 训练使用高斯负对数似然（NLL）

[
\mathcal{L}_{NLL}
=================

\frac{|y-\hat{y}|^2}{\hat{\sigma}^2}
+
\log \hat{\sigma}^2
]

意义：

* 网络自动学习“哪些数据更可信”；
* 控制器可利用 (\hat{\sigma}) 做风险管理与约束收紧。

---

## 4. 传感器角色与物理假设

### 4.1 IMU

* 提供高频：(a_x,a_y,\omega_z,\psi)
* 需进行：

  * 滤波
  * 坐标变换
  * 重力补偿
  * 零偏估计与漂移控制
* 姿态解算结果视为**观测量**，不在本项目中重复解算。
* 在所有对齐与裁剪中，优先使用 MonoNS；EstNS 仅用于与外部系统对齐或可视化参考。

### 4.2 DVL

* 提供低频但绝对可信的速度观测
* 需进行质量门控（beam 数、相关性等）
* 以**异步监督**方式参与训练（不插值造标签）

### 4.3 物理一致性原则

* 加速度与速度满足离散积分一致性：
  [
  v_{k+1} \approx v_k + a_k \Delta t
  ]
* 水下阻尼为耗散过程（避免负阻尼/能量注入）

---

## 5. 网络与计算单元设计原则

### 5.1 水下专用计算单元（设计原则）

模型不直接辨识完整 (M,C,D)，而是引入以下结构先验：

1. **推进器微动力单元**

   * deadzone / saturation
   * 一阶滞后（等效推进器惯性）

2. **Hydro-SSM 隐状态单元**

   * 可学习时间常数
   * 吸收尾流、附加质量等“流体记忆效应”
   * 保证递推稳定性

3. **阻尼一致性输出头**

   * 显式速度相关阻尼项
   * 提升多步 rollout 稳定性

4. **不确定度输出头（U1）**

---

## 6. 数据处理功能规格

### 6.1 原始数据 QA

* 采样率稳定性
* 时间戳单调性
* 丢包与异常值检测
* 单位与量纲一致性

### 6.2 IMU 预处理

* 滤波与去毛刺
* 坐标变换（传感器 → body → ENU）
* 重力补偿
* 零偏估计与随机游走建模

### 6.3 DVL 预处理

* 质量门控与 mask
* 坐标变换
* 异步监督准备

### 6.4 多频率时间线对齐

* 主时间轴：50 Hz（控制频率）
* IMU 100 Hz → 周期内统计/特征
* DVL 10 Hz → 异步监督
* PWM、电参统一映射到 MonoNS

### 6.5 数据集构建

* 滑动窗口化（L / H）
* 标准化与归一化
* 可复现的 train/val/test 划分

---

## 7. 损失函数与训练目标

训练目标由以下部分组合（可配置）：

1. **多步预测误差**
2. **异方差 NLL**
3. **速度–加速度一致性约束**
4. **耗散/平滑正则项**
5. **异步 DVL 监督项**

---

## 8. 评估指标（Metrics）

### 8.1 回归性能

* RMSE / MAE（一步 & 多步）
* 误差随预测步长增长曲线

### 8.2 不确定度校准

* NLL
* 预测区间覆盖率（PICP）

### 8.3 控制相关指标

* 闭环轨迹误差
* 控制代价下降
* 约束触发率（超速、超加速度）

---

## 9. SCI 风格可视化规范

项目需支持以下四类图像输出：

1. **原始数据质量图**
2. **预处理效果图**
3. **预测与不确定度图（含 rollout）**
4. **控制/仿真验证图**

统一风格：

* 物理量清晰标注
* 误差带/置信区间
* 可直接用于论文

---

## 10. 项目目录职责总览

---
虚拟环境激活：conda activate uwnav_train_cpu
# Project Structure

```text
uwnav_dynamics/
├── README.md
├── LICENSE
├── pyproject.toml                # Python 项目元信息（或 setup.cfg）
├── requirements.txt              # 运行依赖
├── requirements-dev.txt          # 开发/可视化/测试依赖
├── .gitignore
├── .editorconfig
│
├── configs/                      # 所有“可变因素”统一放在配置中
│   ├── dataset/                  # 数据集构建规则
│   │   ├── base.yaml              # 通用设置（频率、窗口长度等）
│   │   └── pooltest.yaml          # 某次实验的数据选择与裁剪
│   │
│   ├── preprocess/               # 预处理参数（与算法解耦）
│   │   ├── imu.yaml               # IMU 滤波、坐标、重力、bias
│   │   ├── dvl.yaml               # DVL 质量门控、坐标变换
│   │   ├── power.yaml             # 电压电流（训练期可选）
│   │   └── alignment.yaml         # 多频率时间线对齐规则
│   │
│   ├── model/                    # 网络结构与组件组合
│   │   ├── s1_u1_hyrossm.yaml     # S1 + U1 主模型
│   │   └── ablations/             # 消融实验配置
│   │       ├── no_damping.yaml
│   │       ├── no_uncertainty.yaml
│   │       └── no_hydro_memory.yaml
│   │
│   ├── loss/                     # 损失函数组合
│   │   ├── base.yaml              # NLL + consistency
│   │   └── robust.yaml            # 加强鲁棒性版本
│   │
│   ├── train/                    # 训练策略
│   │   ├── local_cpu_smoke.yaml   # 本地 CPU 冒烟测试
│   │   └── server_gpu.yaml        # 服务器 GPU 正式训练
│   │
│   ├── eval/                     # 评估与指标
│   │   └── base.yaml
│   │
│   └── plot/                     # SCI 风格绘图参数
│       └── sci.yaml
│
├── data/                         # 数据（默认不提交到 git）
│   ├── raw/                      # 原始日志（不可修改）
│   │   └── 2026-01-10_pooltest02/
│   │       ├── imu/
│   │       ├── dvl/
│   │       ├── pwm/
│   │       └── power/
│   │
│   ├── interim/                  # 中间产物（可删除重建）
│   │   ├── aligned/
│   │   └── qa_reports/
│   │
│   ├── processed/                # 网络可直接使用的数据集
│   │   └── pooltest02_s1/
│   │       ├── features.npz
│   │       ├── labels.npz
│   │       └── meta.yaml
│   │
│   └── splits/                   # 训练/验证/测试划分
│       └── split_v1.yaml
│
├── src/
│   └── uwnav_dynamics/
│       ├── __init__.py
│       │
│       ├── core/                 # 全项目共享的数学与语义基础
│       │   ├── timebase.py        # 时间轴抽象（50Hz 主轴）
│       │   ├── frames.py          # 坐标系、旋转、ENU 约定
│       │   ├── units.py           # 单位换算（g↔m/s² 等）
│       │   └── typing.py          # S1 状态、数据结构定义
│       │
│       ├── io/                   # 数据 I/O（无算法）
│       │   ├── readers/           # 原始数据读取
│       │   │   ├── imu_reader.py
│       │   │   ├── dvl_reader.py
│       │   │   ├── pwm_reader.py
│       │   │   └── power_reader.py
│       │   ├── writers/           # 统一写盘接口
│       │   │   └── writer.py
│       │   └── dataset_index.py   # 数据集索引与元信息
│       │
│       ├── preprocess/           # 原始数据 → 物理一致数据
│       │   ├── imu/
│       │   │   ├── filter.py      # 滤波与去毛刺
│       │   │   ├── bias.py        # 零偏估计与随机游走
│       │   │   ├── gravity.py     # 重力补偿
│       │   │   ├── transform.py   # 坐标变换
│       │   │   └── pipeline.py
│       │   │
│       │   ├── dvl/
│       │   │   ├── quality.py     # 质量门控
│       │   │   ├── transform.py
│       │   │   └── pipeline.py
│       │   │
│       │   ├── power/
│       │   │   └── pipeline.py    # 仅训练增强使用
│       │   │
│       │   ├── align/
│       │   │   ├── aligner.py     # 多频率对齐
│       │   │   └── imu_50hz.py    # 100Hz→50Hz 聚合
│       │   │
│       │   └── build_dataset.py   # raw → processed 统一入口
│       │
│       ├── models/               # 网络与损失
│       │   ├── blocks/            # 水下专用计算单元
│       │   │   ├── thruster_lag.py
│       │   │   ├── hydro_ssm_cell.py
│       │   │   ├── damping_head.py
│       │   │   └── uncertainty_head.py
│       │   │
│       │   ├── nets/
│       │   │   ├── s1_predictor.py
│       │   │   └── teacher_predictor.py
│       │   │
│       │   ├── losses/
│       │   │   ├── nll.py
│       │   │   ├── consistency.py
│       │   │   ├── robust.py
│       │   │   └── loss_builder.py
│       │   │
│       │   ├── metrics/
│       │   │   ├── regression.py
│       │   │   ├── calibration.py
│       │   │   └── control_kpi.py
│       │   │
│       │   └── utils/
│       │       └── rollout.py
│       │
│       ├── train/                # 训练流程
│       │   ├── trainer.py
│       │   ├── optim.py
│       │   ├── callbacks.py
│       │   └── logging.py
│       │
│       ├── eval/                 # 离线评估
│       │   ├── evaluate.py
│       │   ├── ablation.py
│       │   └── reports.py
│       │
│       ├── viz/                  # SCI 风格绘图
│       │   ├── sci_style.py
│       │   ├── plot_raw.py
│       │   ├── plot_proc.py
│       │   ├── plot_fit.py
│       │   └── plot_control.py
                eval/
                     plot_horizon_metrics.py     # 画 RMSE/MAE vs horizon（Acc/Gyro/Vel 三组）
                     plot_rollout_samples.py     # 画 y_hat vs y_true（若干样例）
│       │
│       └── cli/                  # 命令行入口
│           ├── preprocess.py
│           ├── train.py
│           ├── eval.py
│           └── plot.py
│
├── scripts/                      # 运行脚本（非算法）
│   ├── local_smoke.sh             # 本地快速跑通
│   ├── server_train.sh            # 服务器 GPU 训练
│   ├── sync_data.sh               # 数据同步
│   └── pack_run.sh                # 打包实验结果
│
├── runs/                         # 所有实验输出（不进 git）
│   └── 2026-01-23_s1u1_h40/
│       ├── config_resolved.yaml
│       ├── checkpoints/
│       ├── logs/
│       ├── metrics/
│       ├── plots/
│       └── report.md
│
├── docs/                         # 文档与论文材料
│   ├── design/
│   │   ├── platform_spec.md
│   │   ├── dataset_spec.md
│   │   └── model_spec.md
│   └── figures/
│
└── tests/                        # 冒烟与稳定性测试
    ├── test_timebase.py
    ├── test_frames.py
    ├── test_preprocess_smoke.py
    └── test_rollout_stability.py
```

---

## 目录设计原则总结（README 可附）

* **configs 决定行为，src 不写死参数**
* **data 分级：raw → interim → processed**
* **preprocess 与 train 完全解耦**
* **models 中只放“物理 + 学习”的核心逻辑**
* **runs 是实验最小复现单元**
* **viz 与 eval 独立，服务 SCI 与控制验证**

---

---

## 11. 运行模式与计划

* **阶段 1**：Ubuntu 本地 CPU 跑通完整流水线
* **阶段 2**：服务器 GPU 加速训练与消融实验
* **阶段 3**：接入 MPC / RL 仿真
* **阶段 4**：实机部署与验证

---

## 12. 项目定位总结（一句话）

> **这是一个“为控制而生”的水下机器人数据驱动动力学建模平台，而不是单纯的系统辨识或预测模型。**
---

## License (开源许可)

This project is licensed under the **GNU Affero General Public License v3.0** (**AGPL-3.0-or-later**).

- ✅ You may use, modify, and redistribute this project.
- ✅ If you distribute modified versions, you must provide the corresponding source code under the same license.
- ✅ If you run a modified version as a network service (e.g., SaaS / online inference / hosted training), you must also provide the corresponding source code to users of that service.

If you want to use this project in a **closed-source** or **commercial** product/service without disclosing your modifications, please contact the author to discuss a **commercial license**.

---

## Citation (学术引用)

If you use this repository in academic research, please cite it.

- A ready-to-use citation metadata file is provided: **`CITATION.cff`** (GitHub will render it automatically).
- When a paper/preprint becomes available, we will add DOI/arXiv information here.

---

## Data & Reproducibility (数据与复现)

This repository is **code-only**. Large datasets, logs, and generated artifacts are not tracked in git.

- `data/`, `out/`, `runs/` are treated as **generated / local-only** directories.
- Please follow the dataset specs under `configs/` to build datasets reproducibly.
- If you release a processed dataset snapshot later, we recommend publishing it via a separate data repository or an artifact host (e.g., releases / Zenodo / HuggingFace Datasets) and linking it here.

---

## Disclaimer

This software is provided “as is”, without warranty of any kind. Use it at your own risk, especially in real-world underwater experiments.

---

