# 第 4.3 节网络训练框架专项审查

## 0. 审查范围与证据基础

本文档仅审查当前仓库中**实际存在**的“用于多源传感器动力学辨识的网络训练框架”，不尝试寻找或对比不存在的“第一阶段网络训练代码”，也不构造“第一阶段 vs 第二阶段”的伪代码级分析。

本次审查的直接目标，是为毕业论文第 4.3 节“基于多源传感器与改进网络结构的辨识实验”提供可直接落笔的方法学骨架，并为后续绘制网络结构图、模块图、数据流图提供文字依据。

本次审查的核心证据文件如下：

- `src/uwnav_dynamics/models/nets/s1_predictor.py:103-403`
- `src/uwnav_dynamics/models/blocks/thruster_lag.py:26-195`
- `src/uwnav_dynamics/models/blocks/hydro_ssm_cell.py:26-144`
- `src/uwnav_dynamics/models/blocks/damping_head.py:26-125`
- `src/uwnav_dynamics/models/blocks/uncertainty_head.py:25-83`
- `src/uwnav_dynamics/train/run_train.py:132-332`
- `src/uwnav_dynamics/train/config.py:371-632`
- `src/uwnav_dynamics/train/trainer.py:44-349`
- `src/uwnav_dynamics/train/data_pipeline.py:129-600`
- `src/uwnav_dynamics/preprocess/build_dataset.py:159-340`
- `src/uwnav_dynamics/models/utils/semantic_output_layout.py:42-187`
- `src/uwnav_dynamics/supervision_mask.py:50-149`
- `configs/train/pooltest02_s1_lstm_v0.yaml:1-100`
- `configs/dataset/pooltest02_s1.yaml:1-92`
- `configs/launch/pooltest02_s1_8gpu_compare.yaml:1-120`
- `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:1-150`

交叉验证命令与结果：

```bash
pytest -q \
  tests/test_train_config_p0.py \
  tests/test_s1_predictor_aux_heads.py \
  tests/test_composite_loss_p0.py \
  tests/test_execution_layout_contract.py \
  tests/test_supervision_mask_contract.py \
  tests/test_train_matrix_launcher.py \
  tests/test_eval_config_parity.py
```

结果：`24 passed in 10.83s`

说明：

- `configs/model/s1_u1_hyrossm.yaml` 当前为空文件，且未被当前训练入口或实验矩阵引用，因此**不应作为本次论文审查的证据来源**。
- 若下文出现“代码实现描述”和“论文建议表述”两种说法，前者对应仓库中的精确实现，后者对应更适合写入论文但仍不偏离代码事实的学术化表述。

---

## 1. 当前网络训练框架的总体定位

### 1.1 总体目标

当前训练框架的总体目标，不是学习一个抽象的端到端黑箱映射，而是基于多源传感器历史窗口与控制输入历史窗口，构建一个**面向短时未来观测状态轨迹的多步动力学辨识预测器**。其训练入口 `run_train.py` 明确采用“模型输出未来状态增量 `dY`，再通过 `y0 + cumsum(dY)` 重构未来状态序列”的方式组织训练与评估语义（`src/uwnav_dynamics/train/run_train.py:132-193`，`src/uwnav_dynamics/models/utils/rollout.py:1-30`）。

更准确地说，该框架学习的是：

\[
\mathbf{X}_{t-L+1:t} \rightarrow \Delta \hat{\mathbf{Y}}_{t+1:t+H}
\rightarrow \hat{\mathbf{Y}}_{t+1:t+H}
\]

其中，输入不是单一传感器，而是 PWM、IMU、速度状态代理量和功率辅助量的组合；输出也不是单一标量，而是未来多步的 9 维观测状态代理量轨迹。

### 1.2 它服务于什么辨识任务

从当前数据配置与模型执行路径看，这一框架服务的是：

- **控制输入驱动的短时运动响应辨识**：利用推进器控制历史和多源观测历史，预测未来短时观测状态序列。
- **多源传感器条件下的动力学辨识**：输入侧融合了控制、惯性、速度状态和功率信息；监督侧同时包含高频 IMU 量和低频 DVL 约束。
- **稀疏速度观测下的状态预测辨识**：速度分量虽然进入输出张量，但其有效监督由 `dvl_mask -> target_mask` 决定，仅在 DVL 有效处参与损失（`src/uwnav_dynamics/train/data_pipeline.py:176-194`，`src/uwnav_dynamics/supervision_mask.py:75-149`）。

因此，该框架更接近“短时控制导向观测状态辨识器”，而不是完整意义上的全状态物理参数辨识器。

### 1.3 相比一般黑箱网络的结构特点

与常见“历史窗直接进 LSTM/MLP，输出未来状态”的黑箱建模相比，当前框架至少有五个显著特点：

1. 它显式区分了输入中的**控制子序列** `u_seq` 与**状态子序列** `y_seq`，而不是完全依赖网络自行隐式分离（`src/uwnav_dynamics/models/nets/s1_predictor.py:300-306`）。
2. 它采用**多步状态增量预测 + rollout 重构**，而不是直接回归未来绝对状态（`src/uwnav_dynamics/models/nets/s1_predictor.py:366-373`，`src/uwnav_dynamics/train/run_train.py:160-167`）。
3. 它允许把不同类型的结构先验分别插入**输入侧、状态侧、输出侧和不确定度侧**，形成“主干网络 + 模块化增强”的组合结构（`src/uwnav_dynamics/models/nets/s1_predictor.py:206-255`）。
4. 它显式处理了**DVL 稀疏监督**问题：速度分量不是简单 dense loss，而是由运行时 `target_mask` 控制（`src/uwnav_dynamics/train/trainer.py:105-145`，`src/uwnav_dynamics/models/losses/nll.py:28-53`）。
5. 它把**执行布局**与**语义布局**分离：`y_in_idx` 只负责 rollout 初值提取，`acc/gyro/vel` 语义分组由 `semantic_output_layout` 统一管理。这一设计比“硬编码列号 + 硬编码语义”的实现更严谨（`src/uwnav_dynamics/models/utils/execution_layout.py:34-66`，`src/uwnav_dynamics/models/utils/semantic_output_layout.py:42-107`）。

### 1.4 更适合被归类为哪一种模型框架

**代码实现描述：**

当前实现最准确的技术归类应为：

- 一个以 `LSTM` 为主干的**多步状态增量预测器**
- 叠加若干可插拔先验模块的**模块化先验增强网络**
- 面向控制与短时预测的**控制导向观测状态预测器**

**论文建议表述：**

建议在论文中优先表述为：

> 一种“面向多源传感器动力学辨识的模块化先验增强多步状态增量预测网络”。

不建议直接写成：

- “完整神经状态空间模型”
- “纯黑箱序列到序列网络”
- “端到端全状态估计器”

原因是当前代码并没有在预测 horizon 内显式递推一个统一的潜在物理状态转移方程；它更像是“历史编码后一次性解码未来多步增量，再通过累加得到未来状态”的结构。

### 1.5 研究意义与工程意义

**研究意义：**

- 它把多源传感器条件下的高频 IMU、低频 DVL 与控制输入统一到同一辨识框架中，能够支持论文讨论“多源信息融合对动力学辨识的贡献”。
- 它把结构先验从单一“网络加深/加宽”转化为“输入非理想性补偿 + 流体记忆建模 + 不确定性建模”等可解释增强，适合形成论文中的“改进网络结构”叙事主线。
- 它用稀疏监督掩码处理 DVL 低频约束，使“多源传感器”不只是输入多源，也体现在监督机制多源。

**工程意义：**

- 训练与评估共享同一套 `split/scaler/layout` 契约，复现性较强（`src/uwnav_dynamics/train/run_train.py:276-297`，`src/uwnav_dynamics/eval/evaluate.py:880-914`）。
- 原始时间轴上的 purged split 避免滑窗边界泄漏，适合科研审查（`src/uwnav_dynamics/train/data_pipeline.py:432-499`，`src/uwnav_dynamics/dataset/split.py:116-208`）。
- 模块化 blocks 便于做消融实验、实验矩阵管理和论文图示提炼（`configs/launch/pooltest02_s1_8gpu_compare.yaml:33-120`）。

---

## 2. 主体网络结构解析

### 2.1 主体网络名称、位置与核心类

| 项目 | 内容 |
| --- | --- |
| 主体网络名称 | `S1Predictor` |
| 代码文件 | `src/uwnav_dynamics/models/nets/s1_predictor.py` |
| 核心类名 | `S1Predictor` |
| 配置类名 | `S1PredictorConfig` |
| 训练入口构造位置 | `src/uwnav_dynamics/train/run_train.py:299-308` |

### 2.2 输入维度、输出维度、时间窗口与预测长度

当前主线配置由 `configs/train/pooltest02_s1_lstm_v0.yaml` 与 `configs/dataset/pooltest02_s1.yaml` 共同确定：

| 项目 | 当前实现 |
| --- | --- |
| 输入维度 `din` | 25 |
| 输出维度 `dout` | 9 |
| 历史窗口 `hist_len` | 100 步 |
| 预测长度 `pred_len` | 10 步 |
| 主时间基准 | `dt = 0.01 s` |
| 历史时长 | 1.0 s |
| 预测时长 | 0.1 s |

证据：

- `configs/train/pooltest02_s1_lstm_v0.yaml:18-25`
- `configs/dataset/pooltest02_s1.yaml:30-34`

### 2.3 主干 backbone 类型

主体 backbone 是一个 `batch_first=True`、单向的 `nn.LSTM` 编码器，默认配置为：

- hidden size = 256
- layers = 2
- dropout = 0.0（仅 `rnn_layers > 1` 时生效）

证据：`src/uwnav_dynamics/models/nets/s1_predictor.py:197-204`，`configs/train/pooltest02_s1_lstm_v0.yaml:22-25`

### 2.4 forward 主流程

`forward_with_aux()` 的主流程可以概括为：

1. 从 `X` 中按索引抽取控制子序列 `u_seq` 与状态子序列 `y_seq`（`src/uwnav_dynamics/models/nets/s1_predictor.py:340-342`）。
2. 将 `u_seq` 送入 `ThrusterLag`，得到等效控制 `u_eff`（同文件:344-345）。
3. 若 `thruster_lag.enabled=true` 且 `use_thruster_as_replacement=true`，则用 `u_eff` 替换原始输入中的控制通道，再送入 LSTM（同文件:347-351）。
4. 将 `u_eff` 与 `y_seq` 输入 `HydroSSMCell`，得到流体记忆隐状态 `h_last`（同文件:353-355）。
5. 用 LSTM 对处理后的整段历史窗口编码，取最后一层最后时刻隐状态 `h_rnn`（同文件:356-358）。
6. 若 `use_hydro_feat=true`，将 `h_rnn` 与 `h_last` 拼接成主 head 特征 `h_feat`；否则仅使用 `h_rnn`（同文件:360-364）。
7. 通过主 head 一次性输出 `H * Dout * 2` 维向量，并重排为未来 `H` 步的 `dY` 与 `logvar_base`（同文件:366-373）。
8. 将 `DampingHead(y_last)` 生成的速度阻尼增量加到 `dY` 上（同文件:375-378）。
9. 若 `uncertainty.enabled=true`，则用专门的不确定度分支替换 `logvar_base`；否则沿用主 head 联合输出的 `logvar_base`（同文件:380-387）。
10. 若 `dvl_obs_head` 存在，则额外输出未来速度语义组的辅助观测预测 `aux["dvl_obs"]`（同文件:389-393）。

### 2.5 主输出、辅助输出与训练语义

**主输出：**

- `dY: (B, H, 9)`，未来 10 步 9 维状态的预测增量
- `logvar: (B, H, 9)`，与主输出同维度的对角对数方差

**辅助输出：**

- `aux["dvl_obs"]: (B, H, 3)`，仅当辅助观测头开启时存在，对应速度语义组

**训练语义：**

- 训练不是“单步直接预测”，也不是“teacher forcing 的自回归解码”
- 当前实现是**一次性多步增量预测**
- 然后执行 `y_hat = y0 + cumsum(dY)` 的 rollout 重构
- 最终对 `y_hat` 与 `Y` 计算对角高斯 NLL；若存在 `target_mask`，则采用 masked NLL

证据：`src/uwnav_dynamics/train/run_train.py:153-193`

### 2.6 推理语义与训练语义是否一致

是一致的。

训练阶段与评估阶段都只接受：

- `rollout.y0_source = "x_last_state"`
- `rollout.mode = "delta_cumsum"`

评估时同样使用 `extract_y0_from_x_last()` 和 `rollout_from_delta()` 恢复未来状态轨迹（`src/uwnav_dynamics/eval/evaluate.py:906-914`）。这意味着论文中可以明确写为：

> 训练与推理采用一致的多步状态增量 rollout 语义，不存在训练期 teacher forcing、推理期 free-run 的语义切换。

### 2.7 论文中最值得突出的主体设计

建议在论文中重点突出以下几点：

1. **多步增量预测而非绝对状态直接回归**。这使网络更贴近短时动力学变化建模。
2. **主干 LSTM 不单独工作，而是被多个先验模块围绕增强**。这使“改进网络结构”具备明确模块边界。
3. **控制量与状态量在输入中被显式切片使用**。这一点增强了结构可解释性。
4. **稀疏 DVL 监督由 runtime mask 处理，而不是简单删除样本**。这一点体现多源传感器条件下的监督设计。
5. **训练与评估共享同一 rollout 解释**。这一点有助于论文学术表达中的严谨性。

---

## 3. 各增强模块逐项解析

### 3.1 ThrusterLag

- 模块名称：`ThrusterLag`
- 所在文件和类名：`src/uwnav_dynamics/models/blocks/thruster_lag.py`，类 `ThrusterLag`
- 模块输入输出：`u_seq:(B,L,8) -> u_eff_seq:(B,L,8)`
- 插入位置：主网络输入侧，在 LSTM 编码前执行；可选择用 `u_eff` 替换原始控制输入（`src/uwnav_dynamics/models/nets/s1_predictor.py:344-351`）
- 数学作用或物理含义：对原始 PWM 指令执行归一化、死区、饱和和一阶滞后递推，近似推进器执行链的静态非线性与动态滞后（`src/uwnav_dynamics/models/blocks/thruster_lag.py:31-46`，`95-195`）
- 设计动机：减少主干 LSTM 用隐状态去“歪学”推进器非理想输入特性
- 可能贡献：提高控制输入到运动响应映射的可解释性，并改善在高频输入场景下的建模稳定性
- 建议论文命名与描述：建议写为“推进器滞后与非线性校正模块”或“输入侧推进器动态先验模块”
- 是否可插拔、可开关、可消融：是。`enabled=false` 时退化为恒等映射
- 当前配置状态：在 `B1`、`B4`、`B4+U1` 结构变体中被启用（`configs/launch/pooltest02_s1_8gpu_compare.yaml:72-120`，`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:88-150`）

### 3.2 HydroSSM

- 模块名称：`HydroSSMCell`
- 所在文件和类名：`src/uwnav_dynamics/models/blocks/hydro_ssm_cell.py`，类 `HydroSSMCell`
- 模块输入输出：`(u_eff_seq, y_seq) -> (h_seq, h_last)`
- 插入位置：状态侧先验模块；`h_last` 可拼接到主 head 特征中（`src/uwnav_dynamics/models/nets/s1_predictor.py:353-364`）
- 数学作用或物理含义：实现稳定递推

\[
h_k = \lambda \odot h_{k-1} + (1-\lambda)\odot \phi(W[u_k, y_k] + b)
\]

其中 `lambda = sigmoid(param)`，保证递推稳定（`src/uwnav_dynamics/models/blocks/hydro_ssm_cell.py:31-42`，`86-143`）
- 设计动机：显式吸收尾流、附加质量和流体记忆等缓变效应，而不是把这些效应完全压给 LSTM
- 可能贡献：在短时多步预测中提供更具物理意味的“流体记忆摘要”，有利于提升 rollout 稳定性
- 建议论文命名与描述：建议写为“流体记忆隐状态模块（Hydro-SSM）”
- 是否可插拔、可开关、可消融：是。关闭时输出全零 `h_last`
- 当前配置状态：在 `B2`、`B4`、`B4+U1` 变体中启用（`configs/launch/pooltest02_s1_8gpu_compare.yaml:84-120`，`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:88-150`）

补充说明：

- 当前实现并未把 `h_seq` 作为未来序列解码器输入逐步递推，而是只取 `h_last` 作为摘要特征。因此论文中更适合称其为“流体记忆摘要模块”，而不宜夸大为“完整流体状态转移器”。

### 3.3 DampingHead

- 模块名称：`DampingHead`
- 所在文件和类名：`src/uwnav_dynamics/models/blocks/damping_head.py`，类 `DampingHead`
- 模块输入输出：`y_last:(B,9) -> dY_damp:(B,H,9)`
- 插入位置：输出侧，对主 head 的 `dY` 做加性修正（`src/uwnav_dynamics/models/nets/s1_predictor.py:375-378`）
- 数学作用或物理含义：仅对速度语义组构造阻尼增量

\[
\Delta v_{\text{damp}} = - d(v)\odot v
\]

并将该阻尼项扩展到整个未来 horizon（`src/uwnav_dynamics/models/blocks/damping_head.py:31-40`，`101-125`）
- 设计动机：把速度相关耗散从“完全隐式学习”改为“显式可学习修正”
- 可能贡献：提升速度分量的物理一致性，并改善长一点的 rollout 稳定性
- 建议论文命名与描述：建议写为“阻尼一致性输出头”或“速度耗散先验分支”
- 是否可插拔、可开关、可消融：是
- 当前配置状态：**代码已实现，但当前维护的 launch 配置未见启用**

需注意的代码事实：

- 当前实现把由 `y_last` 生成的阻尼增量在全部 horizon 上重复展开（`src/uwnav_dynamics/models/blocks/damping_head.py:122-124`），因此它不是一个“逐预测步演化的阻尼递推器”，而是一个“由末时刻状态诱导的 horizon-wise 常值阻尼偏置”。论文中应按此保守表述。

### 3.4 UncertaintyHead

- 模块名称：`UncertaintyHead`
- 所在文件和类名：`src/uwnav_dynamics/models/blocks/uncertainty_head.py`，类 `UncertaintyHead`
- 模块输入输出：`feat:(B,feat_dim) -> logvar:(B,H,9)`
- 插入位置：不确定度侧。当开启时，它**替换**主 head 联合输出的 `logvar_base`（`src/uwnav_dynamics/models/nets/s1_predictor.py:380-387`）
- 数学作用或物理含义：为未来每一步每个输出分量估计对角异方差
- 设计动机：将“状态增量预测”和“不确定度估计”从同一线性头中部分解耦
- 可能贡献：提高异方差建模的表达能力，并为后续置信区间分析或控制筛查提供辅助信息
- 建议论文命名与描述：建议写为“解耦异方差不确定度分支（U1）”
- 是否可插拔、可开关、可消融：是
- 当前配置状态：`U1` 与 `B4+U1` 主线变体均启用（`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:39-150`）

非常重要的表述建议：

- **代码实现描述：**即使 `uncertainty.enabled=false`，主 head 仍然输出 `logvar_base`，因此基础模型本身已经支持 `nll_diag` 训练。
- **论文建议表述：**`U1` 更准确地说是“专门化不确定度分支”或“解耦不确定度建模”，而不是“首次引入不确定性建模”。

### 3.5 DVL 辅助观测头

- 模块名称：`dvl_obs_head`
- 所在文件和类名：定义在 `src/uwnav_dynamics/models/nets/s1_predictor.py:242-249`，通过 `forward_with_aux()` 返回 `aux["dvl_obs"]`
- 模块输入输出：`h_feat:(B,head_in) -> dvl_obs:(B,H,3)`
- 插入位置：训练期辅助分支，与主 head 共用高层特征（`src/uwnav_dynamics/models/nets/s1_predictor.py:389-393`）
- 数学作用或物理含义：仅预测速度语义组，用于额外的 DVL 观测辅助监督
- 设计动机：在不改变主输出协议的前提下，加强速度观测语义的训练约束
- 可能贡献：在稀疏 DVL 条件下，为共享表示提供更直接的速度观测学习信号
- 建议论文命名与描述：建议写为“训练期 DVL 辅助观测分支”
- 是否可插拔、可开关、可消融：是
- 当前配置状态：**代码已实现，但当前维护的 train/launch YAML 未见启用**

需特别强调：

- 该分支只通过 `forward_with_aux()` 暴露，默认 `forward()` 并不输出它（`src/uwnav_dynamics/models/nets/s1_predictor.py:397-403`）。
- 因此它是一个**训练期辅助监督分支**，不是最终部署接口中的主输出分支。
- 若论文第 4.3 节要把该模块写成“已完成的主实验模块”，需要人工确认是否存在相应训练产物；仅从当前维护配置看，尚不能这样写死。

---

## 4. 损失函数、训练机制与辅助监督

### 4.1 主损失项

当前主损失项是对 rollout 后未来状态序列计算的对角高斯负对数似然：

\[
\mathcal{L}_{\text{state}}
= \text{NLL}_{\text{diag}}\left(\hat{\mathbf{Y}}_{t+1:t+H}, \mathbf{Y}_{t+1:t+H}, \log \hat{\sigma}^2\right)
\]

其实现路径为：

`model -> dY/logvar -> extract y0 from x_last -> delta_cumsum rollout -> gaussian_nll_diag(_masked)`

证据：`src/uwnav_dynamics/train/run_train.py:153-167`

### 4.2 是否存在辅助损失项

存在，但为可选项。

当同时满足以下条件时，会额外加入 DVL 辅助观测损失：

- `model.aux_heads.dvl_obs.enabled = true`
- `loss.dvl_obs_weight > 0`
- batch 中存在 `target_mask`

辅助损失采用 masked Huber loss，仅作用于速度语义组（`src/uwnav_dynamics/train/run_train.py:169-193`，`src/uwnav_dynamics/models/losses/auxiliary.py:1-44`）。

### 4.3 是否存在不确定性建模

存在。

- 基础模型主 head 已经联合输出 `logvar_base`
- `U1` 变体在此基础上用专门的 `UncertaintyHead` 替换 `logvar_base`

因此，论文中更准确的写法应为：

> 当前框架在基线层面已经采用对角异方差 NLL 训练；`U1` 进一步将不确定度预测从联合主 head 中解耦为专门分支。

### 4.4 是否存在辅助观测头或多任务训练

存在代码级实现，但不是当前维护配置中的主线启用项。

- 辅助头：`dvl_obs_head`
- 多任务形态：主状态路径 + DVL 辅助观测路径
- 当前维护配置状态：未见主线 YAML 启用

因此：

- 若论文写“本节主实验采用多任务训练”，需要人工确认实际实验配置；
- 若仅依据当前维护的 train/launch YAML，主线实验更稳妥的说法仍应是**单主任务训练 + 可选辅助监督机制已实现**。

### 4.5 优化器、学习率调度与训练超参数

**代码支持的训练配置空间：**

- 优化器：仅支持 `AdamW`（`src/uwnav_dynamics/train/config.py:575-577`）
- 学习率调度：`none` 或 `reduce_on_plateau`（同文件:578-583）
- 梯度裁剪：支持 `grad_clip`
- AMP：支持，且仅在 CUDA 下启用（`src/uwnav_dynamics/train/trainer.py:218-230`）
- early stopping：支持 `patience + min_delta`（`src/uwnav_dynamics/train/trainer.py:236-239, 322-329`）

**基础 train YAML 的默认值：**

- batch size = 256
- epochs = 3
- scheduler = `reduce_on_plateau`
- lr = `1e-3`
- weight decay = `1e-4`
- grad clip = `1.0`

证据：`configs/train/pooltest02_s1_lstm_v0.yaml:9-16`，`81-100`

**正式实验矩阵的主线覆盖值：**

- device = `cuda`
- amp = `true`
- batch size = 512
- epochs = 120

证据：`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:26-37`

这一点在论文中必须写清：

- `configs/train/pooltest02_s1_lstm_v0.yaml` 更像是“基础单次实验定义 + 本地轻量默认值”
- 实际用于结构比较与正式对照的训练预算，由 `configs/launch/*.yaml` 在矩阵层统一覆盖

### 4.6 是否存在 rollout、residual、delta prediction、teacher forcing

| 机制 | 当前实现情况 | 论文建议写法 |
| --- | --- | --- |
| rollout | 存在，且训练/评估一致 | 多步 rollout 一致训练评估 |
| delta prediction | 存在 | 多步状态增量预测 |
| residual around physics model | 不存在完整 physics model 残差学习，但存在可选加性 prior 分支 | 建议写“结构先验增强”，不建议写“纯残差物理网络” |
| teacher forcing | 不存在 | 应明确说明未采用 teacher forcing |
| 自回归逐步解码 | 不存在 | 属于一次性 horizon 解码 |

---

## 5. 数据输入输出语义与辨识任务定义

### 5.1 当前模型输入由哪些传感器/控制量组成

当前输入特征维度为 25，构成为：

| 类别 | 维度 | 字段 |
| --- | --- | --- |
| 推进器控制 | 8 | `ch1_cmd ... ch8_cmd` |
| IMU 线加速度 | 3 | `AccX/Y/Z_body_mps2` |
| IMU 角速度 | 3 | `GyroX/Y/Z_body_rad_s` |
| 速度状态代理量 | 3 | `VelX/Y/Z_state_mps` |
| 功率辅助量 | 8 | `P0_W ... P7_W` |

证据：`configs/dataset/pooltest02_s1.yaml:36-68`，`configs/train/pooltest02_s1_lstm_v0.yaml:20-21`

### 5.2 每类输入在辨识中的作用

- **PWM 控制量**：提供外部激励，是动力学辨识中的直接控制输入。
- **IMU 加速度与角速度**：提供高频惯性响应，是短时动态变化的主体观测。
- **速度状态代理量**：作为当前可观测运动状态的一部分参与历史编码。
- **功率辅助量**：反映执行机构负载/功耗信息，为输入侧提供额外工作状态线索。

需要特别注意：

- 当前模型**并不直接输入** `has_dvl`、`dvl_mask_hist` 或 `power_mask_hist`。
- 这些 mask artifact 会被存盘，但不会被拼接进 `X`（`src/uwnav_dynamics/preprocess/build_dataset.py:299-314`，`src/uwnav_dynamics/train/data_pipeline.py:143-194`）。

因此，论文中不宜写成“模型显式输入 DVL 有效性掩码”；更准确的说法是：

> 模型输入包含数值型多源特征，而 DVL 有效性通过训练/评估阶段的监督掩码机制参与。

### 5.3 当前模型预测的目标量是什么

当前模型预测的目标张量 `Y` 维度为 9，对应：

- `AccX/Y/Z_body_mps2`
- `GyroX/Y/Z_body_rad_s`
- `VelX/Y/Z_state_mps`

语义布局由 `semantic_output_layout` 固化为：

- `acc = [0,1,2]`
- `gyro = [3,4,5]`
- `vel = [6,7,8]`

证据：`src/uwnav_dynamics/models/utils/semantic_output_layout.py:46-85`

### 5.4 输出目标为什么这样设计

当前输出设计本质上是“未来观测状态代理量轨迹”，而不是“未来原始传感器全量输出”或“完整动力学隐状态”。

其中速度分量的处理尤其需要精确表述：

- **代码实现描述：**
  `VelX/Y/Z_state_mps` 在数据集构建阶段由 DVL 体速度列前向填充得到；若序列开头尚无 DVL，则填 0.0（`src/uwnav_dynamics/preprocess/build_dataset.py:159-214`）。
- **监督实现描述：**
  训练与评估时，速度分量是否参与损失并不由 `Y` 是否有数值直接决定，而是由 `dvl_mask -> target_mask` 决定；只有 DVL 有效位置参与速度监督（`src/uwnav_dynamics/train/data_pipeline.py:176-194`，`src/uwnav_dynamics/supervision_mask.py:75-149`）。
- **论文建议表述：**
  建议写为“网络预测未来 9 维观测状态代理量，其中速度分量采用 DVL 可用性约束下的稀疏监督”，而不建议写成“网络对未来速度执行稠密真值监督”。

### 5.5 这套任务定义如何体现“多源传感器与改进网络结构”

“多源传感器”体现在两层：

1. **输入层多源**：PWM + IMU + 速度状态代理量 + 功率
2. **监督层多源**：IMU 分量为高频密集监督，速度分量为 DVL 掩码约束下的稀疏监督

“改进网络结构”体现在两层：

1. **主干结构层**：从纯 LSTM 扩展为“LSTM + 输入侧/状态侧/输出侧/不确定度侧增强模块”
2. **训练语义层**：从简单直接回归扩展为“多步增量 rollout + mask-aware NLL + 可选辅助监督”

### 5.6 在论文中应如何表述该辨识问题

建议在论文中把辨识问题定义为：

> 在 100 Hz 主时间轴下，给定最近 1 s 的推进器控制、惯性观测、速度状态代理量及功率辅助量，学习其到未来 0.1 s 内 9 维观测状态代理量轨迹的映射关系；其中速度分量的监督仅在 DVL 有效观测处参与损失，以适应多频多源传感器条件下的稀疏速度约束。

补充校正：

- `configs/dataset/pooltest02_s1.yaml` 中“下一个时刻的观测状态”这一注释并不完全准确，因为实际 `pred_len=10`，`Y` 是未来 10 步序列而非单步（`configs/dataset/pooltest02_s1.yaml:30-34`, `70-80`）。论文正文应统一改写为“未来 10 步观测状态序列”。

---

## 6. 主线版本与可选变体

### 6.1 主线版本判断

从当前维护的配置体系看，应区分三个层次：

1. **基础单次训练定义**：`configs/train/pooltest02_s1_lstm_v0.yaml`
2. **早期结构扫描矩阵**：`configs/launch/pooltest02_s1_8gpu_compare.yaml`
3. **收敛后的正式主线矩阵**：`configs/launch/pooltest02_s1_round4_controlconfirm_e120.yaml` 与 `configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml`

若以“当前维护主线”而不是“最初扫描全体”为准，则可以得到：

- `B0`：基线
- `U1`：强对照/消融
- `B4+U1`：当前主线 primary 候选结构

证据：

- 早期矩阵中比较 `B0 / B1 / B2 / U1 / B4`（`configs/launch/pooltest02_s1_8gpu_compare.yaml:33-120`）
- round5 中 `U1` 被标注为 `ablation`，`B4+U1` 被标注为 `primary`（`configs/launch/pooltest02_s1_round5_finalconfirm_e120.yaml:39-150`）

### 6.2 主线版本与可选版本的区别

| 版本 | 结构特征 | 当前角色 |
| --- | --- | --- |
| B0 | 纯 LSTM 主干，无 blocks | 基线 |
| B1 | 仅 `ThrusterLag` | 早期消融 |
| B2 | 仅 `HydroSSM` | 早期消融 |
| B4 | `ThrusterLag + HydroSSM` | 早期 primary / 中间候选 |
| U1 | 专门化不确定度分支 | 当前强对照/消融 |
| B4+U1 | `ThrusterLag + HydroSSM + UncertaintyHead` | 当前主线 primary 候选 |

### 6.3 哪些模块不应直接写成主实验事实

以下能力虽然存在代码实现，但当前维护配置中未见主线启用：

- `DampingHead`
- `dvl_obs` 辅助观测头及其对应辅助损失

因此，论文中若要将其写成“本节主实验已采用的机制”，需要人工确认对应 run artifact；否则更合适的写法是：

> 当前框架已实现可插拔扩展接口，但本节主实验聚焦于 `ThrusterLag`、`HydroSSM` 与 `UncertaintyHead` 三类已进入维护矩阵的结构增强。

### 6.4 需人工确认的点

- `U1` 与 `B4+U1` 中谁是最终论文主模型，需要结合实际训练输出目录中的 `summary.csv / metrics.yaml` 人工确认。当前代码与 launch 注释只能确认二者是最终一轮主线比较对象，不能单凭配置文件断言最终胜出者。

---

## 7. 面向论文写作的提炼结果

### 7.1 第 4.3 节推荐的小节标题结构

建议采用如下结构：

1. **4.3.1 多源传感器辨识任务与输入输出定义**
2. **4.3.2 主体多步状态增量预测网络**
3. **4.3.3 结构先验增强模块设计**
4. **4.3.4 损失函数与掩码监督机制**
5. **4.3.5 实验配置与结构变体说明**

如果希望更强调“改进网络结构”，也可以写成：

1. **4.3.1 多源传感器数据表征与辨识问题定义**
2. **4.3.2 基线 LSTM 多步预测器**
3. **4.3.3 推进器动态与流体记忆增强模块**
4. **4.3.4 不确定度分支与稀疏速度监督机制**
5. **4.3.5 主线结构与消融变体设置**

### 7.2 当前网络训练框架的创新点摘要

建议概括为 5 点：

1. 提出面向多源传感器辨识的**多步状态增量预测范式**，通过 `delta-cumsum rollout` 实现未来状态序列重构。
2. 在统一主干网络上引入**输入侧推进器动态校正模块**，显式建模控制执行非理想性。
3. 引入**流体记忆隐状态模块**，在历史编码阶段补充 hydrodynamic memory 先验。
4. 采用**解耦异方差不确定度分支**，增强多步预测中不确定性表达能力。
5. 设计**DVL 稀疏监督掩码机制**，在多频多源条件下保持速度约束而避免无效监督污染。

### 7.3 适合用于绘制网络结构示意图的模块层级说明

建议按如下层级绘图：

- 输入层：`PWM(8) + IMU(6) + Vel_state(3) + Power(8)`
- 输入侧增强层：`ThrusterLag`
- 状态侧增强层：`HydroSSM`
- 主干编码层：`LSTM Encoder`
- 主预测头：`MLP Head -> dY + logvar_base`
- 输出侧增强层：`DampingHead`
- 不确定度分支：`UncertaintyHead`
- 训练期辅助分支：`DVL Aux Head`
- Rollout 层：`y0 + cumsum(dY)`
- 损失层：`Masked Gaussian NLL (+ optional masked Huber aux loss)`

### 7.4 文字版网络示意图草图说明

可按如下方式绘制：

1. 左侧放置输入块 `X_{t-L+1:t}`，内部标注四类输入通道：PWM、IMU、速度状态代理量、功率。
2. 从输入块向右分出两条细箭头，分别标注为 `u_seq slicing` 与 `y_seq slicing`。
3. `u_seq` 先进入 `ThrusterLag` 模块，输出 `u_eff`。
4. `u_eff` 一路回填到主输入通道，形成 `x_enc` 后送入 `LSTM Encoder`；另一路与 `y_seq` 一起进入 `HydroSSM`。
5. `HydroSSM` 输出 `h_last`，与 `LSTM` 输出的 `h_rnn` 在特征拼接节点融合，形成 `h_feat`。
6. `h_feat` 进入主 `MLP Head`，输出 `dY` 与 `logvar_base`。
7. 从 `y_last` 再引出一条支路进入 `DampingHead`，输出 `dY_damp`，与主干 `dY` 相加，得到修正后的 `dY`。
8. 从 `h_last + y_last + u_last` 引出不确定度分支进入 `UncertaintyHead`，其输出在开启时替换 `logvar_base`。
9. 若需要画训练期辅助分支，则从 `h_feat` 再引出一条支路进入 `DVL Aux Head`，输出未来速度观测预测。
10. 主分支的 `dY` 与从输入最后一步提取的 `y0` 一起进入 `Rollout` 模块，输出未来状态序列 `\hat{Y}`。
11. `\hat{Y}` 与真实 `Y` 及 `target_mask` 进入损失层，计算 masked NLL；若启用辅助头，则 `dvl_obs` 与速度分量监督再进入 masked Huber loss。

### 7.5 适合写入论文正文的方法总述段落草稿

可直接参考如下表述：

> 为适应水下机器人多源传感器条件下的短时动力学辨识需求，本文采用一种模块化先验增强的多步状态增量预测网络。该网络以最近 1 s 的推进器控制、IMU 观测、速度状态代理量及功率辅助量为输入，利用 LSTM 对历史序列进行编码，并一次性输出未来 0.1 s 内 9 维观测状态代理量的增量序列。随后，通过对增量序列执行累积求和，并结合输入末时刻状态构造 rollout，可恢复未来状态轨迹。为增强模型对推进器非理想性和流体记忆效应的表达能力，网络在主干 LSTM 外进一步引入推进器滞后校正模块和流体记忆隐状态模块；同时，通过解耦的不确定度分支对未来多步预测的异方差进行建模。针对 DVL 速度观测低频、稀疏的特点，本文在损失函数中引入基于观测可用性的掩码监督机制，使速度分量仅在 DVL 有效时刻参与误差回传，从而在保持多源信息利用的同时避免无效监督对训练过程的污染。

### 7.6 适合写入论文图注或图说明的简短描述

可写为：

> 图 4-x 展示了本文采用的多源传感器动力学辨识网络结构。网络以 PWM、IMU、速度状态代理量和功率辅助量为历史输入，以 LSTM 为主干，并叠加推进器滞后校正、流体记忆隐状态和不确定度建模等可插拔模块，最终通过多步状态增量 rollout 预测未来观测状态轨迹；其中速度分量采用 DVL 掩码约束下的稀疏监督。

### 7.7 适合在第 4.4 节进行人工层面对比的内容

建议在第 4.4 节进行如下层面的人工对比，而不是构造不存在的“旧代码阶段对比”：

- `B0` 基线 vs `U1`：说明专门化不确定度分支的作用
- `B0` 基线 vs `B4`：说明推进器动态先验与流体记忆先验的作用
- `U1` vs `B4+U1`：说明“仅不确定度增强”与“结构先验 + 不确定度联合增强”的差异

若论文确实没有对应 `DampingHead` 或 `dvl_obs` 的实验结果，则不建议在第 4.4 节把它们写成已完成消融；更合适的定位是“当前框架已预留的可扩展实现”。

---

## 8. 一句话结论

当前项目中的网络训练框架，最准确的论文定位并不是“一个简单的黑箱 LSTM”，也不是“与不存在旧阶段代码相对应的新阶段系统”，而是一个以 `S1Predictor` 为核心、采用**多步状态增量 rollout 语义**、并通过 `ThrusterLag / HydroSSM / UncertaintyHead` 等模块实现结构增强的**模块化先验增强短时动力学辨识网络**；其中，DVL 的多源特征主要体现在**稀疏速度监督机制**而非显式输入掩码之中。
