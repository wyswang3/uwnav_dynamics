# 网络方案简称规范

更新时间：2026-04-10

## 1. 目的

当前仓库同时保留了：

- 历史原始数据主线
- 当前 `KF / ESKF` 主线
- `step-transition` 主线

历史文档与配置中存在大量 `B0 / B4 / U1 / B4+U1 / STEP_B4` 这类代号。
这些代号对研发阶段内部沟通有用，但在：

- 训练矩阵图
- replay 模型筛选图
- 论文图注
- 技术汇报

里不够直观，也容易把“结构模块组合代号”误当成“正式模型名”。

因此当前阶段统一采用：

- **对外展示名使用简短英文简称**
- **artifact 路径与历史 `variant` 字符串保留 legacy 代号**

也就是：

- 图表和说明里尽量写新简称
- 复现与审计路径里继续保留旧 `variant`

## 2. 简称表

| 新简称 | legacy 代号 / 历史标签 | 典型结构开关 | 所属主线 | 说明 |
| --- | --- | --- | --- | --- |
| `Base` | `B0` | 无额外结构块 | 历史原始主线 | 纯 `S1Predictor` 基线 |
| `BaseWide` | `B0 hidden512` | 更大 hidden size | 历史原始主线 | 基线宽度扩展分支 |
| `BaseDeep` | `B0 512x3` | 更深 RNN | 历史原始主线 | 基线深度扩展分支 |
| `Thr` | `B1` | `ThrusterLag` | 历史原始主线 | 推进器动态增强分支 |
| `Hydro` | `B2` | `HydroSSM` | 历史原始主线 | 流体记忆增强分支 |
| `Dyn` | `B4` | `ThrusterLag + HydroSSM` | 历史原始主线 | 动力学先验增强版 |
| `Unc` | `U1` | `UncertaintyHead` | 历史原始主线 | 专门化异方差不确定度分支 |
| `DynUnc` | `B4+U1` / `B4U1` | `ThrusterLag + HydroSSM + UncertaintyHead` | 历史原始主线 | 动力学先验与不确定度联合增强 |
| `StepNLL` | `STEP_A0` | `joint head + nll_diag` | 当前 step 主线 | 一步转移基线，偏对照用途 |
| `StepBase` | `STEP_B0` | `grouped head + transition_balance` | 当前 step 主线 | 一步转移主线基线 |
| `StepDelta` | `STEP_B2 strong-delta` | `StepBase + stronger delta loss` | 当前 step 主线 | 强化增量约束的消融分支 |
| `StepDyn` | `STEP_B4` | `StepBase + ThrusterLag + HydroSSM` | 当前 step 主线 | 当前更接近状态转移求解器的主候选 |
| `KFNLL` | `KF joint NLL` | KF dense + `joint head` | KF 主线 | KF 版一步 NLL 对照 |
| `KFBase` | `KF grouped TB` | KF dense + `grouped head` | KF 主线 | KF 版 grouped baseline |
| `KFDvlAux` | `KF grouped TB DVLaux` | `KFBase + dvl_obs aux` | KF 主线 | 带 DVL 辅助头的 KF 分支 |
| `KFDyn` | `KF grouped TB B4` | `KFBase + ThrusterLag + HydroSSM` | KF 主线 | KF 动力学增强版 |
| `KFCtxNLL` | `KF Ctx joint NLL` | KF 上下文 + `joint head` | KF 上下文主线 | KF 上下文 NLL 对照 |
| `KFCtxBase` | `KFCTX_B0` / `QV3_B0` | KF 上下文，多步 dense 监督 | KF 多步主线 | KF 上下文基线 |
| `KFCtxTail` | `KF Ctx grouped TB longtail` | `KFCtxBase + long-tail reweight` | KF 上下文主线 | 长尾强化分支 |
| `KFCtxDyn` | `KFCTX_B4` / `QV3_B4` / `KF_B4` | `KFCtxBase + ThrusterLag + HydroSSM` | KF 多步主线 | KF 多步动力学增强版 |
| `QualBase` | `QV3 grouped TB` | quality ctx + grouped TB | quality 主线 | 质量上下文基线 |
| `QualTail` | `QV3 grouped TB longtail` | `QualBase + long-tail reweight` | quality 主线 | 长尾强化分支 |
| `QualDyn` | `QV3 grouped TB blocks` | `QualBase + ThrusterLag + HydroSSM` | quality 主线 | 质量上下文动力学增强版 |
| `QualDynV2` | `V2 grouped TB blocks` | 上一版 quality dyn | quality 主线 | 历史对照保留名 |

## 3. 当前使用规则

### 3.1 图表与表格

训练对比图、replay compare 图、论文结果表优先使用：

- `Base`
- `Dyn`
- `Unc`
- `DynUnc`
- `StepBase`
- `StepDyn`

示例：

- `DynUnc s8`
- `StepDyn s9`
- `Legacy-Unc s8`

### 3.2 配置与路径

以下对象继续保留 legacy 字段，不主动重命名：

- `run.variant`
- 已生成的 `configs/train/generated/.../*.yaml`
- 已存在的 `out/ckpts/...`
- 已存在的 `out/train_matrix/...`

原因：

- 它们已经进入复现实验链路
- 直接改名会破坏产物追溯与论文审计证据

## 4. 当前推荐写法

### 4.1 历史原始主线

- 不再写 `B4+U1`
- 改写为 `DynUnc`

### 4.2 当前 step 主线

- 不再写 `STEP_B4`
- 改写为 `StepDyn`

### 4.3 新旧主线对比

推荐写法：

- `Legacy-Unc`
- `Legacy-DynUnc`
- `StepBase`
- `StepDyn`

不推荐写法：

- `Old U1`
- `Old B4U1`
- `Current Step B4`

## 5. 备注

- 本规范只改**对外展示名**，不强行改历史 artifact 标识。
- 若后续新增结构分支，应优先按“功能语义”命名，而不是继续沿用 `Bx/Ux` 递增代号。
