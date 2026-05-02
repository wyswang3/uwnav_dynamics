> 历史说明：本文档记录旧 `B0 / B4 / U1 / B4+U1` 模块组合表，
> 用于解释早期论文材料中的结构代号，不代表当前最终状态转移求解器。
> 当前最终方案是 `StepBase s11 / step_b0_grouped_tb_seed11`，详见
> [current_transition_solver_selection.md](/home/wys/uwnav_dynamics/docs/design/current_transition_solver_selection.md)。

下面这张表可直接用于论文或实验说明。命名依据主要来自 configs/launch/
  pooltest02_s1_8gpu_compare.yaml 和 configs/launch/
  pooltest02_s1_round5_finalconfirm_e120.yaml，共享基础配置见 configs/train/
  pooltest02_s1_lstm_v0.yaml。

  | 结构名 | LSTM主干 | ThrusterLag | HydroSSM | DampingHead | UncertaintyHead |
  DVL辅助头 | 含义概括 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | B0 | 开 | 关 | 关 | 关 | 关 | 关 | 纯基线 LSTM 多步增量预测器 |
  | B1 | 开 | 开 | 关 | 关 | 关 | 关 | 在基线上加入推进器输入动态校正 |
  | B2 | 开 | 关 | 开 | 关 | 关 | 关 | 在基线上加入流体记忆隐状态 |
  | B4 | 开 | 开 | 开 | 关 | 关 | 关 | 输入侧先验 + 状态侧先验联合增强 |
  | U1 | 开 | 关 | 关 | 关 | 开 | 关 | 基线 + 专门化异方差不确定度分支 |
  | B4+U1 | 开 | 开 | 开 | 关 | 开 | 关 | B4 结构增强 + U1 不确定度分支 |

  补充说明：

  - B4+U1 在旧阶段就是“thruster_lag + hydro_ssm + uncertainty 三个模块同时开启”的组
    合。
  - 这些变体里 DampingHead 没有被启用，见 configs/train/
    pooltest02_s1_lstm_v0.yaml:58。
  - DVL 辅助观测头虽然代码已实现，但当前维护的这些 launch 配置里也没有启用。
  - 共享基础行为还有两点：
      - use_thruster_as_replacement: true，即开启 ThrusterLag 时用 u_eff 替换原
        始控制输入。
      - use_hydro_feat: true，即开启 HydroSSM 时把 h_last 拼接到主 head 特征中。
        见 configs/train/pooltest02_s1_lstm_v0.yaml:32。

  如果你需要，我可以继续把这张表改成“论文写作版”，换成更正式的表题、列名和注释。
