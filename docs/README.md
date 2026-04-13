# 文档导航

更新时间：2026-04-10

本文档只做一件事：告诉协作者现在应该先读什么、不同文档分别负责什么。

## 1. 首先看哪些文档

第一次接手或切回当前主线时，按这个顺序：

1. [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)
2. [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)
3. [data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)
4. [ARCHITECTURE.md](/home/wys/uwnav_dynamics/ARCHITECTURE.md)

如果只是要开始执行命令，再看：

- [reference/quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)
- [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
- [design/transition_solver_phase2_replay_upgrade.md](/home/wys/uwnav_dynamics/docs/design/transition_solver_phase2_replay_upgrade.md)
- [evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)
- [design/model_variant_aliases.md](/home/wys/uwnav_dynamics/docs/design/model_variant_aliases.md)

如果目标是在 8 卡服务器上一键跑完整流程，优先看：

- [handover_kf_training_server_v2.md](/home/wys/uwnav_dynamics/docs/handover_kf_training_server_v2.md)
- [reference/quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)
- `configs/launch/pooltest02_server_full_pipeline_8gpu_v2.yaml`

如果目标是比较“过去主线 vs 当前主线”谁更适合状态转移求解器，优先看：

- [design/old_vs_current_transition_compare_v1.md](/home/wys/uwnav_dynamics/docs/design/old_vs_current_transition_compare_v1.md)
- [evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)
- [design/model_variant_aliases.md](/home/wys/uwnav_dynamics/docs/design/model_variant_aliases.md)

## 2. 当前文档分层

### 2.1 根目录文档：只放高层入口与契约

- [handover_guide.md](/home/wys/uwnav_dynamics/docs/handover_guide.md)：当前阶段交接第一入口
- [project_status.md](/home/wys/uwnav_dynamics/docs/project_status.md)：当前事实、风险与下一步
- [data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)：数据目录与 git 入库规则
- [config_contract.md](/home/wys/uwnav_dynamics/docs/config_contract.md)：训练/评估配置契约
- [evaluation_protocol.md](/home/wys/uwnav_dynamics/docs/evaluation_protocol.md)：评估口径与 artifact 约定
- [engineering_roadmap.md](/home/wys/uwnav_dynamics/docs/engineering_roadmap.md)：工程升级路线
- [modeling_roadmap.md](/home/wys/uwnav_dynamics/docs/modeling_roadmap.md)：建模路线
- [repo_index.md](/home/wys/uwnav_dynamics/docs/repo_index.md)：关键文件定位索引
- [repo_tree.txt](/home/wys/uwnav_dynamics/docs/repo_tree.txt)：仓库结构快照

### 2.2 `design/`：设计说明与接口约束

- 数据契约、绘图规范、阶段设计说明
- [design/model_variant_aliases.md](/home/wys/uwnav_dynamics/docs/design/model_variant_aliases.md)：网络方案简称规范

### 2.3 `reference/`：命令手册与执行参考

- [reference/quick_commands.md](/home/wys/uwnav_dynamics/docs/reference/quick_commands.md)：当前主线命令手册
- [reference/cli_command_index.md](/home/wys/uwnav_dynamics/docs/reference/cli_command_index.md)：各 CLI 入口索引与最小调用模板

### 2.4 `math/` 与 `thesis/`：理论与论文支撑材料

- `math/`：LaTeX 理论说明
- `thesis/`：论文写作与结果审计材料

### 2.5 `archive/`：不作为当前主入口的历史草稿

- [archive/figures_mermaid.md](/home/wys/uwnav_dynamics/docs/archive/figures_mermaid.md)：历史 Mermaid 草图，保留但不作为当前主说明

## 3. 当前整理规则

- `docs/` 根目录只保留高层入口、契约和路线图。
- 命令大全、服务器操作手册这类“查阅型文档”放进 `docs/reference/`。
- 草稿、临时图示、已退出主线的辅助材料放进 `docs/archive/`。
- 代码结构说明、算法设计与接口约束继续放在 `docs/design/`。

## 4. 本次最关键的治理变化

- `data/raw/` 作为原始数据底稿目录，允许入 git。
- `data/interim/`、`data/processed/`、`out/`、`runs/` 作为可重建产物目录，不入 git。
- `raw/` 目录不再混放 `_aligned.*`、绘图输出、离线分析目录等派生产物。

详细规则见 [data_management.md](/home/wys/uwnav_dynamics/docs/data_management.md)。
