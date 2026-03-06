# 技术审计（2026-03-06）

审计基线：
- 基于当前工作区，而非仅 `HEAD`。
- 审计范围覆盖 `align -> build_dataset -> train -> eval -> cli` 主链路，以及当前新增的 `dataset/experiment/train/runtime` 相关基础设施。

已执行的本地命令：
- `pytest -q`
  结果：测试收集阶段即失败，`apps/dev/test_*.py` 与 `tests/*.py` 都因 `ModuleNotFoundError: No module named 'uwnav_dynamics'` 中断，最终 `7 errors during collection`。
- `PYTHONPATH=src pytest -q tests`
  结果：`1 failed, 5 passed`；唯一失败为 `tests/test_align_multirate.py::test_align_multirate_masks`。
- `PYTHONPATH=src python -c '<block-enabled eval parity check>'`
  结果：同一份权重在“训练配置启用 thruster_lag”和“评估配置默认关闭 thruster_lag”时，`max|dY diff|=0.0066051632`、`max|logvar diff|=0.0081077218`，并打印 `True False`，说明权重可加载但前向语义已偏离。
- `PYTHONPATH=src python -c '<float compare check>'`
  结果：`t_main[60]=0.6`、`t_pw[3]=0.6000000000000001`、`t_pw[3] <= t_main[60]` 为 `False`，差值为 `-1.1102230246251565e-16`。

---

## P0 / PR-1

### PR 标题
评估端复用完整训练模型配置，避免 block-enabled checkpoint 静默失真

### 修改文件列表
- `src/uwnav_dynamics/eval/config.py`
- `src/uwnav_dynamics/train/config.py`
- `tests/` 下新增一个 `eval-config parity` 单测

### 风险点
- 评估端当前只重建 `din/dout/pred_len/rnn_*` 等基础字段，没有把 `model.blocks.*`、`use_hydro_feat` 对应的完整训练语义复原。
- `S1Predictor.forward()` 明确依赖 `cfg.blocks.thruster_lag.enabled`、`cfg.use_hydro_feat`、`cfg.blocks.uncertainty.enabled` 等开关；因此“同一份权重 + 不同 cfg”会得到不同输出，而且 `load_state_dict(strict=True)` 不会报错。
- 这会让 block 消融实验出现“训练是 A，评估是 B”的静默污染，最坏情况下会把整组离线指标做成伪结果。

### 证据
- `src/uwnav_dynamics/eval/config.py:118-139`
  `_build_model_config()` 只构造 `S1PredictorConfig(...)` 的基础字段，完全未读取 `model.blocks`。
- `src/uwnav_dynamics/models/nets/s1_predictor.py:223-257`
  前向路径直接依赖 `cfg.blocks.*.enabled` 和 `cfg.use_hydro_feat`。
- 本地命令输出：
  `max|dY diff|=0.0066051632`
  `max|logvar diff|=0.0081077218`
  `thruster_lag enabled: True vs False`

### 验收方式（最小 smoke test）
- `PYTHONPATH=src pytest -q tests/test_eval_config_parity.py`
- 覆盖点：
  同一 train yaml 经 train/eval 两侧构造出的 `S1PredictorConfig` 必须等价；
  同一 state_dict + 同一输入在 train/eval 两侧模型输出必须逐元素一致。

---

## P0 / PR-2

### PR 标题
修复 Power 对齐的浮点边界错误，保证 `power_mask` 与采样点一一对应

### 修改文件列表
- `src/uwnav_dynamics/preprocess/align/aligner.py`
- `tests/test_align_multirate.py`

### 风险点
- Power 对齐走 `_sample_last_before_with_max_dt()`，循环里使用 `t_src[i] <= tk` 和严格 `max_dt=0` 比较。
- 当主轴与源轴是浮点数时，逻辑上相同的采样点可能因为 IEEE 误差被错判为“不在当前时刻之前”，从而漏掉有效样本。
- 当前最直观后果是 `power_mask` 偶发性少命中，后续窗口 mask、辅助特征和对齐 QA 都会被污染。

### 证据
- `src/uwnav_dynamics/preprocess/align/aligner.py:247-292`
  `_sample_last_before_with_max_dt()` 未做任何浮点容差处理。
- `src/uwnav_dynamics/preprocess/align/aligner.py:563-584`
  Power 主流程直接依赖该函数产出的 `power_mask`。
- `tests/test_align_multirate.py:86-110`
  期望命中索引是 `0,20,40,60,80`。
- 本地测试输出：
  `N_main_with_power=4`
  断言失败位置为 `tests/test_align_multirate.py:110`。
- 本地浮点复现：
  `0.6` vs `0.6000000000000001`
  `t_pw[3] <= t_main[60] == False`

### 验收方式（最小 smoke test）
- `PYTHONPATH=src pytest -q tests/test_align_multirate.py::test_align_multirate_masks`
- 额外建议补一个更小的单元测试，只校验 `_sample_last_before_with_max_dt()` 在 `max_dt=0` 且存在浮点舍入误差时仍能命中理论同一采样点。

---

## P1 / PR-3

### PR 标题
打通 `dvl_mask/power_mask` 到 loss 与评估，停止把前向填充速度当无条件真值

### 修改文件列表
- `src/uwnav_dynamics/preprocess/build_dataset.py`
- `src/uwnav_dynamics/train/data_pipeline.py`
- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/models/losses/nll.py`
- `src/uwnav_dynamics/eval/evaluate.py`
- `tests/` 下新增一个 `mask-aware loss/metric` 单测

### 风险点
- 数据集构建阶段把 DVL 速度前向填充成 `Vel*_state_mps`，并把这三列直接写进 `target_cols`；这已经把原本稀疏的 DVL 监督稠密化了。
- 同时代码又额外保存了 `dvl_mask/power_mask`，并在配置注释中写明“后续在训练脚本中做逐元素 loss masking”，但训练和评估主链路目前都没有真正消费这些 mask。
- 结果是模型会把“沿时间保持的旧 DVL 速度”当作每一步都同样可信的监督目标；评估指标也会把这些伪稠密标签一起算进去，偏离“稀疏监督”的设计目标。

### 证据
- `configs/dataset/pooltest02_s1.yaml:55-58,70-85`
  输入和目标都使用 `VelX_state_mps/VelY_state_mps/VelZ_state_mps`，并明确写了“后续在训练脚本中用 has_dvl 做逐元素 loss masking”。
- `src/uwnav_dynamics/preprocess/build_dataset.py:145-188`
  `_add_state_velocity_cols()` 对 DVL 速度执行 `ffill()`，开头缺失还会填 `0.0`。
- `src/uwnav_dynamics/preprocess/build_dataset.py:231-275`
  数据集同时保存了 `dvl_mask_hist/dvl_mask/power_mask_*`。
- `src/uwnav_dynamics/train/data_pipeline.py:46-81,117-125`
  训练侧只读取 mask 的 shape，真正 DataLoader 仍然只返回 `(X, Y)`。
- `src/uwnav_dynamics/train/run_train.py:34-39`
  loss 函数只有 `(model, X, Y)` 三个输入，没有 mask。
- `src/uwnav_dynamics/eval/evaluate.py:112-115,161-181`
  评估只打印发现了 mask，随后仍对全部元素直接求 RMSE/MAE。

### 验收方式（最小 smoke test）
- `PYTHONPATH=src pytest -q tests/test_mask_aware_loss_eval.py`
- 覆盖点：
  只在 `dvl_mask==1` 的位置累计速度维 loss/metric；
  对无效位置改写标签值后，masked loss/metric 必须保持不变。

---

## P1 / PR-4

### PR 标题
统一 run 目录契约，消除 `out_dir` 与 `variant` 的重复嵌套

### 修改文件列表
- `configs/train/pooltest02_s1_lstm_v0.yaml`
- `src/uwnav_dynamics/experiment/layout.py`
- `src/uwnav_dynamics/train/run_train.py`
- `src/uwnav_dynamics/cli/utils.py`
- `tests/test_experiment_layout.py`

### 风险点
- 当前运行目录契约是 `run_dir = out_dir / variant`。
- 但示例训练配置里，`run.out_dir` 已经包含一次 `B0_baseline`，同时 `run.variant` 又写了 `B0_baseline`，最终路径会变成 `.../B0_baseline/B0_baseline`。
- 这不一定马上崩，但会显著提高人工排障成本；一旦有人按注释把 `out_dir` 当“最终 run 目录”理解，就会在 split/scaler/ckpt/eval 目录上反复踩坑。

### 证据
- `src/uwnav_dynamics/experiment/layout.py:27-60`
  `RunLayout.run_dir` 固定等于 `Path(out_dir) / variant`。
- `src/uwnav_dynamics/train/run_train.py:92-99`
  训练主入口按同一契约打印并创建 `run_dir`。
- `src/uwnav_dynamics/cli/utils.py:33-42`
  CLI 也用同一契约去定位 ckpt。
- `configs/train/pooltest02_s1_lstm_v0.yaml:6-7`
  `out_dir` 已含 `B0_baseline`，`variant` 再写一次 `B0_baseline`。

### 验收方式（最小 smoke test）
- `PYTHONPATH=src pytest -q tests/test_experiment_layout.py`
- 额外建议补一个配置校验：
  若 `out_dir.name == variant`，则直接报警或拒绝加载。

---

## P2 / PR-5

### PR 标题
规范 pytest 收集边界与导入路径，恢复仓库级一键回归

### 修改文件列表
- 根目录新增 `pytest.ini` 或 `pyproject.toml`
- `apps/dev/test_dvl_plots.py`
- `apps/dev/test_imu_pipeline.py`
- `apps/dev/test_power_plots.py`

### 风险点
- 当前没有 pytest 配置文件，`pytest -q` 会把 `apps/dev/test_*.py` 当正式测试收集。
- 同时仓库默认也没有安装为包，导致未设置 `PYTHONPATH=src` 时，连 `tests/*.py` 的导入都会在收集阶段失败。
- 这会让 CI、交接和日常回归都失去“默认命令可用”的基本工程保障。

### 证据
- `pytest -q`
  结果：`7 errors during collection`。
- 错误文件包含：
  `apps/dev/test_dvl_plots.py`
  `apps/dev/test_imu_pipeline.py`
  `apps/dev/test_power_plots.py`
  `tests/test_align_multirate.py`
  `tests/test_experiment_layout.py`
  `tests/test_split_no_leak.py`
  `tests/test_train_runtime.py`
- `rg --files -g 'pyproject.toml' -g 'setup.cfg' -g 'pytest.ini' -g 'tox.ini'`
  结果为空，说明当前没有 pytest 收集边界配置。

### 验收方式（最小 smoke test）
- `pytest -q`
- 预期：只收集正式测试目录；开发脚本不再被当成测试；无需额外手工设置即可完成测试收集。

---

## P2 / PR-6

### PR 标题
补齐基础设施注释与职责说明，降低交接和误用成本

### 修改文件列表
- `src/uwnav_dynamics/experiment/layout.py`
- `src/uwnav_dynamics/eval/config.py`
- `src/uwnav_dynamics/dataset/split.py`
- `src/uwnav_dynamics/train/data_pipeline.py`
- `src/uwnav_dynamics/cli/utils.py`

### 风险点
- 这些文件负责 run 目录契约、评估配置重建、split 持久化、训练数据准备和 ckpt 解析，属于“出错时影响全局、平时又不直观”的基础设施层。
- 原始职责边界不够明确时，人类开发者很容易在 CLI、train、eval 三侧重复拼路径，或者误判 split/mask 的真实语义。
- 该项不改行为，属于低风险可读性增强，但对后续 PR 的理解成本有直接帮助。

### 证据
- 本次已完成注释性修改，未改动函数签名和行为分支。
- 可直接查看：
  `src/uwnav_dynamics/experiment/layout.py`
  `src/uwnav_dynamics/eval/config.py`
  `src/uwnav_dynamics/dataset/split.py`
  `src/uwnav_dynamics/train/data_pipeline.py`
  `src/uwnav_dynamics/cli/utils.py`

### 验收方式（最小 smoke test）
- `PYTHONPATH=src pytest -q tests/test_experiment_layout.py tests/test_train_runtime.py tests/test_split_no_leak.py`
- 预期：注释增强后行为不变，相关单测继续通过。
