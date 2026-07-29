# Orbital 工程化改写（第一阶段）

本目录是对原 `orbital/` 代码的第一阶段整理，目标是**保持算法行为不变，先抽取公共函数和接口对象**。

## 当前已完成

- 保留原始脚本到 `legacy/`，作为回归基准；
- 抽取地球常数、坐标转换、轨道动力学、量测模型和指标计算；
- 按接口文档建立 `ModuleInput`、`Observation`、`LocalEstimate`、`SingleFusionResult`、`NodeReport`、`ModuleOutput` 等数据对象；
- 加入基础单元测试；
- 未修改原有 EKF、CI、NIS 和各实验主流程的数值逻辑。

## 下一阶段

1. 抽取 `DynamicsEKF`、`CentralizedDynamicsEKF` 和 `LocalDynamicsEKF` 的公共预测/更新逻辑；
2. 将 CI 函数集中到 `orbital_core/ci_fusion.py`；
3. 为传统三模态和 NN 增强三模态建立 pipeline；
4. 添加输入、输出适配器，使现有实验脚本能够返回 `ModuleOutput`；
5. 用原脚本结果建立回归测试。

## 运行测试

```bash
python -m pytest tests
```

## 第二阶段改写

本阶段继续保持 legacy 脚本不变，新增：

- `orbital_core/filters.py`：统一单模态/联邦局部轨道 EKF，保留旧代码的固定步长数值雅可比、伪逆、Joseph 协方差更新和 NIS 软/硬门限行为。
- `orbital_core/ci_fusion.py`：统一两路与三路协方差交集融合，并返回具名权重。
- `orbital_core/quality.py`：提供不侵入滤波算法的质量评分辅助函数。
- `pipelines/single_modal.py`：将单模态运行循环从实验脚本中抽离，可输出历史结果及标准 `LocalEstimate`。
- 回归测试：直接加载 legacy `DynamicsEKF`，验证新旧预测和更新结果一致。

当前测试：`8 passed`。

下一阶段建议改写联邦 CI 主循环，并将其输出封装为 `SingleFusionResult`，随后再处理集中式流程。

## 第三阶段：联邦 CI 流程与标准输出

新增 `pipelines/federated_ci.py`，完成以下内容：

- 将传统光学、红外、雷达等局部 EKF 组织为统一联邦流程；
- 保留原代码的 CI 有效后验选择、全模态缺失时保持上一融合结果、可选回灌等数值行为；
- 输出局部状态、融合状态、协方差、NIS、门限标志、CI 权重和统计信息；
- 将模态缺失、硬门限拒绝和软门限降权转换为 `AbnormalEvent`；
- 可直接生成 `LocalEstimate`、`SingleFusionResult` 和 `ModuleOutput`。

回归测试直接调用 `legacy/federated_ci_dynamics_fusion_ekf.py`，比较新旧实现的融合状态、融合协方差和各局部滤波结果。

## v4：标准输入适配与学习增强量测

本版本新增 `adapters/module_input_adapter.py` 和统一入口
`interfaces/state_awareness_module.py`，外部调用形式为：

```python
output = StateAwarenessModule().run(module_input)
```

适配器将接口文档中的 `ModuleInput`、`Observation` 转换为现有联邦 CI
流程所需的数组和局部滤波器。为保持算法改动最小，主星状态历史和
`q_eci2pri` 历史仍作为 `config.runtime` 中的辅助模型数据提供。

新增学习增强光学量测支持：

- `Observation(modality="OPTICAL", source_type="LEARNING")` 自动映射为 `nn`；
- 支持 ECI/SPRI 下的三维位置量测；
- 支持 ECI/SPRI 下的位置—伪速度六维量测；
- NN EKF 的数值行为已与原 `federated_ci3_nn_ir_rad_fusion_ekf.py` 对照。

当前适配器为保证与旧代码一致，要求同一模态在一个任务内使用固定量测协方差。
时变量测协方差和基于置信度的动态缩放将在后续作为可选功能加入，不改变默认行为。

运行示例：

```bash
python examples/run_standard_interface.py
```

## Phase 5: SHIRT and NN prediction data adapters

This version adds a data-source layer that converts the files used by the
legacy experiments into the documented interface objects without changing the
filter or CI implementation.

New modules:

- `adapters/shirt_data_adapter.py`
  - loads SHIRT `metadata.json` and `roe*.json`;
  - preserves the verified quaternion/coordinate convention of the legacy code;
  - constructs runtime auxiliary histories and a standard `ModuleInput`.
- `adapters/nn_prediction_adapter.py`
  - reads `predictions.npz` using the existing `image_path` / `t_pred` contract;
  - aligns predictions to SHIRT filenames;
  - optionally builds pseudo velocity;
  - emits learning-enhanced optical `Observation` objects.
- `adapters/synthetic_measurement_adapter.py`
  - creates IR azimuth/elevation and radar range/range-rate observations;
  - supports explicit dropout windows;
  - emits standard traditional `Observation` objects.

The resulting application path is now:

```text
SHIRT JSON + predictions.npz
        -> data-source adapters
        -> ModuleInput / Observation
        -> StateAwarenessModule
        -> ModuleOutput
```

The checkpoint files are not required for this pipeline when precomputed
`predictions.npz` is available. Checkpoints are only required by the separate
neural-network inference/training tools.

Run all regression and adapter tests with:

```bash
python -m pytest
```

Current result: `18 passed`.

## 第六阶段：集中式融合与统一结果导出

本阶段新增集中式多模态 EKF 管线，并保持外部 `ModuleInput` / `ModuleOutput`
接口不变。通过 `config["filter"]["architecture"]` 选择单星融合架构：

```python
config["filter"]["architecture"] = "centralized"   # 集中式 EKF
# 或
config["filter"]["architecture"] = "federated_ci" # 联邦 EKF + CI（默认）
```

新增模块：

- `orbital_core/centralized_filter.py`：集中式预测、逐模态 NIS 预检、联合量测更新；
- `pipelines/centralized.py`：集中式主循环、异常事件与标准输出封装；
- `exporters/result_exporter.py`：统一导出 CSV、NPZ 和 JSON；
- `adapters/module_input_adapter.py`：新增集中式输入适配，复用原标准接口。

集中式算法支持传统光学、红外、雷达以及学习增强光学（NN）量测。工程化阶段
保留旧脚本的固定差分步长、伪逆、Joseph 协方差更新和 NIS 软/硬门限行为。

统一导出示例：

```python
from exporters import export_run_bundle

paths = export_run_bundle(
    history,
    module_output,
    output_directory="outputs/run_001",
    stem="centralized_result",
)
```

输出包括：

- `centralized_result.csv`：逐时刻位置、速度、加速度、NIS、门限与 CI 权重（如有）；
- `centralized_result.npz`：完整状态、协方差和诊断历史；
- `centralized_result.json`：标准 `ModuleOutput`，供上层系统与可视化使用。

当前回归与接口测试总计 `20 passed`。集中式轨迹和协方差与原脚本结果保持一致，
仅存在约 `1e-8 ~ 1e-7` 的浮点级差异。

## Multi-satellite cooperative extension

The first cooperative-estimation extension adds:

- absolute two-body + J2 orbit propagation;
- Keplerian-elements to ECI initialization;
- per-observer RTN/PRI frame histories;
- target/observer absolute and relative trajectory generation;
- synthetic optical, infrared, radar, and NN-surrogate observations;
- `StateAwarenessModule.run_history()` while preserving the stable `run()` API;
- multi-node history execution helpers;
- conversion of node-relative estimates to a common target absolute ECI state;
- one-to-three-node simultaneous covariance-intersection fusion.

Run the trajectory and observation example from the project root:

```bash
python examples/run_multi_sat_scenario.py
```

The existing `legacy/` directory remains unchanged and should continue to be
used only as a numerical-regression reference.

## Multi-satellite cooperative estimation

The cooperative extension provides an end-to-end path from generated orbital
truth to local filtering and epoch-wise multi-node covariance intersection.

## v13.1 symmetric fleet baseline

The v13.1 path separates the new per-satellite problem from the legacy
observer-target pipeline:

- `scenarios/fleet_scenario.py` creates a symmetric N-satellite truth scenario;
- `orbital_core/fleet_centralized_ekf.py` maintains the centralized `6N` state;
- `pipelines/fleet_centralized.py` runs the centralized inter-satellite EKF;
- distributed inter-satellite updates include neighbor covariance through
  `H_j P_j H_j^T`;
- direct CI between different satellites' absolute states is rejected.

Run the three-satellite range/range-rate comparison with:

```bash
python examples/run_v13_1_baseline.py
```

## v13.2 Fleet-State CI baseline

The v13.2 validation path lets every satellite maintain a local copy of the
same `6N` fleet state. Nodes apply only locally available observations, exchange
validated `FleetStateMessage` objects, and use CI only when the state dimension
and satellite ordering are identical. Optional absolute ECI position anchors
remove the global drift mode of a relative-only constellation.

The acceptance scenario uses reproducible Gaussian white noise for inter-satellite
range, range-rate, RTN azimuth/elevation, and the low-rate GNSS position anchor.
The generated covariance matrices match the configured noise standard deviations.

Run the four-way acceptance comparison with:

```bash
python examples/run_v13_2_fleet_ci.py
```

Run the configurable noise/process Monte Carlo sweep with:

```bash
python examples/run_parameter_sweep.py --seeds 5 \
  --modes combined --angle-sigmas-deg 1.0 2.0 \
  --process-noises 1e-7
```

The command exports one raw CSV row per seed and algorithm plus an aggregated
`*_summary.csv` containing means and standard deviations.

## v13.3 attitude MEKF migration

The attitude baseline migrated from the original thesis code uses explicit
`wxyz` quaternions and a nine-dimensional multiplicative error state:

```text
[attitude error, angular-velocity error, gyro-bias error]
```

It includes rigid-body angular-rate propagation, gyro white noise and bias
random walk, star-tracker quaternion updates, body-vector updates, NIS output,
and Joseph covariance updates. The attitude subsystem is currently independent
from the orbital filter; the next integration step will provide its estimated
inertial-to-body DCM and attitude covariance to the inter-satellite angle model.

Run the complete example from the project root:

```bash
python examples/run_multi_sat_interface.py
```

The example performs the following steps:

1. Propagates one target and three observer satellites with the two-body + J2 model.
2. Generates heterogeneous infrared, radar, and synthetic learning-enhanced observations.
3. Builds one independent `ModuleInput` and single-node filter for each observer.
4. Converts each local target-relative estimate into target absolute ECI state.
5. Fuses the available node reports by covariance intersection at each epoch.
6. Reports local and cooperative position/velocity RMSE and final node weights.

The main reusable API is:

```python
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
```

A complete communication outage after initialization is handled by holding the
last cooperative posterior. Node-level validity histories can be supplied with
`node_validity_by_node` to simulate node or link failures.
