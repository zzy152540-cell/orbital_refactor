# 大项目集成快速开始

## 1. 稳定边界

集成方只依赖以下两个入口之一：

```python
from interfaces.state_awareness_module import StateAwarenessModule
```

或使用跨进程数据包：

```python
from interfaces.public_api import run_module_bundle
```

`adapters`、`orbital_core`、`pipelines`、`cooperative`、`experiments`和
`brain_inspired`均属于内部实现。集成方不应依赖这些包的聚合导出。

## 2. 目录约定

当前仓库包含一层Python项目子目录：

```text
Git仓库根目录/orbital_refactor/pyproject.toml
```

安装、测试和运行命令均在含有`pyproject.toml`的目录执行。程序不得依赖盘符、用户目录或
当前机器的绝对路径；输入、输出和数据集路径均由调用方提供。

## 3. 环境层级

```bash
# 正式核心运行
python -m pip install -e .

# 核心回归与旧版数值对照
python -m pip install -r requirements-test.txt

# Qt可视化
python -m pip install -r requirements-visualization.txt

# GNN/PPO研究链
python -m pip install -r requirements-gnn.txt
```

Qt和PyTorch不是正式滤波主链的必要依赖。

## 4. 集成前验证顺序

### L0：公开接口冒烟

```bash
python -m examples.run_cross_platform_smoke
python -m pytest tests/test_cross_platform_smoke.py tests/test_public_api.py \
  tests/test_module_serialization.py tests/test_interface_contracts.py \
  tests/test_raw_sensor_frame_serialization.py -q
```

### L1：核心算法

```bash
python -m pytest tests/test_import_boundaries.py \
  tests/test_centralized_pipeline_export.py tests/test_nn_interface_adapter.py \
  tests/test_single_satellite_cann_sidecar.py \
  tests/test_multimodal_sensor_simulator.py \
  tests/test_single_satellite_multimodal_source.py \
  tests/test_multi_neighbor_replay_coordinator.py \
  tests/test_network_schmidt_orchestrator.py -q
```

### L2：平台数值库对照

```bash
python -m pytest \
  tests/test_absolute_anchor_and_nees.py::test_absolute_anchor_reduces_selected_position_covariance \
  -q -vv
```

### L3：可视化数据与离屏GUI

```bash
python -m pytest tests/test_visualization_data_contract.py \
  tests/test_visualization_simulation_config.py -q
QT_QPA_PLATFORM=offscreen python -m pytest \
  tests/test_v15_topology_control_visualization.py -q
```

GNN/PPO和长时间、20星、Monte Carlo测试在上述层级通过后单独执行，不作为外部状态估计接口
首次联调的阻塞条件。

## 5. 文件交换

请求和响应使用版本化JSON+NPZ目录，不使用pickle。集成方应：

1. 为每次请求选择新的输出目录；
2. 保存`module_bundle.json`与`arrays.npz`的完整组合；
3. 检查公开协议版本；
4. 使用结构化错误字段`error_type`、`code`、`field`和`message`；
5. 不解析Python traceback作为业务错误协议。

## 6. 集成记录

每轮联调至少记录Git提交号、系统、Python及依赖版本、配置文件、执行命令、输入数据版本、
通过/失败数量、数值指纹、运行时间以及异常堆栈。算法正式输出与实验性CANN/GNN功能应分别标注。
