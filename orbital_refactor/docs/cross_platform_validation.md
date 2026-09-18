# Windows 收口与 Ubuntu 兼容性验证基线

## 1. 目的与边界

本文档固化进入 Ubuntu 验证前的 Windows 参考环境、统一冒烟入口和分层测试顺序。
跨平台验证检查公开接口、数值结果和可视化兼容性，不改变滤波、CANN或拓扑算法。

## 2. 依赖入口

- 核心与测试：`python -m pip install -r requirements.txt`
- 可视化：`python -m pip install -r requirements-visualization.txt`
- GNN/PPO研究环境：`python -m pip install -r requirements-gnn.txt`
- 也可按 `pyproject.toml` 安装：`python -m pip install -e .[visualization]`

PyTorch不属于正式状态估计主链的必要依赖。Ubuntu首次验证先安装核心依赖；通过后再分别增加
可视化和GNN依赖，避免将CUDA、Qt或显示服务器问题误判为滤波问题。

## 3. Windows参考快照（2026-09-18）

| 项目 | 参考值 |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.5 |
| SciPy | 1.15.3 |
| pytest | 9.1.1 |
| Matplotlib | 3.10.9 |
| PySide6 | 6.8.3 |
| pyqtgraph | 0.14.0 |
| PyTorch（研究链） | 2.13.0 |

这些版本是可复现参考值，而不是全部平台必须逐项相同的硬锁定值。正式兼容范围由
`pyproject.toml`和requirements文件中的版本约束决定。

## 4. 统一冒烟入口

在项目根目录运行：

```bash
python -m examples.run_cross_platform_smoke
```

该命令依次执行：

1. 构造标准`ModuleInput`；
2. 直接运行`StateAwarenessModule`；
3. 将请求保存为无pickle的JSON+NPZ数据包；
4. 从数据包重新加载并运行；
5. 保存并重新加载`ModuleOutput`；
6. 以严格容差比较直接结果和数据包结果；
7. 输出平台、依赖版本、公开协议版本和数值指纹。

返回码0且JSON字段`ok=true`表示公开接口最小链路通过。Windows与Ubuntu的浮点结果使用容差
比较，不使用文本或二进制逐字节比较。

## 5. Windows已知问题

完整`python -m pytest`在当前Windows环境进入
`tests/test_absolute_anchor_and_nees.py::test_absolute_anchor_reduces_selected_position_covariance`
时，曾在`numpy.linalg.pinv`调用处发生原生进程`Fatal Python error: Aborted`。设置
`OMP_NUM_THREADS=1`和`MKL_NUM_THREADS=1`后仍可复现。

该现象没有产生Python断言失败，也不位于公开接口改动链。相关接口定向回归已经通过。
因此将它登记为数值库/运行环境待对照项：Ubuntu应先单独运行该测试；若Ubuntu通过，再回到
Windows检查NumPy、SciPy、BLAS/MKL组合，而不修改算法规避环境崩溃。

## 6. Ubuntu分层验证顺序

1. 创建全新Python 3.10环境并安装`requirements.txt`；
2. 运行`python -m examples.run_cross_platform_smoke`；
3. 运行公开接口、序列化和核心滤波定向测试；
4. 单独运行上述绝对导航/MKL对照测试；
5. 运行完整非GUI测试集；
6. 安装可视化依赖，先以无显示方式生成记录和图片；
7. 在具备桌面或X/Wayland条件时测试Qt窗口；
8. 如需继续GNN/PPO，再建立独立研究环境并测试CPU/CUDA路径。

通过记录至少应包含：Git提交号、操作系统、Python及依赖版本、执行命令、通过/失败数量、
冒烟JSON、关键数值差异、运行时间和失败堆栈。
