# 卫星状态感知软件 RC3

RC3 将 P05/P07 验收候选组的因果二体+J2速度残差修正、对应五种子统计、
代表性回放元数据以及可选择单种子的全时长检查一并纳入独立运行包。

本目录是独立运行候选包，不要求安装 Python、Conda 或 Visual Studio Code。

设置页面的验收模式默认选择“单种子全时长”：保持120 s场景和正式故障
窗口，但仅运行seed 0，用于快速演示和流程检查。正式验收模式仍保留seed
0～4；单种子结果不得替代正式五种子统计结论。

## 启动

- Windows：双击 `SatelliteStateEstimation.exe`。
- Ubuntu：在本目录执行 `./SatelliteStateEstimation`。

程序从同目录的 `configs/` 读取可选配置，并将仿真、验收测试和回放数据写入
`results/`。请将整个目录解压到当前用户具有写权限的位置，不要只复制主程序，
也不要安装到需要管理员权限才能写入的系统目录。

## 首次检查

1. 打开设置窗口并确认各标签页可切换；
2. 运行一次自由配置的短时仿真；
3. 打开生成的录像；
4. 在“Acceptance tests”中执行接口符合性测试；
5. Ubuntu 包额外执行 Ubuntu 环境与核心回归测试。

构建提交、平台和时间记录在 `VERSION.txt`。正式验收时应保留对应的
`results/acceptance_gui/` 报告目录。

## 从源码构建

Windows：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\build_windows.ps1 `
  -PythonExe D:\Anaconda3\envs\state_estimate\python.exe
```

Ubuntu：

```bash
sudo apt install -y libxcb-cursor0
python -m pip install -r requirements-build.txt
bash packaging/build_linux.sh
```

构建脚本会依次生成独立运行目录、执行接口 smoke、执行无显示 GUI smoke，
最后在 `release/` 中生成对应平台的压缩交付包。两个平台必须分别在本机
构建，不能使用 Windows 构建结果代替 Ubuntu 构建结果。

Ubuntu 的 Qt XCB 图形插件需要系统库 `libxcb-cursor0`。构建描述还会将当前
Conda 环境中与 Python `pyexpat` 扩展 ABI 匹配的 `libexpat.so.1` 显式放入
独立运行包，避免误加载 Ubuntu 系统中的另一版本。
