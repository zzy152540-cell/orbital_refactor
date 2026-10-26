# 多目标十项指标临时验收配置 V3

本配置用于验收前冻结 F01--F04、P05、R06、P07、R08、P09 和 F10 的测试口径。机器可读配置为 `configs/multitarget_acceptance_candidate_v3.json`。

## 1. 冻结结论

共同场景固定为 15 颗我方观测星、15 个敌方目标、120 s、1 s 采样，使用 seeds 0--4。核心候选固定为同一先验上的雷达与 LOS 联合量测更新，并启用 realtime CANN。

R06 控制组及三类缺失组统一采用雷达主用、光电径向代理故障接管策略。光学和红外代理的距离标准差均为 2.5 m、距离率标准差均为 0.01 m/s；雷达正常时代理径向分量被抑制，雷达缺失时红外优先接管、光学备用。该理想化代理不映射到真实传感器，结果和报告必须继续标记 `physicalValidation=false`。

## 2. 完整指标冻结表

| 指标 | 对照与候选 | 主指标 | 门限 | 当前五 seed 证据 |
|---|---|---:|---:|---:|
| F01 | 15我方+15敌方目标 | 卫星总数 | ≥25 | 配置30颗，已打通 |
| F02 | 敌方目标状态输出 | 位置、速度、加速度分别输出 | 三类齐全 | 已实现；加速度由匹配速度有限差分得到 |
| F03 | 三模态原始前端 | 光学、红外、雷达 | 三类齐全 | 原始帧、量测及协方差均进入链路 |
| F04 | 单观测星与固定拓扑协同 | 两条策略+CANN参与 | 功能齐全 | 两条路径均可独立运行 |
| P05 | 单观测星旧多模态无 CANN → 单观测星联合融合+CANN | 共同样本位置 RMSE 改善 | ≥8% | 池化 83.10%，最差 seed 78.12% |
| P07 | 联合融合+CANN 独立节点 → 同算法固定拓扑共识 | 共同样本位置平均误差改善 | ≥6% | 池化 18.90%，最差 seed 15.22% |
| R06 | 统一雷达主用/光电故障接管配置下，正常三模态 → 单模态时段缺失 | 池化位置 RMSE 增幅 | ≤15% | 光学 +0.48%、红外 +0.10%、雷达 +3.67%，三类均通过 |
| R08 | 无失效 → 三节点观测和通信同时失效 | 存活节点共识位置平均误差增幅 | ≤12% | 四类组合均通过，位置均值增幅 −4.08% 至 −0.15% |
| P09 | 目标报告构造 → 接收节点应用 | 平均同步延迟 | ≤100 ms | 120帧平均0.307 ms，P95 0.372 ms，最大0.607 ms |
| F10 | 30星双方可视化 | 要素与交互检查 | 全部具备 | 双方30星录像与自动数据回归通过，待人工GUI检查 |

P05 的 83.10% 是“联合融合+CANN”组合改善，主要由联合融合贡献；不得写成 CANN 单独改善 83.10%。P07 必须使用联合融合算法的独立/共识同算法配对，不能与旧联邦融合臂混比。

## 3. 明确排除的实验设置

- 不使用旧独立 `RADIAL_BACKUP` 影子通道；
- 不使用 7 跳同步传播实验作为本轮主验收配置；
- 不启用完整径向血缘整报告门控；
- 不使用仅距离代理；
- 不把理想化光电径向代理写成已经具备物理传感器依据。

## 4. P05/P07 执行命令

```powershell
0..4 | ForEach-Object {
  python -m experiments.run_bilateral_baseline_matrix `
    --duration 120 `
    --seed $_ `
    --output "results/external_acceptance/frozen_v3/baseline_seed${_}_120s.json"
}

python -m experiments.summarize_bilateral_baseline_matrix `
  results/external_acceptance/frozen_v3/baseline_seed0_120s.json `
  results/external_acceptance/frozen_v3/baseline_seed1_120s.json `
  results/external_acceptance/frozen_v3/baseline_seed2_120s.json `
  results/external_acceptance/frozen_v3/baseline_seed3_120s.json `
  results/external_acceptance/frozen_v3/baseline_seed4_120s.json `
  --output results/external_acceptance/frozen_v3/p05_p07_summary_5seed_120s.json
```

汇总时读取 `experimentalJointMeasurementCandidate` 作为 P05，读取 `adjustedJointMeasurementAlgorithm` 作为 P07；不得读取旧 P05/P07 默认字段作最终判定。

## 5. R06 执行命令

```powershell
# 每个 seed 共享一次正常前端，并派生三类缺失组
0..4 | ForEach-Object {
  python -m experiments.run_bilateral_dropout_prescan `
    --duration 120 `
    --dropout-start 30 `
    --dropout-end 89 `
    --seed $_ `
    --acceptance-cann-only `
    --local-unified-multimodal-update `
    --electro-optical-radial-failover `
    --eo-radial-range-sigma-m 2.5 `
    --eo-radial-range-rate-sigma-mps 0.01 `
    --modalities OPTICAL INFRARED RADAR `
    --output "results/external_acceptance/r06_eo_failover_formal/r06_eo_failover_seed${_}_120s.json"
}
```

控制组和三个实验组必须统一使用上述故障接管参数；不得同时传入旧 `--backup-radial-*` 或 `--independent-radial-backup-channel` 参数。不要传入 `--distributed-consensus`、`--prefer-complete-radial-lineage` 或多跳参数；当前五 seed 通过证据来自独立节点池化口径。

## 6. R08 执行命令

以下命令模板分别用于空间分散、拓扑相邻和高可见性组合；三组索引依次为 `0 5 10`、`0 1 2`、`0 12 14`。

```powershell
0..4 | ForEach-Object {
  python -m experiments.run_bilateral_node_failure_prescan `
    --duration 120 `
    --failure-start 30 `
    --failure-end 89 `
    --seed $_ `
    --failed-observer-indices 0 5 10 `
    --local-unified-multimodal-update `
    --output "results/external_acceptance/frozen_v3/r08_spatial_seed${_}_120s.json"
}
```

随机组合必须按 seed 固定：seed 0=`7 8 11`，seed 1=`6 7 11`，seed 2=`1 3 10`，seed 3=`1 2 10`，seed 4=`9 13 14`。不得在看到结果后重抽节点。

## 7. 验收报告用语

可以写：P05、R06、P07、R08 和 P09 在当前冻结的工程数值配置下达到 L2 门限；R06 四组统一采用雷达主用、光电径向代理故障接管配置，三类模态缺失均满足门限。

不应写：现有红外/光学传感器已经实现该径向精度；CANN 单独贡献 P05 的全部改善；部分节点径向备份已经通过；当前结果已经完成物理传感器正式验收。

## 8. P09执行命令

```powershell
python -m experiments.run_multitarget_state_sync_latency_acceptance `
  --frames 120 `
  --threshold-ms 100 `
  --output results/external_acceptance/frozen_v3/p09_state_sync_latency_120frames
```

P09不包含本地滤波、原始传感器生成、GUI和文件记录。约152 ms的算法整历元耗时是单独的实时性风险，不得替代或否定P09消息同步延迟结果。

## 9. F01--F04与F10回归

```powershell
python -m pytest `
  tests/multitarget/test_bilateral_scenario_contracts.py `
  tests/test_multimodal_sensor_simulator.py `
  tests/test_raw_sensor_frame_contract.py `
  tests/test_raw_sensor_frame_serialization.py `
  tests/test_visualization_data_contract.py `
  tests/test_visualization_replay_model.py `
  tests/test_visualization_observation_panel.py `
  tests/test_v15_visualization_raw_frontend.py `
  tests/test_bilateral_visualization_recording.py `
  -q
```

生成并直接打开双方30星回放：

```powershell
python -m experiments.run_bilateral_visualization_recording `
  --duration 20 --dt 1 --seed 0 `
  --cann-node-limit 3 `
  --output results/visualization_recordings/bilateral30_20s_seed0 `
  --open
```

也可打开设置页面，在 `Scenario` 中选择 `Bilateral: 15 friendly + 15 hostile`，或加载 `configs/visualization_bilateral30.json` 后点击 `Run simulation`：

```powershell
python -m experiments.run_v15_visualization_launcher
```

新生成录像的CANN主视图为由方向环与RT解码共同转换得到的绝对ECI位置X/Y/Z（km）及三轴历史。原轨道相位只作为内部转换诊断保留，不再作为新录像的主显示；旧录像仍可按旧相位界面兼容回放。

F10还必须人工检查30星双方角色、通信边与观测边、三模态面板、敌方目标估计和误差、CANN方向环/位置热力图、播放暂停及目标切换。当前双方入口已经具备上述数据；人工GUI检查通过后才能关闭F10。

## 10. 当前验收就绪度

- 已有L2开发证据：F01、F02、F03、F04、P05、R06、P07、R08、P09；
- R06三模态数值门限通过：光学 +0.48%、红外 +0.10%、雷达 +3.67%；四组统一采用光电径向故障接管配置，旧独立径向备份未启用；
- 未闭环：F10双方30星自动数据回归已通过，仅剩人工GUI检查。
