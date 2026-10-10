# 多目标分布式协同估计实现边界

## 当前处理链路

每个观测节点持有独立的 `OnlineMultiTargetTracker`，独立执行量测关联、
局部滤波、机动检测和航迹生命周期管理。节点之间不共享滤波器内部状态，
只发送同一物理目标的 J2000 后验状态、协方差、质量、航迹身份和有限长度的
信息谱系。

`DistributedTargetNodeNetwork` 按以下顺序处理一个历元：

1. 每个在线节点处理自己的观测和本地航迹；
2. 可选的 `CooperativeTrackIdentityResolver` 将各节点本地航迹映射为协同身份；
3. `DistributedTargetConsensus` 按拓扑投递目标报告；
4. 延迟报告传播到当前历元后，按 `target_id + track_id` 独立执行 CI；
5. 协同估计映射回节点本地身份，作为下一历元局部滤波先验。

## 一致性约束

- CI 只融合多个节点对同一物理目标的估计，不融合不同目标或不同卫星的物理状态。
- 一至三份报告使用原有同时 CI；更多报告使用按观测节点标识排序的确定性顺序 CI。
- 输入顺序不会改变四个以上节点的顺序 CI 结果，但顺序 CI 不等价于全局最优多元 CI。
- 每条消息携带信息标识。纯回传且没有新信息的报告会被拒绝；部分相关报告仍采用 CI 保守融合。
- 信息标识仅保留最近 256 个，控制长期运行中的消息尺寸。
- 目标级 CI 消息包含完整后验，因此链路恢复不依赖历史重放。精确协方差传输和 Schmidt
  重放继续用于相对观测的 consider-state 链路，不与目标级 CI 混用。
- 新本地航迹采用节点内联合一对一运动学分配；休眠全局航迹按轨道动力学传播到
  当前时刻。模糊匹配不会被强制送入 CI。

## 已验证范围

- 三观测节点、两目标独立融合；
- 延迟、丢包、节点退出和恢复；
- 不同节点使用不同本地目标名称时的身份映射和反向反馈；
- 航迹终止后的身份绑定清理；
- 链式拓扑中的跨历元多跳信息传播；
- 四、六、十节点的链式和全连接目标级协同预扫描。
- 三观测节点、两目标的原始雷达/红外/光学量测、动态物理可见性、
  本地关联滤波、目标级 CI 和反馈端到端链路。
- 五观测节点、两目标、600 秒三模态动态可见性正式场景；
- 十观测节点、两目标、600 秒三模态动态可见性正式场景。
- 五观测节点场景中的 10 秒通信延迟、20% 随机丢包、节点连续退出恢复，
  以及全连接到链式拓扑的联合切换。
- 五个随机种子、五观测节点、两目标、600 秒长弧通信扰动正式验收。
- 三观测节点、两目标最近约 30.86 m 的受控强交叉；交叉期间全节点遮挡
  70 秒并重建本地航迹后，全局身份无交换、无错误新生且全部重捕获。
- 原始无标签雷达、红外和光学图像链路的完整端到端验收：各节点独立完成
  自主 IOD、局部跟踪、航迹终止与重新 IOD，再执行全局身份恢复和目标级 CI。
  三个随机种子均无身份交换或错误全局新生，最大最终位置误差为 61.93 m。
- 运行中动态成员变更：原身份创建节点退出、新节点加入并继承已有身份，随后
  其余原节点全部退出。两个全局目标身份均保持不变，且退出节点的在途消息
  不再参与后续 CI。

本机正式运行的规模结果如下。耗时用于同一环境内的相对比较，不作为跨设备
实时性承诺。

| 观测节点 | 目标 | 时长/步长 | 最大位置误差 | 最大速度误差 | 平均每历元耗时 | PSD 失败 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 2 | 600 s / 10 s | 143.92 m | 2.586 m/s | 0.374 s | 0 |
| 10 | 2 | 600 s / 10 s | 133.42 m | 2.537 m/s | 0.843 s | 0 |

五随机种子通信扰动验收中，本地目标保持率均为 100%，最终协同状态可用率最低为
90%，最大连续缺失为 3 个历元，节点恢复后最长 10 秒重新收到协同结果。最大位置
误差为 242.43 m，最大速度误差为 2.581 m/s；拒收消息数和协方差 PSD 失败数均为 0。

该端到端场景可通过以下命令重复执行：

```powershell
python -m experiments.run_multitarget_multimodal_dynamic_visibility `
  --duration 600 --dt 10 --observers 5 `
  --output results/multitarget_multimodal_dynamic_visibility/summary.json
```

通信扰动与拓扑切换可通过以下参数复现：

```powershell
python -m experiments.run_multitarget_multimodal_dynamic_visibility `
  --duration 120 --dt 10 --observers 5 --seed 19 `
  --communication-delay 10 --packet-loss 0.2 `
  --outage-node observer-5 --outage-start 30 --outage-end 50 `
  --topology fully_connected --topology-switch-time 60 `
  --topology-after-switch chain
```

多随机种子长弧通信验收可通过以下命令重复执行：

```powershell
python -m experiments.run_multitarget_communication_robustness_acceptance `
  --seeds 0 1 2 3 4 --duration 600 --dt 10 --observers 5
```

强交叉和长遮挡身份重捕获场景可通过以下命令执行：

```powershell
python -m experiments.run_multitarget_identity_reacquisition_acceptance
```

原始无标签三模态完整验收可通过以下命令执行：

```powershell
python -m experiments.run_multitarget_autonomous_identity_acceptance --seed 0
```

动态节点加入、退出和身份继承可通过以下命令执行：

```powershell
python -m experiments.run_multitarget_dynamic_membership_acceptance
```

## 尚未覆盖

- 超过三份报告时顺序 CI 与通用多元 CI 优化器的精度对比；
- UDP、跨进程部署和真实时钟性能测试。

规模预扫描只验证当前目标级消息与 CI 路径，不代表完整多模态系统已经通过规模验收。
