# GNN+RL 动态拓扑算法设计基线 V1（初稿）

状态：供讨论与下一阶段实现使用的初步冻结基线。
适用节点：V15 分层 GNN 已通过短闭环 RL 初始化资格验证之后。
目标：在继续编写 PPO 代码前，固定职责边界、网络骨架、动作结构、
奖励/约束接口、训练流程和验收方法，避免在实现过程中反复改变算法定义。

本文件不替代早期的 `gnn_rl_design_baseline.md`。早期文件记录研究空间和
候选路线；本文件从中选择一条当前主线，并结合已完成实验形成实施规范。

## 1. 问题定义

将星群动态拓扑控制定义为一个带安全门控和资源约束的序贯决策问题：

```text
可信分布式估计器
Schmidt / EKF / exact covariance transport / replay
                    ↓
部署安全的动态图观测
                    ↓
Residual Edge-aware MPNN
                    ↓
Masked Hierarchical PPO
                    ↓
合法 keep / add / remove / swap
                    ↓
通信、量测、滤波与下一决策状态
```

优化目标是长期估计质量，而不是单步反事实最优动作命中率。通信、时延、
拓扑变化和安全风险保留为独立成本或硬约束。

## 2. 不可破坏的架构边界

以下内容作为硬约束冻结：

1. 轨道动力学、量测模型、状态更新和协方差一致性由现有估计层负责。
2. GNN 不直接输出卫星物理状态，也不替代 EKF、Schmidt、CI、协方差传输、
   replay 或动力学传播；该边界在后续版本中保持不变，而非仅为 V1 简化。
3. RL 只选择合法拓扑/通信动作，不直接修改状态和协方差。
4. CI 只用于同一物理状态的相关估计，不直接融合不同卫星的物理状态。
5. policy observation 不含仿真真值、未来噪声、未来 RMSE 或 NEES。
6. 真值可用于仿真训练奖励和离线评价；NEES 仅用于训练诊断和评价。
7. 所有学习动作必须通过确定性合法性与安全层；`keep` 是最终 fallback。

当前不将学习 CI 权重、直接状态估计、模态选择和连续资源分配纳入主线。
其中，学习 CI 权重不是本路线的必需环节，可以长期不开展；只有同一物理状态
确实形成多源相关估计、现有确定性 CI 成为可测瓶颈且收益能够独立验证时，才
作为后置研究项重新评估。无论是否学习 CI 权重，CI 的适用边界均不改变。

## 3. 图定义与输入

### 3.1 多图语义

正式实现必须区分：

- 候选图：物理和通信上可能建立的边，用于动作生成与动作评分；
- 活动图：当前实际启用的通信拓扑；
- 信息流图：本周期实际发生的量测、状态消息或估计依赖关系。

当前把候选边复制为双向消息边的做法只属于 GNN 接口验证，不作为 PPO
正式消息传递基线。第一版 PPO 应以实际信息流边进行消息传递，同时让候选边
进入动作评分器。

### 3.2 节点候选特征

- 平移、尺度归一化后的相对位置和速度；
- `log1p` 协方差对角线及可用性掩码；
- 绝对导航可用性；
- NIS、连续异常和量测可用性摘要；
- history、journal、pending delivery、replay、fallback 状态；
- 重同步、挂起邻居和节点健康状态；
- 当前活动度数与资源占用。

不直接使用未归一化的完整绝对 ECI 状态。允许字段全集先冻结，最小特征子集
通过后续消融确定。

### 3.3 边候选特征

- 归一化距离和相对速度；
- 几何可见性与预计剩余可见时间；
- 通信可用性、时延和丢包率；
- 量测模态、age、NIS 和异常计数；
- 当前是否 active、是否为估计依赖；
- 链路最后接收时间、失联长度和资源状态；
- 预期 replay/resynchronization 风险。

物理不可行、故障端点和违反工程规则的边首先被硬门控，而不是仅依赖 GNN
学习低权重。

### 3.4 全局特征

- 时间或 episode 进度；
- 候选边、活动边和资源预算；
- 平均/最大时延、信息 age 和资源占用；
- 最近拓扑变化率、驻留时间和 cooldown；
- 全群在线一致性与估计健康摘要。

连续特征必须使用固定、可复现且不依赖未来结果的归一化。缺失值必须同时
提供 availability mask，不能用无语义的零静默替代。

### 3.5 时序与部分可观测性

第一版不急于引入 RNN，但必须按 POMDP 设计观测接口。节点和边至少保留
`age`、最后接收时间、连续丢包长度、prediction-only 持续时间、历史窗口占用、
等待重同步状态和上一决策动作。若前馈 MPNN 在长时延或长失联场景明显退化，
再比较帧堆叠与 GRU/LSTM，而不是预先扩大网络。

节点编号和候选边排列不得携带策略语义；任何仅用于日志或动作索引的
`action_id` 不得作为模型特征。

## 4. GNN backbone

第一版冻结为 `Residual Edge-aware MPNN`：

```text
node feature ── node MLP ── h_i
edge feature ── edge MLP ── g_ij

m_ij = MLP_message(h_i, h_j, g_ij)
m_i  = mean/sum_{j in N(i)} m_ij
h_i' = h_i + MLP_update(h_i, m_i)
```

基线超参数：

| 项目 | V1 默认值 |
|---|---:|
| message-passing layers | 2 |
| hidden size | 64 |
| 激活 | ReLU |
| 聚合 | mean |
| graph pooling | mean + max |
| Actor/Critic encoder | V1 独立；Actor 可 warm-start |

3 层消息传递、attention 和共享 Actor/Critic encoder 只作为后续消融。
当前不采用 GAT、Graph Transformer、GraphSAGE 或 MARL 作为首版主线。

## 5. 层次动作与策略分解

### 5.1 第一级：动作类型

```text
a_type ∈ {keep, add, remove, swap}
```

类型概率只在当前至少存在一个合法实例的类型上归一化。

### 5.2 第二级：类型内动作

- `keep`：无第二级选择；
- `add`：选择一条合法非活动候选边；
- `remove`：选择一条合法活动边；
- `swap`：选择一条删除边和一条新增边。

正式 Actor 不再把所有 swap 组合视为互不相关的扁平类别。建议分解为：

```text
score_swap(e_remove, e_add)
    = score_remove(e_remove)
    + score_add(e_add)
    + score_pair(e_remove, e_add)
```

V1 可先令 `score_pair=0`，再根据类型内 Top-K 与 regret 决定是否加入小型
配对 MLP。

### 5.3 合法动作掩码

掩码是算法定义的一部分，至少保证：

- 可见性、通信规则、健康状态和硬资源限制；
- add/remove 与当前 active 状态相符；
- 驻留和 cooldown 合法；
- 动作后满足必要连通性；
- swap 的删除边和新增边组合合法；
- 无其他合法动作时 `keep` 可执行。

非法动作 logit 置为负无穷，不允许 PPO 在非法集合上探索。

## 6. Actor-Critic 与 PPO

第一版采用 centralized graph policy。这里的 centralized 仅表示策略读取全局
在线安全图，不表示使用集中式状态估计器。

```text
GraphObservation
       ├── warm-start Residual Edge-aware MPNN
       │        ├── type Actor
       │        └── add/remove/swap conditional Actor
       └── independent graph value encoder
                ├── task Critic V_r(G)
                └── cost Critics V_c1(G), ..., V_ck(G)
```

PPO-V1 使用 clipped PPO、GAE、entropy bonus、合法动作 masked categorical
distribution。Critic 随机初始化；Actor encoder、type head 和可兼容的类型内
评分头从当前监督 checkpoint warm start。

V1 不共享 Actor 与 Critic 参数，以保持 warm-start 参数语义稳定，并隔离
value loss 对已验证策略表示的干扰。只有独立 Critic 被证实造成明显的计算或
样本效率瓶颈时，才重新比较共享 encoder。

建议从一开始保留 task value 与多个 cost value 输出，即使 PPO-V1 暂时使用
penalty reward，以便后续升级 Lagrangian PPO 时不重构网络接口。

## 7. Reward 与 Cost

### 7.1 任务奖励

PPO-V1 只使用归一化估计精度改善作为主奖励：

```text
r_est(t) = [E(t) - E(t+1)] / [E_scale + epsilon]
```

其中第一版 `E` 采用全群位置 RMSE，`E_scale` 使用 episode 初始 RMSE 或训练
场景固定统计尺度。episode 末尾加入归一化终端精度奖励。

不在每一个 PPO step 内额外运行反事实 keep 分支。相对 keep 的收益用于离线
评价，不作为在线训练必须付出的双分支计算。

### 7.2 辅助信息增益

信息增益仅作为可选 shaping：

```text
r_IG = q_consistency * [logdet(P_before) - logdet(P_after)]
```

PPO-V1 默认 `beta_IG=0`。只有确认在线一致性 gate 后才启用，避免奖励虚假
协方差收缩。

### 7.3 独立成本向量

环境独立返回：

```text
c = [communication, delay, topology, risk]
```

- `communication`：消息量/字节数相对预算归一化；
- `delay`：实际交付延迟、信息 age 或超时比例；
- `topology`：switch 与 resynchronization 的组合成本；
- `risk`：fallback、断连、协议失败和严重在线一致性异常。

建议：

```text
c_topology = c_switch + alpha_resync * c_resync
```

避免对高度相关的 switch 和 resync 重复惩罚。

### 7.4 约束训练顺序

PPO-V1 先采用 penalty：

```text
r' = r_est - lambda_comm*c_comm - lambda_delay*c_delay
           - lambda_topology*c_topology - lambda_risk*c_risk
```

但日志和 trajectory 始终保存独立 reward/cost。PPO-V1 稳定后，再实现
Lagrangian PPO/CMDP，使用明确预算而不是继续人工调大 penalty。

## 8. 决策周期与 episode

不允许每个滤波历元无驻留地反复切换拓扑。冻结机制如下：

- Actor 仅在决策时刻运行；
- 一个动作至少保持 `K` 个滤波历元；
- 驻留期间滤波、量测和通信继续运行；
- cooldown、安全门和紧急故障处理仍有效。

PPO-V1 默认 `K=5`；`K∈{2,5,10}` 作为首轮消融。episode 长度先按决策次数
定义，具体数值在算法调试场景测定后冻结。

## 9. 训练阶段

### Stage 0：PPO 正确性调试

- 3～5 星、RANGE-only；
- 人为构造明确的导航失效或链路恢复机会；
- 第一项冻结测试采用三星、一次拓扑决策、随后固定运行完整滤波评价窗口；
- 独立短时域 Oracle 只确定验收动作，不向 PPO 提供动作标签；
- 验证 mask、log-prob、GAE、PPO update 和 checkpoint；
- 要求随机初始化策略能从 keep 收敛到唯一的已知正确行为。

该阶段是算法单元测试，不替代 Walker-20 主验证。

### Stage 1：随机初始化 PPO 基线

- 随机初始化 GNN+Actor；
- 记录样本效率、收益、cost、动作熵和失败事件；
- 用于判断 RL 本身是否可学。

### Stage 2：监督 warm-start PPO

- 迁移当前分层监督 GNN 的兼容参数；
- Critic 随机初始化；
- 初期可降低 encoder 学习率或短暂冻结，但不得长期冻结 encoder；
- 可使用随训练衰减的行为克隆正则，避免初始策略骤变而又允许 RL 超越教师；
- 与 Stage 1 使用完全相同的环境分布和预算。

### Stage 3：场景 curriculum

依次加入通信成本、delay、dropout、link failure、navigation outage、动态
可见性和多模态条件；再从小星座扩到 Walker-20。

Stage 1 的第一版冻结为三星 RANGE-only、12 个滤波历元、每 2 个历元决策、
最小驻留 1 个决策周期，并在 8 个可复现环境 seed 间循环。每个 seed 随机化
一个节点的绝对导航失效窗口、0～20% 丢包率和 0～2 s 通信时延。

首轮尺度审计显示：每 episode 通信/回放约 21～36 次、切换 0～3 次、重同步
0～6 次，任务回报约 1.7～3.5。PPO-V1 因而采用每决策 4 条消息、1 次切换和
2 次重同步作为归一化尺度。初始权重 0.05、0.02、0.02 经统一倍率扫描后，
Stage 1 训练倍率冻结为 0.05，即实际权重分别为 0.0025、0.001、0.001。
replay 与通信当前高度相关，V1 只记录而不重复惩罚。

### Stage 4：约束与风险

在 penalty PPO 稳定后升级 Lagrangian PPO，并加入尾部风险评价。只有现有
一致性机制经过相应场景验证后，才启用信息增益 shaping。

## 10. Domain randomization 与数据划分

训练 episode 随机化：

- Walker 初相位、轨道构型和初始估计误差；
- Q、R、量测噪声和传感器可用性；
- packet loss、delay、链路/节点故障时间；
- navigation outage 节点与持续时间；
- visibility timing 和资源预算。

最终划分不能只按随机 seed：

- 训练集：场景族 A 及其随机 seed；
- 验证集：未见 seed 和部分未见故障组合；
- 测试集：未见 Walker 相位、可见性和失效组合。

当前 seed 0～5/6～7 的划分只证明噪声过程隔离，不代表物理场景泛化。

## 11. 基线与消融矩阵

### 非学习基线

- always keep；
- full/static topology；
- random legal；
- LowChurnConnectedTree / low-churn observable；
- information greedy；
- age-greedy / longest-outage-first；
- short-horizon oracle（仅仿真上界）。

### 学习基线

- 当前监督 hierarchical GNN；
- MLP+PPO（无图消息传递）；
- 随机初始化 GNN+PPO；
- warm-start GNN+PPO。

### 首要研究问题

1. RL 是否优于监督 GNN？
2. GNN 是否优于 MLP？
3. warm start 是否提高样本效率和稳定性？
4. 学习策略是否优于确定性策略？
5. 资源和风险是否满足预算，而非仅提高平均 RMSE？

## 12. 评价指标

每个策略统一报告：

- 全程/终端位置 RMSE、速度 RMSE、最差节点 RMSE；
- NIS、NEES 和协方差校准指标；
- 消息量、字节量、delay、age 和丢包；
- replay、resynchronization、switch、fallback；
- 断连、协议失败和严重退化事件；
- reward、各 cost return、动作熵和类型/边选择多样性；
- 分别记录动作类型熵和条件边选择熵，避免总熵掩盖某一层策略塌缩；
- 多 seed 均值、置信区间、P10/下尾和最坏场景；
- 训练样本效率和推理时间。

不得只报告训练 reward 或平均 RMSE。

## 13. PPO-V1 进入实现前的暂定参数

| 参数 | 初稿建议 | 状态 |
|---|---:|---|
| GNN layers | 2 | 暂定 |
| hidden size | 64 | 暂定 |
| decision dwell K | 5 | 待 K=2/5/10 对照 |
| task error | position RMSE | 暂定 |
| information shaping | 关闭 | 冻结到 PPO-V1 |
| RL algorithm | clipped PPO + GAE | 冻结 |
| action structure | masked hierarchical | 冻结 |
| encoder sharing | V1 不共享 | 冻结 |
| cost interface | comm/delay/topology/risk | 冻结 |
| constraint method | penalty → Lagrangian | 冻结顺序 |
| warm start | encoder + hierarchical Actor | 暂定 |
| Stage 0 场景 | 三星 RANGE-only；一次决策 + 六历元窗口 | 冻结 |
| Stage 0 行为阈值 | 3/3 seed 最终确定性动作正确；末20回合采样正确率不低于90% | 冻结 |

实现前还需一次性确定：

- episode 的决策步数；
- terminal reward 权重；
- cost 的归一化基准和 PPO-V1 penalty；
- cost Critic 数量及输出接口；
- 实际信息流边在 `GraphObservation` 中的正式字段；
- 训练/验证/测试的物理场景清单。

## 14. PPO-V1 代码里程碑验收

只有满足以下条件才算 PPO-V1 闭环完成：

1. 固定 seed 时 rollout 和 update 可复现；
2. policy 输入不含真值和未来信息；
3. 分层 log-prob、entropy 和 mask 正确；
4. 非法动作概率为零且始终存在 keep fallback；
5. reward 与每个 cost 独立返回、累计和导出；
6. GAE、clipping、value loss 和 checkpoint 有单元测试；
7. Stage 0 能学习已知策略，不只降低 loss；
8. 随机初始化和 warm-start 使用相同训练预算；
9. 对节点重编号和候选边重排进行置换测试，策略分布和执行结果保持等价；
10. PPO 训练使用多个随机种子，结论不依赖单次幸运训练；
11. Walker-20 未见场景无协议错误、滤波发散或不可接受退化；
12. 现有 V14/V15 估计、反事实和拓扑测试保持通过。

## 15. 当前不做与重新开启条件

| 暂不采用 | 重新考虑条件 |
|---|---|
| 独立自监督 GNN | PPO 样本效率或跨场景迁移不足 |
| GAT/Graph Transformer | MPNN 消融显示表达瓶颈 |
| MARL/分布式 Actor | centralized graph policy 成熟且部署要求明确 |
| 学习 CI 权重 | 默认可不开展；仅在同一状态多源相关估计中，确定性 CI 被证实为瓶颈 |
| 连续资源动作 | 离散拓扑策略稳定且资源模型可信 |
| 模态选择 RL | 拓扑控制闭环完成并明确传感器调度需求 |
| 精确监督 Oracle 拟合 | 只作为诊断，不重回主线 |

## 16. 当前节点结论

现有 hierarchical GNN 已满足 RL 初始化候选的最低条件：在线推理无真值、
合法动作执行率 100%、未见 seed 的短闭环平均不劣于 keep，且未出现滤波
灾难。它不是成熟的独立拓扑优化器，也不需要继续追求精确反事实 Oracle。

下一步应先评审并补齐本文件第 13 节的实施参数，再按第 14 节验收标准实现
PPO-V1。未经设计基线评审，不继续新增 GNN loss、复杂图模型、自监督或 MARL
分支。

### Stage 0 首轮结果

三星 RANGE-only、一次决策加六历元滤波窗口的随机初始化 PPO 已通过算法
单元验证。3 个策略 seed 的初始确定性动作均为 keep，训练后均收敛到独立
Oracle 的 action 2；末 20 回合采样正确率分别为 100%、100%、90%。

监督 GNN warm-start 在 3 个 seed 上初始确定性动作已经全部为 action 2，训练后
仍全部正确；末 20 回合采样正确率同样为 100%、100%、90%。连续 20 次正确的
首次位置分别为 51、17、40，而随机初始化为 54、57、未达到。由此只得出：

- warm-start 与 PPO 接口兼容，并可提供合理初始策略；
- 它在部分 seed 上提高稳定速度，但当前样本不足以宣称稳定的总体优势；
- 随机初始化也能完成 Stage 0，因此 warm-start 是推荐初始化而非算法依赖；
- 后续必须在多场景 Stage 1 使用相同预算继续比较，不能据单一固定场景定论。

### Stage 1 接口体检

20 回合随机初始化短跑中未出现 NaN、KL early-stop 或两级熵塌缩，说明多步
rollout、随机故障、GAE 和 penalty 更新链路数值可用。但后 8 回合平均 RMSE
约 0.804 m，略差于前 8 回合约 0.771 m，不能据此声称策略已经改善。正式扩大
训练量之前，必须先用相同未见 seed 对 keep、information-greedy、随机初始化
PPO 和 warm-start PPO 进行配对评价。

20 回合训练后在未见 seed 100～107 上的小预算对照进一步表明：keep、
information-greedy、随机初始化 PPO、warm-start PPO 的平均 RMSE 分别约为
0.850、0.837、0.837、0.892 m。随机初始化 PPO 此时与 information-greedy
产生相同行为；warm-start PPO 切换和重同步更多，尚未形成有效多步策略。

更重要的是，现有 penalty 使 information-greedy 相对 keep 的精度收益约
0.0125 被额外成本约 0.182 覆盖，说明初始成本权重明显偏强。该结果只用于
暴露尺度问题，不作为策略排名结论。扩大训练前先在固定评价轨迹上扫描 penalty
缩放，并保留 task reward 与 cost 的独立报告。

统一 penalty 缩放扫描的策略排序转折位于约 0.05～0.10；Stage 1 选择 0.05
作为保守训练倍率。按新权重训练 40 回合后，随机初始化 PPO 仍与
information-greedy 相同（平均 RMSE 约 0.837 m），warm-start PPO 改善至约
0.866 m，但仍不及 keep。

动作诊断表明，随机初始化 PPO 的确定性序列稳定为首次 add、随后 keep；
warm-start 则几乎在每个 cooldown 解除时选择 swap。warm-start 的动作类型熵
约为 0.01，而 swap 类型内熵约为 0.69，说明主要问题是监督 checkpoint 的类型
先验过强，而非边选择完全塌缩。此外，在驻留约束下每个 episode 只有约 3 个
真正可选择动作，逐 episode 更新的方差偏大。下一轮实现应：

- 汇总多个环境 seed 的 rollout 后再执行 PPO update；
- advantage 在跨 episode batch 上归一化；
- warm-start 保留 encoder 与边表征，但降低或重置 type head 的先验强度；
- keep-only cooldown 步仍进入价值回报，但不作为有效 Actor 学习样本。

上述修正已经实现：每 8 个环境 episode 汇总后更新，GAE 保持 episode 边界，
advantage 在跨 episode batch 上归一化；cooldown 的单一 keep 步只训练 Critic；
warm-start 保留 encoder/边表征但将 type head 置零。40 回合复核中，先前的
warm-start swap 偏置消失，随机与 warm-start PPO 都稳定退回 keep，RMSE 和
成本与 keep 基线一致。

这说明训练稳定性问题得到缓解，但 Stage 1 尚未学到 information-greedy 的首个
add。下一步先导出每批有效 Actor transition 数量、各动作类型概率、advantage
分布和 explained variance；在确认信号稀疏程度前，不继续扩大网络或宣称
warm-start 无效。

诊断表明，每个 8-episode batch 共 48 个 transition，其中 29～33 个为有效
Actor 样本，并非样本被 cooldown 全部耗尽；但 Critic explained variance 只有
约 0.01～0.02，且原实现 40 个 episode 只有 5 次全批 PPO update，KL 仅约
1e-5，策略概率几乎不移动。

在保持跨 episode batch 的同时，加入大小 16 的随机 minibatch 和每批 4 个
update epoch 后，随机初始化 PPO 在 40 回合重新学到首次 add，未见 seed 上与
information-greedy 一致（RMSE 约 0.837 m）；warm-start 保持安全 keep（约
0.850 m）。Critic explained variance 后期仍接近零或略负，因此 Stage 1 当前
只能认定 Actor 学习链路有效，不能认定价值模型或 warm-start 优势已经成立。

进一步使用 3 个策略训练 seed、相同训练环境 seed 0～7 和未见测试 seed
100～107 配对复现后，随机初始化仅 seed 0 学到 add 并优于 keep，seed 1、2
均退回 keep；3 个 warm-start seed 全部退回 keep。因此 Stage 1 当前成功率为
随机初始化 1/3、warm-start 0/3，不能通过稳定性验收。此前单 seed 的成功只
能证明学习链路可行，不能证明训练方案稳健。

任务可学习性审计显示，沿 keep 参考轨迹的 48 个决策时刻中，成本修正后的
最优类型为 keep 20 次、swap 25 次、add 3 次；约 54% 的时刻具有超过
0.001 m 的正 penalized gain。因此三星任务并非简单到只有 keep，但动作优势
很弱：最佳相对 keep 增益中位数约 0.0049 m，最佳与次优 margin 中位数约
0.0047 m。

同时，1步与2步前视的最优具体动作一致率约64.6%，1步与3步仅47.9%，并存在
keep、add、swap 之间的双向翻转。当前主要困难因而是弱信号和时域敏感，而非
单纯动作空间太小。后续5星场景在训练前也必须通过动作多样性、margin 和前视
稳定性审计；短时 Oracle 只能作为诊断，不能直接充当稳定长期标签。

随后修复了纯 cooldown minibatch 被整体跳过的问题：此类样本现在只更新
Critic、不产生 Actor loss。专门测试确认 Critic 参数会更新，但 seed 1 的 40
回合复核中 explained variance 仍约在 -0.08～0.04，策略仍为 keep。这排除了
“Critic 只是漏掉 cooldown 样本”这一主要解释；后续应检查 value target 的
跨场景可预测性、回报噪声和 Critic 输入表达，而不是继续增加同类 PPO 步数。

### 随机场景分布与五星预审计

Stage 1 不再把三星规模和随机扰动范围隐含在环境实现中。紧凑星群场景现在通过
显式分布配置给出星数、候选邻居 Top-K、丢包范围、通信时延范围和导航中断节点
数量；同一配置与 seed 在训练、验证和反事实审计中产生相同场景。最终训练目标
仍是随机场景分布上的泛化，不以挑选某个 Oracle 表现理想的固定场景代替分布。

首轮五星 Top-2 小样本审计（seed 0～1、每个 seed 3 个决策点）得到 6 个有效
决策点，每点 18 个合法动作；成本修正后的最优类型均为 swap，所有决策点相对
keep 的收益均超过 0.001 m，最佳收益中位数约 0.0294 m，最佳与次优间隔中位数
约 0.00621 m。该结果说明五星场景相较当前三星样本提供了更强的非 keep 信号，
但样本数太少且动作类型尚不多样，不能据此认定分布已经定型。

seed 0 的两步前视复核中，单步与两步的具体最优动作一致率仅 33.3%，但动作类型
均保持 swap，说明五星任务仍具有明显的边级时域敏感性。后续采用分层审计：先在
较多随机种子上做单步动作多样性与 margin 扫描，再对固定的训练外种子子集做多步
稳定性检查；确认分布不是单一 swap 偏置后，才开始五星 PPO 训练。

扩大到 seed 0～7 后，共得到 24 个决策点：成本修正后的最优类型为 swap 21 次、
add 2 次、keep 1 次；原始精度最优同样以 swap 为主，双导航中断对照也没有改变
这一结构，因此该现象不是成本权重或单故障数量造成的。将初始拓扑扩展为
chain/ring/star 随机族后，类型计数仍相近，但24个决策点出现了17种不同的最优
具体边动作，最常见的单个动作仅出现3次。这说明任务不是固定换同一条边，而是
“类型以 swap 为主、边级决策随场景变化”；不应为了表面类别均衡而人为扭曲任务。

五星基线由此固定为 Top-2 候选、chain/ring/star 初始拓扑族、0～0.2 丢包、
0～2 s 时延和单节点导航中断。种子分区固定为训练 0～63、验证 100～115、测试
200～215，三区互斥。验证 seed 100～101 的两步时域复核中，具体动作一致率为
66.7%，6个决策点的类型均保持 swap。该分布已具备开展小预算五星 PPO pilot 的
条件，但 pilot 仍只用于检验学习链路和跨 seed 泛化，不作为最终策略性能结论。

首轮五星 PPO pilot 使用16回合和验证 seed 100～103。随机初始化与现有 Walker
层级 GNN warm-start 均确定性选择 keep，平均 RMSE 约 1.09886 m；同期 keep 与
information-greedy 分别约为 1.09886 m 和 1.09717 m。两种初始化最后一批均有
17/24 个有效 Actor transition，正 advantage 比例约47%，但 KL 仅约 1e-4，说明
短预算下策略移动不足。把学习率提高至 1e-3 并训练32回合后，KL 提高到
0.003～0.009，但策略仍为 keep，Critic explained variance 仍为负。

因此下一步不继续盲目增加 PPO 回合或网络规模，而是复用现有反事实数据接口，
构建一个小型、严格 seed 隔离的五星层级 GNN 初始化实验。该初始化只需证明能够
区分场景相关的 swap 边并让 PPO 离开 keep；训练后仍由 PPO reward 优化，监督
Oracle 不作为最终策略或长期标签。

五星轻量初始化随后完成了两轮验证。第一轮使用18个训练组和6个验证组，验证动作
类型命中率达到100%，平均所选动作收益约 +0.0119 m；保留同任务类型头后，策略
能够离开 keep，且在 seed 100～103 上经16回合 PPO 后平均 RMSE 由约1.08579 m
改善到1.07062 m。但完全未参与模型选择的 seed 200～203 上平均结果反而劣于
keep，因此该结果不能作为泛化成功证据。

扩展到48个训练组和12个验证组后，验证 regret 由约0.0283 m 降至0.0131 m，
但 seed 200～207 盲测仍仅4/8改善，平均 RMSE 比 keep 差约0.0140 m。进一步沿
当前初始化策略采集换边后的状态进行一轮 on-policy 数据聚合，盲测仍仅3/8改善，
平均比 keep 差约0.0131 m；PPO16没有消除退化。

独立快照分析显示测试动作的单步平均收益可以为正，而连续执行却退化，说明主要
问题不是单纯样本量或置信阈值，而是一步反事实标签与序列目标不一致：策略形成
“swap—冷却 keep—再次 swap”的节奏，后一次局部有利换边可能破坏长期收益。
因此停止继续扩大一步监督数据集，不将本轮实验检查点作为正式基线。下一轮应优先
比较2步/3步回报标签、限制 episode 内换边预算，或直接改善 PPO 的序列信用分配；
不能根据单步 Oracle 指标宣称五星初始化已通过验收。

为隔离连续换边影响，环境增加了可选的 episode 最大拓扑切换次数约束，默认关闭。
在五星盲测中限制为一次后，一步初始化由4/8改善提高到5/8改善，但平均 RMSE 仍比
keep 差约0.0126 m，说明第二次短视换边只是部分原因，首次换边本身仍有风险。
使用两步前视标签后，盲测策略完全退回 keep：避免退化但没有获得精度收益。

进一步在一次切换预算下训练64回合纯 PPO，策略同样选择 keep；但 Critic explained
variance 从接近0逐步提高到约0.52，表明简化动作序列后价值学习已经明显改善，
“仍选择 keep”不能继续简单归因于PPO没有更新。当前更可能的问题是：Oracle换边
收益主要受未来量测噪声影响，而不是由策略当前可见的故障和几何特征稳定决定。

下一项审计应解耦场景条件种子与过程/量测噪声种子：固定导航中断、丢包、时延和
初始拓扑，只改变噪声实现，统计最优动作类型、具体边和收益符号的一致率。如果
标签随噪声大幅翻转，则应重新设计具有持续且可观测异质性的训练场景，而不是继续
调整GNN或PPO；如果标签稳定，再检查现有 GraphObservation 是否遗漏关键特征。

条件/噪声种子解耦后的首轮配对审计固定3个场景条件，每个条件改变8组滤波与量测
噪声。在同一条件下，逐噪声 Oracle 的动作类型始终为 swap，且最佳收益均为正；
但具体最优边分别出现5、7、7种，证明精确边标签高度受噪声实现影响。逐次Oracle
因此只适合作为不可实现上界，不适合作为确定性策略的直接监督标签。

按动作签名跨8组噪声取期望后，每个条件仍存在稳定的鲁棒动作：平均收益约为
0.0280～0.0329 m，在62.5%～87.5%的噪声样本中收益超过0.001 m；其收益仅达到
逐次Oracle均值的33.4%～50.5%。下一轮监督初始化若继续，应使用“场景条件下跨
噪声期望收益”或鲁棒动作标签，而不是每个噪声实现的精确Oracle边。RL评估也应
与该期望基线和 keep 比较，不能再以逐次Oracle命中率作为主要验收指标。

已实现噪声鲁棒快照数据集：每个场景条件选择一个代表性在线观测作为策略输入，
动作目标由同一条件下多个噪声实现的收益与成本取平均；训练/验证按条件种子隔离，
不会把同一物理条件的噪声副本拆到两侧。该接口保留真实在线输入，不向策略暴露
condition seed、真值或未来噪声。

首轮最小可行性实验使用4个训练条件、1个验证条件、每条件2组目标噪声，只训练
首次决策；在4个全新条件、每条件4组噪声的16组盲测中，平均 RMSE 从 keep 的
1.20772 m 降至1.19919 m，平均改善约0.00853 m。逐样本仅7/16优于keep，且四个
新条件中一个显著改善、三个轻微退化，因此只能说明“期望收益标签方向可行”，
不能认为鲁棒初始化已经通过泛化验收。下一步应按条件分片、可恢复地扩大条件覆盖，
并报告逐条件平均收益、最差条件退化和改善条件比例。

后续二维对照表明，纯均值目标即使扩展到8个条件、每条件4组噪声，也只达到整体
近似持平：32组盲测平均改善约0.000205 m、16/32逐样本改善，逐条件仍有明显
退化。原因是对多个候选动作取最大平均值仍存在选择偏差，且类型头在所有训练条件
中只看到“必须换边”。

数据接口随后加入可配置的下置信收益目标：

`robust_gain = mean(gain) - beta * std(gain)`

默认 beta=0 保持旧数据兼容；beta=1 时，keep 的收益与方差保持为0，只有跨噪声
收益足够稳定的换边才优于keep。使用4个训练条件、1个验证条件、每条件4组噪声
训练首次决策后，三个固定训练随机种子在8个全新条件×4组噪声上的结果均优于
keep：平均改善分别约0.00847、0.01453、0.00599 m；逐样本改善分别为24/32、
28/32、22/32；逐条件改善分别为7/8、7/8、6/8。由此可认定 beta=1 LCB
初始化满足进入PPO对照的最低条件，但尚不能作为最终鲁棒策略。

正式候选固定使用预先指定的训练 seed 0，不根据测试结果选择seed 1。对应合并数据
与检查点保存为 `results/v15_robust_lcb_conditions40_43_val50.npz` 和
`results/v15_robust_lcb_gnn_conditions40_43_val50_seed0.pt`。下一步在保持该初始化
不变的前提下进行相同条件分布上的PPO微调，并继续使用独立条件60～67作为盲测，
验证RL是否提高平均收益且不扩大最差条件退化。

在LCB seed 0初始化上，原Stage 1的8-episode batch、学习率3e-4微调64回合后，
平均改善由0.00847 m下降到约0.00508 m，Critic explained variance持续为负，
说明每批仅8个有效Actor样本时更新方差过大。保持总回合和场景不变，将batch扩大
到32、学习率降至1e-4后，盲测平均 RMSE 降至1.14532 m，相对keep改善约
0.01180 m，并在7/8个条件上平均改善；唯一退化条件约为-0.00191 m。逐噪声组合
为23/32改善，略低于初始化的24/32，但条件级最差退化和总体均值均更好。

由此固定首个可接受的五星鲁棒 PPO pilot 配置：64回合、训练条件40～43、噪声
0～7、一次切换预算、batch 32、minibatch 16、4个update epoch、学习率1e-4，
并保留LCB初始化类型头。该结果证明“GNN提供鲁棒初始状态，再由PPO微调”的链路
可行；尚未证明更长训练、更多条件或Walker-20迁移必然继续改善。

对上述固定配置补做PPO训练随机种子0、1、2复核后，三个种子在同一组独立条件
60～67、每条件4组噪声上，相对keep的平均改善分别约为0.01180、0.00508和
0.01233 m，逐噪声改善均为23/32；但逐条件改善分别为7/8、4/8和7/8，最差
条件退化分别约为-0.00191、-0.00698和-0.00191 m。因此当前配置可作为后续
稳健化实验的固定候选，但尚未通过“结论不依赖单次训练随机性”的正式验收。
下一轮优先保持数据、网络和评估集不变，比较更保守的Actor更新（如降低学习率、
KL早停或按验证条件保留最佳checkpoint）；不得按盲测结果挑选策略种子，也暂不
扩大网络或迁移到Walker-20。
