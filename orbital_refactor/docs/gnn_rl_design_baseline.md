# V14 GNN/RL dynamic-topology design baseline

Status: design baseline, not a frozen algorithm specification.

This document separates decisions already supported by the V14 implementation
from research choices that still require controlled experiments. Its purpose is
to keep the learning layer subordinate to the trusted estimation layer.

## 1. Problem definition and architecture boundary

V14 treats dynamic topology control as risk-constrained sequential resource
allocation for distributed state estimation:

```text
orbital truth and measurements
        -> EKF / Schmidt / exact transport / CI
        -> GraphObservation
        -> GNN relation encoder
        -> RL topology decision
        -> legal-action and safety layer
        -> online communication and estimation
```

The estimator remains responsible for physical state estimation and covariance
consistency. The GNN does not replace orbital dynamics or the filter. RL selects
which legal observation/communication relations are activated. CI is used only
for correlated estimates of the same physical state, never to fuse different
satellites' physical states directly.

Primary objective: improve long-horizon fleet estimation quality, including the
worst node. Communication, latency, reliability, consistency and resynchroniza-
tion are constraints or costs rather than interchangeable objectives.

## 2. Graph and observation baseline

At decision time `t`, construct a candidate graph `G_t=(V_t,E_t^candidate)`.
The policy must receive only information available at that time. Simulation
truth, future noise, realized future RMSE and NEES are forbidden policy inputs.

### 2.1 Hard candidate-edge gates

An edge is a candidate only if it satisfies all applicable conditions:

- geometric/occultation/FOV visibility;
- designed communication rule and range limit;
- endpoint health and communication availability;
- resource and resynchronization feasibility;
- action-specific connectivity requirements.

The published four-factor adjacency construction (distance, communication rule,
fault availability and resource availability) is retained as an interpretable
decomposition. Binary physical/engineering conditions become hard gates. The
continuous factors remain separate edge features; they are not collapsed into
one fixed multiplicative score before the GNN.

### 2.2 Initial node features

- normalized covariance diagonal and selected covariance summaries;
- local NIS summaries and consecutive anomaly counts;
- absolute-navigation availability;
- replay/history occupancy and resynchronization state;
- local communication/resource budget;
- optional local state representation after translation/scale normalization.

Full absolute ECI state is not a mandatory first-version feature. Geometry is
preferably represented by relative, frame-aware edge features to improve
invariance and transfer across constellation phases.

### 2.3 Initial edge features

- normalized range and relative velocity;
- visibility flag and predicted remaining visibility time;
- measurement modalities and their availability;
- measurement age and per-modality NIS statistics;
- packet-loss estimate and communication delay;
- communication/resource availability;
- topology-active flag, last reception time and outage duration;
- expected replay/resynchronization cost;
- source/covariance freshness and trust indicators.

### 2.4 Global features

- active/candidate edge counts and current link budget;
- fleet covariance and NIS summaries;
- mean/max communication delay and resource occupancy;
- recent topology-change rate;
- decision interval and mission/scenario context that is observable online.

All continuous features require documented normalization. Missing values require
explicit availability masks rather than silent zero filling.

## 3. Action baseline

The first learning environment uses the existing structured action family:

- `keep`;
- `add(edge)`;
- connected `swap(remove_edge, add_edge)`.

Pure `remove` remains a diagnostic counterfactual unless an explicit resource
policy and connectivity proof make it operationally legal. `keep` is always
available as the safe fallback.

The policy does not output an arbitrary adjacency matrix. A deterministic legal
action generator and mask enforce visibility, communication, health, degree,
resource and connectivity rules. Rejected actions fall back to `keep` and are
reported separately.

For larger constellations, use hierarchical or autoregressive selection:

```text
select node requiring assistance
        -> select candidate neighbor
        -> select keep / add / connected swap
```

This avoids enumerating the power set of all edges.

## 4. Reward and constraint baseline

### 4.1 Separation of signals

Four signal classes must remain separately logged even if an algorithm later
combines them:

1. task return: windowed fleet and worst-node estimation improvement;
2. dense proxy: consistency-qualified information gain;
3. constraint costs: communication, delay, replay, resync and switching;
4. safety events: disconnect, protocol failure and severe consistency loss.

Training may use truth-based RMSE. Policy observations and deployment decisions
must not use truth. Evaluation reports both truth-based performance and online
observable diagnostics.

### 4.2 Task return

Use an `H`-epoch window rather than one-step error because a topology action has
delayed effects. Prefer normalized improvement relative to `keep` or the prior
window to reduce scenario-scale dependence. Report fleet RMSE, relative RMSE,
worst-node RMSE and velocity error separately.

### 4.3 Dense information signal

Candidate definitions include covariance log-determinant reduction, trace
reduction and generalized/Fisher information gain. None is accepted as valid
information merely because covariance shrinks. Apply a trust factor derived
from NIS, covariance provenance/freshness, modality health and resynchronization
state. NEES may supervise/calibrate this factor in simulation but is unavailable
as a general flight-time input.

### 4.4 Constraints and risk

First preference is a constrained MDP rather than unrestricted scalarization:

- mean and peak communication budget;
- mean/max delay and information age;
- topology switch rate;
- replay/resynchronization budget;
- consistency degradation and severe-loss probability;
- connectivity and protocol integrity as hard constraints.

Retain the existing safe-positive probability, confidence lower bound, P10,
lower-tail mean and severe-loss probability. They support risk-sensitive or
CVaR-style objectives and must remain evaluation metrics even when not used in
the first training loss.

Reward weights and limits are not frozen. They must be calibrated using the
counterfactual experiments and checked for invariance to units and horizon.

## 5. Self-supervised GNN baseline

Self-supervision is used to learn graph representations, not an artificial
human label for the globally optimal topology. Initial pretext tasks are:

- masked node/edge feature reconstruction;
- next-epoch visibility and link-failure prediction;
- future delay/loss and measurement-age prediction;
- future NIS anomaly or consistency-risk prediction;
- consistency-qualified short-window information-value prediction;
- temporal contrast between nearby graph snapshots.

Graph augmentations must preserve physical semantics. Small measurement noise,
delay perturbations and masking of redundant features are plausible. Arbitrary
deletion of critical edges, permutation that breaks orbital identity/geometry,
or treating materially different physical topologies as positive pairs is not.

The pretrained encoder may initialize an RL policy/value network. The project
does not require a complete supervised optimal-action dataset, but it still
requires simulator trajectories/replay and disjoint evaluation scenarios.

## 6. Learning stages

### Stage A - environment closure

Implement a multi-step `TopologyControlEnvironment` around the online
orchestrator. It must expose reset, legal actions/mask, step, decomposed reward,
constraint costs, termination and diagnostics. Verify fixed-seed reproducibility
and absence of truth/future leakage.

### Stage B - opportunity and baseline audit

Compare always-keep, random legal, current deterministic policy, information-
greedy policy and counterfactual oracle. If the oracle rarely differs from
`keep`, enrich the scenario distribution before training a policy.

### Stage C - self-supervised representation

Train and evaluate the graph encoder on the pretext tasks. Measure prediction,
calibration and transfer across seeds, topology phases, failures and constellation
sizes; do not accept representation loss alone as evidence of control value.

### Stage D - learning policy

Begin with single-policy centralized training on the five-node structured action
space. Compare masked value-based and policy-gradient methods only after the
environment is validated. Add constrained/risk-sensitive optimization before
claiming safety. Expand to hierarchical decisions before 20-node training.

### Stage E - held-out validation

Split by physical scenario/configuration, not merely future-noise seed. Test
unseen Walker phase, visibility pattern, navigation dropout, link fault, sensor
noise and communication regime. Always compare precision, consistency, tail
risk, resource use and switching against deterministic baselines.

## 7. Existing support and open decisions

Already implemented:

- multi-modal, visibility-aware graph observations;
- structured legal topology actions and deterministic policy interface;
- online orchestrator with loss, delay, exact transport, replay and resync;
- fixed-prefix counterfactual evaluation and adaptive Monte Carlo branches;
- RMSE, covariance, NIS, NEES, communication, replay and tail-risk metrics.

Still open and requiring experiments or source verification:

- decision interval and rollout horizon;
- exact normalization and minimal feature subset;
- information-gain definition and trust correction;
- CMDP limits and risk measure;
- centralized versus decentralized actor execution;
- DQN/PPO/constrained algorithm choice;
- self-supervised loss weights and physically valid augmentations;
- five-to-twenty-node action decomposition.

## 8. Immediate implementation acceptance criteria

The next code milestone is environment closure, not policy training. It is
accepted when:

- reset/step produce reproducible multi-step episodes;
- every action is generated from the current candidate graph and mask;
- no truth or future statistic appears in policy observations;
- reward and every constraint cost are returned separately;
- invalid/high-risk actions cannot bypass the safety layer;
- always-keep, random-legal and deterministic policies complete episodes;
- existing estimation/counterfactual regressions remain unchanged;
- an oracle audit demonstrates whether the scenario distribution contains a
  meaningful, learnable topology-control opportunity.

## 9. Literature roles to verify before freezing formulas

- Distributed orbit determination and formation-flight papers: physical
  measurement, geometry and flight realism; not direct RL prescriptions.
- Network energy minimization via sensor selection/topology control: explicit
  estimation-resource trade-off and deterministic baselines.
- Wireless/remote state-estimation scheduling: filter-policy separation,
  temporal decision state and limited-channel constraints.
- Generalized information-gain sensor selection: theoretical information-value
  definitions and non-RL oracle/reference methods.
- Dynamic GNN/MARL sensing and communication papers: graph encoder and structured
  policy design, subject to application and safety differences.
- Self-supervised graph papers: representation objectives and augmentations,
  adopted only when they preserve orbital and estimator semantics.

No formula attributed to a paper is considered final until checked against the
paper's assumptions, state definition, observability model and experimental
setting.

---

# V14 GNN/RL 动态拓扑设计基线（中文版）

状态：设计基线，并非已经冻结的最终算法规范。

本文档将 V14 当前代码已经支持的决定，与仍需通过受控实验验证的研究选择分开说明。基本原则是：学习层必须建立在可信状态估计层之上，不能绕过或替代现有估计与一致性机制。

## 1. 问题定义与架构边界

V14 将动态拓扑控制定义为一个面向分布式状态估计的、带风险约束的时序资源分配问题：

```text
轨道真值与量测
        -> EKF / Schmidt / 精确协方差传输 / CI
        -> GraphObservation
        -> GNN 星间关系编码器
        -> RL 拓扑决策
        -> 合法动作与安全约束层
        -> 在线通信和状态估计
```

状态估计器继续负责物理状态估计与协方差一致性。GNN 不替代轨道动力学模型或滤波器；RL 只负责选择要激活的合法观测或通信关系。CI 只用于融合关于同一物理状态的相关估计，不能直接融合不同卫星各自的物理状态。

主要优化目标是提高长时间范围内的星群估计质量，同时关注最差节点。通信、时延、可靠性、一致性以及重新同步应建模为约束或代价，而不是与估计精度任意交换的同级目标。

## 2. 图结构与观测基线

在决策时刻 `t`，构建候选图 `G_t=(V_t,E_t^candidate)`。策略只能接收该时刻真实可获得的信息。仿真真值、未来噪声、未来实现的 RMSE 和 NEES 都禁止作为策略输入。

### 2.1 候选边的硬门控

一条边只有满足所有适用条件时才能进入候选集合：

- 满足几何、地球遮挡和视场可见性；
- 满足预先设计的通信规则与最大距离限制；
- 两端节点健康，且通信功能可用；
- 满足资源与重新同步可行性；
- 满足具体动作要求的网络连通性。

相关文章提出的四因素邻接构造——距离、通信规则、故障可用性和资源可用性——可以保留为具有工程解释性的分解。二值的物理或工程条件用于硬门控；连续因素作为彼此独立的边特征输入 GNN，不在进入 GNN 前压缩为单一固定乘积权重。

### 2.2 第一版节点特征

- 归一化后的协方差对角线及选定的协方差摘要；
- 本地 NIS 摘要和连续异常计数；
- 绝对导航是否可用；
- 回放历史占用和重新同步状态；
- 本地通信或资源预算；
- 可选的、经过平移和尺度归一化的本地状态表示。

第一版不强制输入完整绝对 ECI 状态。几何关系优先通过相对且带坐标系语义的边特征表达，以提高策略对星座相位和构型变化的迁移能力。

### 2.3 第一版边特征

- 归一化星间距离与相对速度；
- 可见性标志和预计剩余可见时间；
- 可用量测模态及其可用状态；
- 量测年龄和分模态 NIS 统计；
- 估计丢包率和通信时延；
- 通信或链路资源可用度；
- 当前拓扑激活标志、最后接收时间和连续失联时长；
- 预计回放或重新同步代价；
- 状态来源、协方差新鲜度和可信度指标。

### 2.4 第一版全局特征

- 当前有效边数、候选边数和链路预算；
- 星群协方差与 NIS 摘要；
- 平均/最大通信时延和资源占用；
- 近期拓扑切换频率；
- 决策周期以及部署时可观测的任务或场景上下文。

所有连续特征都需要记录明确的归一化规则。缺失值必须配套显式可用性掩码，不能简单用零值代替而不加说明。

## 3. 动作空间基线

第一版学习环境沿用当前已经实现的结构化动作：

- `keep`：保持当前拓扑；
- `add(edge)`：增加一条候选边；
- `swap(remove_edge, add_edge)`：删除一条边并增加另一条边，同时保持网络连通。

单独的 `remove` 暂时只作为反事实诊断动作，除非后续明确给出资源策略并证明操作后仍满足连通和任务约束。`keep` 必须始终作为安全回退动作存在。

策略不直接输出任意邻接矩阵。确定性的合法动作生成器与动作掩码负责执行可见性、通信规则、节点健康、度约束、资源约束和连通性约束。被拒绝的动作退化为 `keep`，并单独记录拒绝原因。

对于更大规模星座，应采用分层或自回归决策：

```text
选择当前需要改善的节点
        -> 选择候选邻星
        -> 选择 keep / add / connected swap
```

这样可以避免枚举全部候选边子集导致的组合爆炸。

## 4. 奖励与约束基线

### 4.1 信号分层

即便后续某种算法会将多个指标合成标量，也必须分别记录以下四类信号：

1. 任务收益：时间窗口内星群整体和最差节点的估计改善；
2. 稠密代理信号：经过一致性校正的信息增益；
3. 约束代价：通信、时延、回放、重新同步和拓扑切换；
4. 安全事件：网络断连、协议失败和严重一致性损失。

训练时可以使用仿真真值计算 RMSE 奖励，但策略观测和部署决策不能使用真值。评价报告需要同时给出真值性能指标和在线可观测诊断指标。

### 4.2 任务收益

使用未来 `H` 个历元的窗口评价，而不是只看一步误差，因为拓扑动作的影响通常存在延迟。优先采用相对于 `keep` 或上一个时间窗口的归一化改善，以减少不同场景量纲差异。星群 RMSE、相对 RMSE、最差节点 RMSE 和速度误差需要分别报告。

### 4.3 稠密信息反馈

候选定义包括协方差对数行列式下降、协方差迹下降以及广义/Fisher 信息增益。但协方差缩小不能被无条件认定为有效信息。需要使用由 NIS、协方差来源与新鲜度、模态健康状态和重新同步状态构成的可信因子对信息增益进行修正。

在仿真中可以使用 NEES 监督或校准该可信因子，但一般飞行部署时没有真值，NEES 不能作为常规在线策略输入。

### 4.4 约束与风险

第一选择是使用约束马尔可夫决策过程（CMDP），而不是没有限制的线性加权奖励。需要考虑：

- 平均和峰值通信预算；
- 平均/最大时延与信息年龄；
- 拓扑切换频率；
- 回放与重新同步预算；
- 一致性退化和严重损失概率；
- 作为硬约束的连通性与协议完整性。

继续保留当前已经实现的安全正收益概率、置信区间下界、P10、下尾均值和严重损失概率。这些指标可以支撑风险敏感或 CVaR 类目标；即便第一版训练损失暂时不使用，也必须作为评价指标保留。

奖励权重与约束阈值目前不冻结，需要借助反事实实验校准，并检查它们对单位和预测窗口长度是否敏感。

## 5. 自监督 GNN 基线

自监督学习用于学习图表示，不是生成一个人工标注的“全局最优拓扑”分类数据集。第一批候选预训练任务包括：

- 遮蔽节点或边特征重建；
- 下一历元可见性和链路失效预测；
- 未来时延、丢包和量测年龄预测；
- 未来 NIS 异常或一致性风险预测；
- 经过一致性校正的短窗口信息价值预测；
- 相邻时间图快照的时序对比学习。

图增强必须保持物理语义。小幅量测噪声扰动、时延扰动和冗余特征遮蔽是合理候选；任意删除关键连通边、打乱会破坏轨道身份或几何关系的节点，或者将物理意义明显不同的拓扑作为正样本，都是不合理的做法。

预训练后的编码器可以用于初始化 RL 的策略网络或价值网络。本项目不要求构造完整的监督式最优动作数据集，但仍然需要仿真交互轨迹、经验回放，以及与训练场景相互独立的验证场景。

## 6. 学习阶段规划

### 阶段 A——训练环境闭环

围绕在线 orchestrator 实现多步 `TopologyControlEnvironment`。接口需要提供 reset、合法动作/动作掩码、step、分项奖励、约束代价、终止状态和诊断信息。必须验证固定随机种子的可复现性，以及不存在真值或未来信息泄漏。

### 阶段 B——决策机会与基线审计

比较始终 `keep`、随机合法动作、当前确定性策略、信息增益贪心策略和反事实 oracle。如果 oracle 也很少偏离 `keep`，则应该先丰富场景分布，而不是直接训练策略。

### 阶段 C——自监督图表示

基于预训练任务训练和评价图编码器。需要检查其预测效果、校准性，以及跨随机种子、拓扑阶段、故障状态和星座规模的迁移能力。仅仅降低自监督损失，不能作为其有助于拓扑控制的充分证据。

### 阶段 D——学习策略

首先在五节点结构化动作空间中采用单策略集中训练。只有在环境通过验证后，才比较带动作掩码的价值方法和策略梯度方法。在宣称策略安全之前，需要加入约束或风险敏感优化。扩展到 20 节点训练前，应先实现分层动作结构。

### 阶段 E——独立场景验证

数据划分应按照物理场景和构型进行，而不是只按照未来噪声种子划分。测试场景应覆盖未参与训练的 Walker 相位、可见性模式、绝对导航失效、链路故障、传感器噪声和通信条件。必须始终将精度、一致性、尾部风险、资源消耗和切换次数与确定性基线进行比较。

## 7. 已有支持与未决问题

当前已经实现：

- 多模态、可见性感知的图观测；
- 结构化合法拓扑动作和确定性策略接口；
- 支持丢包、时延、精确协方差传输、回放与重新同步的在线 orchestrator；
- 固定历史前缀的反事实评价和自适应 Monte Carlo 未来分支；
- RMSE、协方差、NIS、NEES、通信、回放和尾部风险指标。

仍需通过实验或原文核对确定：

- 决策周期与 rollout 时间窗；
- 精确归一化方式和最小必要特征集合；
- 信息增益定义及其可信修正方式；
- CMDP 约束阈值和风险度量；
- 集中式还是分布式 actor 执行；
- DQN、PPO 或约束强化学习算法选择；
- 自监督损失权重和物理合理的图增强；
- 从五节点扩展到二十节点时的动作分解方式。

## 8. 下一代码里程碑的验收标准

下一项代码里程碑是训练环境闭环，而不是立即训练策略。满足以下条件后才算完成：

- `reset/step` 能产生可复现的多步 episode；
- 每个动作都根据当前候选图和动作掩码生成；
- 策略观测不包含真值或未来统计量；
- 奖励与每项约束代价分别返回；
- 非法或高风险动作无法绕过安全层；
- 始终 `keep`、随机合法策略和确定性策略均可完成完整 episode；
- 现有估计与反事实回归测试保持不变；
- oracle 审计能够说明当前场景分布是否存在有意义且可学习的拓扑控制机会。

## 9. 在冻结公式前需要核对的文献作用

- 分布式定轨与编队飞行论文：用于确定物理量测、几何关系和飞行场景真实性，不直接规定 RL 设计；
- 基于传感器选择和拓扑控制的网络能耗优化：用于建立估计性能与资源消耗间的显式权衡和确定性基线；
- 无线/远程状态估计调度：用于支持滤波器与策略的职责分离、时序决策状态和有限信道约束；
- 广义信息增益传感器选择：用于建立信息价值的理论定义，以及非 RL 的 oracle 或参考方法；
- 动态 GNN/MARL 感知与通信论文：用于参考图编码器和结构化策略设计，但必须考虑应用和安全条件的差异；
- 自监督图学习论文：用于参考表示学习目标与图增强，只有保持轨道和估计语义时才采用。

任何来自论文的公式，在冻结进项目方案前，都必须重新核对论文假设、状态定义、可观测模型和实验条件。
