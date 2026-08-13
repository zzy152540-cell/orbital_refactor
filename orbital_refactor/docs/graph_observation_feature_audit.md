# V15 GraphObservation feature audit

Status: implementation audit for the first V15 environment milestone.

## 1. Audit categories

- **D - deployment available:** may be computed from estimator, sensor or
  communication state available at the decision time.
- **S - simulation/evaluation only:** may be used for rewards or validation but
  must not enter policy observations.
- **P - partially available:** represented by the interface, but some builders
  omit it or use a simulation-only substitute.
- **M - missing:** required by the design baseline but not represented yet.

## 2. Current field matrix

| Scope | Current field | Class | Current source and finding | V15 decision |
| --- | --- | --- | --- | --- |
| graph | `timestamp` | D | Runtime clock | Retain; encode mission time cyclically or normalize by episode duration only when that duration is observable. |
| node | `node_id` | D | Network identity | Retain for indexing, not as an ordinal numeric feature. |
| node | `state` | P | Estimated state in counterfactual builders; Walker planning currently uses truth | Require explicit estimated-state source for learning environments. Reject truth-backed builders. Prefer relative/frame-normalized geometry. |
| node | `covariance_diagonal` | D/P | Estimator covariance; optional in the data class | Require value or explicit availability mask. Add trace/logdet/condition summaries during tensorization. |
| node | `estimator_metrics` | D/P | Generic map; currently often only navigation availability | Define a versioned vocabulary instead of silently dropping unknown metrics. |
| edge | `distance` | P | Some builders compute from truth, others from visibility plan | Require a declared source: onboard estimate/measurement for policy input, truth only for evaluation. Add scale normalization. |
| edge | `geometrically_visible` | D/P | Represented, but candidate builders commonly imply visibility by membership and leave the flag true | Retain explicit flag plus availability/source metadata. |
| edge | `measurement_modalities` | D | Current observation opportunities | Retain modality mask and distinguish observable direction when needed. |
| edge | `communication_available` | D | Link/network state | Retain as hard gate and feature. |
| edge | `delay` | D/P | Configured or estimated link delay | Add availability mask and clarify commanded versus observed delay. |
| edge | `packet_loss_rate` | D/P | Configured or estimated rate | Add availability/sample-count mask; do not imply exact knowledge. |
| edge | `nis_by_modality` | D | Innovation history | Retain per modality; normalize by measurement degrees of freedom. |
| edge | `nis_sample_count_by_modality` | D | Innovation history | Retain as confidence/exposure feature. |
| edge | `consecutive_anomaly_count_by_modality` | D | Integrity diagnostics | Retain and clip/scale for tensorization. |
| edge | `observation_age` | D | Last valid relative observation | Retain with explicit availability mask. |
| graph | `previous_active_edges` | D | Current communication topology | Retain; required to define keep/add/swap and churn. |
| graph | `estimation_dependency_edges` | D | Current estimator dependency graph | Retain separately from communication topology. |
| graph | `graph_metrics` | D/P | Generic runtime totals | Define a stable vocabulary and prohibit truth-derived metrics. |
| measurement | modality/frame/covariance/attitude | D | Sensor configuration and current observation metadata | Retain directed measurement semantics; covariance and attitude require masks and normalization. |
| reward | realized RMSE/NEES/future gain | S | Counterfactual/evaluation metrics | Never include in `GraphObservation` or policy tensor. Keep in environment reward/diagnostics only. |

## 3. Missing V15 deployment features

### Node-level

- history occupancy, pinned-history count and memory/resource pressure;
- whether resynchronization is required or pending;
- local transmitted/dropped/rejected message rates over a defined window;
- local communication/resource budget and node-health flag;
- covariance freshness/source trust summaries.

### Edge-level

- relative velocity/range rate as a geometry feature independent of modality;
- predicted remaining visibility time;
- last receive timestamp and continuous outage length;
- delay/loss estimate availability and sample count;
- expected replay/resynchronization cost;
- link resource occupancy/remaining capacity;
- topology version/freshness and covariance-source trust;
- directional features when observer and receiver roles differ.

### Global

- current link budget and active/candidate edge counts;
- mean/max delay and information age;
- replay/resynchronization resource totals;
- recent topology-switch rate;
- explicit observation schema/version.

## 4. Tensorization audit

The existing `graph_action_tensor_dataset` is a V14 diagnostic tensorizer, not
the V15 deployment feature contract.

Current strengths:

- immutable node, candidate-edge, measurement and action tensors;
- directed measurement edges and BODY-frame attitude metadata;
- explicit observation-age availability;
- no future target values in input feature tensors.

Current gaps:

- raw six-state ECI values, distance, covariance and delay are not normalized;
- node `estimator_metrics` and graph metrics are discarded;
- per-modality NIS values are averaged into one scalar;
- loss/delay and covariance availability masks are incomplete;
- no resource, history, resynchronization or freshness fields;
- feature version still identifies the V14 causal diagnostic schema;
- the older seed-only split helpers are unsuitable as final physical-scenario
  validation splits, although newer Monte Carlo scenario splitting is safer.

V15 should introduce a new tensor schema rather than mutate the V14 tensorizer
silently. The schema must publish feature names, units/scales, clipping ranges,
availability masks and provenance classes.

## 5. Builder safety findings

- `short_horizon_topology_counterfactual` uses estimated decision states but
  computes candidate distance from truth. This is acceptable for controlled
  evaluation, not for a deployment observation builder.
- `v14_walker_dynamic_topology` builds planning observations directly from
  Walker truth. It is a geometry/planning baseline and must not be connected to
  a learned policy without an estimator-backed adapter.
- `walker_graph_dataset` builds policy input from provided state/covariance;
  callers must declare whether those states are estimated or true.
- `build_graph_observation` cannot currently record provenance, so identical
  data objects may hide materially different information sources.

## 6. Required interface changes before environment closure

1. Add an observation provenance/schema descriptor with at least
   `schema_version`, `state_source` and `geometry_source`.
2. Introduce typed/versioned node, edge and global metric vocabularies for the
   V15 policy path; retain generic tuples for backward compatibility.
3. Add availability masks for optional/estimated communication and covariance
   values.
4. Add the missing online resource, history, resynchronization and freshness
   summaries using orchestrator state only.
5. Create an estimator-backed online graph-observation adapter. It must never
   read truth histories or future observations.
6. Add an explicit validation function that rejects simulation-only provenance
   when an observation is passed to a deployment policy/environment.
7. Create a new normalized V15 tensorizer while retaining the V14 tensorizer as
   a regression and research-data baseline.

## 7. Recommended implementation order

1. Provenance and schema metadata plus validation tests.
2. Online estimator/orchestrator adapter for currently available features.
3. Missing-value masks and documented normalization schema.
4. Resource/resynchronization/freshness feature export.
5. V15 tensorization tests, including permutation, scale and no-truth-leakage
   checks.
6. Only then implement `TopologyControlEnvironment.reset/step`.

This audit deliberately does not freeze the final minimal GNN feature subset.
The first environment should expose trustworthy, separable signals; subsequent
ablation studies decide which features the learned policy actually needs.

## 8. Initial V15 reward and deterministic-policy calibration

The first closed-loop calibration used five nodes, six two-second epochs,
RANGE-only relative measurements, 10% packet loss, one-epoch communication
delay and seeds 0--2.  Truth was restricted to reward/evaluation; policy
observations came from the online estimator/orchestrator adapter.

With zero costs, the short-horizon oracle averaged 5 topology switches and 10
resynchronizations.  Weights of 0.005 per switch and 0.002 per
resynchronization reduced these means to 3.67 and 7.33 without degrading mean
final position RMSE (1.116875 versus 1.115330 in this sample).  This is an
initial scale estimate, not a universal reward specification.

The graph-information baseline exhibited three regimes across five seeds:

- churn weight 0.6--0.7: add an edge at every decision;
- churn weight 0.8--1.6: add one edge, then keep the topology;
- churn weight 2.0 or greater: always keep the initial topology.

The default cost-aware deterministic baseline therefore uses churn weight 1.0.
Its role is to provide a low-churn, truth-free comparator for future GNN/RL
policies.  These findings must not be generalized beyond the current small,
RANGE-only calibration until multi-modal, visibility-varying and larger Walker
experiments reproduce them.

The resumable command-line entry point is:

```powershell
python -m experiments.run_topology_reward_calibration `
  --output results/v15_reward_calibration.csv `
  --nodes 5 --epochs 6 --seeds 0,1,2 `
  --switch-weights 0.003,0.005,0.007 `
  --resync-weights 0.001,0.002,0.003 `
  --policies keep,information_greedy,short_horizon_oracle
```

Weight lists form a Cartesian grid. Each completed seed/weight/policy cell is
flushed to CSV immediately, and rerunning the same command skips existing
keys. Large Oracle scans should therefore be run in seed batches.

### Local five-node cost sweep

A three-seed local sweep showed that topology switches and resynchronizations
are currently structurally coupled: every resumed undirected edge produces two
endpoint resynchronizations. Consequently, the Oracle action is primarily
controlled by the effective cost
`switch_weight + 2 * resynchronization_weight`; many points in a two-dimensional
weight grid are behaviorally equivalent.

An effective-cost sweep produced the following mean trade-off:

| effective cost | final position RMSE | switches | resynchronizations |
| ---: | ---: | ---: | ---: |
| 0.013 | 1.119186 | 3.33 | 6.67 |
| 0.020 | 1.129600 | 2.00 | 4.00 |
| 0.030 | 1.136276 | 1.33 | 2.67 |

The initial knee-point candidate is 0.020. It retains a margin over the
three-seed keep baseline (RMSE 1.168695) while substantially reducing topology
churn. Penalized returns from different weights are not directly comparable,
because changing a weight changes the objective itself; selection must use the
accuracy/resource Pareto curve. More seeds and non-RANGE scenarios are required
before freezing a training reward.

### Dynamic visibility, resource recovery and dwell time

In the five-node 2406 m visibility stress test, a cost-aware policy added one
redundant edge when visible and removed it when its measurements disappeared at
decision 10. Compared with retaining that edge for all 30 decisions, removal
saved 19--20 transmitted messages and 19--20 replay operations per seed, with
negligible RMSE change.

A minimum dwell of two decisions reduced the two-step Oracle smoke-test switch
count from 5 to 4 and resynchronizations from 10 to 8. Final RMSE changed from
0.584136 to 0.594093 for seed 0. Emergency removal of an invisible redundant
edge remains legal during cooldown, while ordinary add/swap/remove actions are
masked. Dwell state is included in both graph metrics and the normalized policy
tensor.

Calibration records now include a canonical configuration ID containing the
environment parameters. Resume keys and summaries separate different scenario
configurations as well as policies, seeds, weights and Oracle horizons.

### Scalable action candidate selection

The full graph remains in `GraphObservation` and the GNN tensor. Only the set of
inactive edges eligible for add/swap enumeration is reduced. For every node,
the selector retains its Top-K currently visible and communication-available
inactive edges, ordered by estimated distance, packet loss, delay and stable
edge identity. Active edges and connectivity-safe remove actions are never
discarded by this selector. No truth or future visibility is used.

For a synthetic 20-node complete candidate graph with a 19-edge chain baseline,
the action counts were:

| Top-K per node | eligible additions | legal actions |
| ---: | ---: | ---: |
| unrestricted | 171 | 1483 |
| 1 | 18 | 55 |
| 2 | 22 | 71 |
| 3 | 39 | 143 |

Connectivity-breaking swaps are omitted from the catalog instead of emitted as
permanently masked actions. `K=2` is the initial Walker-scale candidate: it
retains more geometric alternatives than K=1 while using about 4.8% of the
unrestricted action count in this audit. Existing small-fleet baselines retain
the unrestricted default; Walker experiments must enable K explicitly.

### Walker 20/5/3 environment pre-scan

The V15 environment now reuses the existing Walker 20/5/3 truth generator and
RADAR/INFRARED/OPTICAL filter-case builder. The policy candidate topology is the
union of physically visible edges over the episode, while the initial active
topology is the first-epoch connected tree selected by the existing V14 rule.

Over 60 two-second decisions at 7000 km maximum measurement range, the physical
candidate union contained 30 edges and current visibility changed 28 -> 26 ->
28. With Top-K=2, the action catalog varied from 53 to 59 actions. A three-seed
closed-loop comparison produced:

| policy | final position RMSE | transmitted messages | switches | resyncs |
| --- | ---: | ---: | ---: | ---: |
| keep | 0.681402 | 1073.67 | 0.00 | 0.00 |
| cost-aware information | 0.672298 | 1101.67 | 4.33 | 4.67 |

The deterministic policy improved mean RMSE by about 1.34% for about 2.61%
more transmitted messages. Seed-level trade-offs varied substantially: RMSE
improvements were 0.56%, 2.44% and 1.00%, while message overheads were 5.22%,
2.33% and 0.28%. This variation is evidence for a state-dependent learned
decision policy, not evidence that every available topology change is useful.

The observed switches were concentrated near visibility transitions. Emergency
removals may immediately follow an add during dwell cooldown when a different
redundant active edge becomes invisible; this is intentional resource recovery,
not a cooldown violation.

### Walker snapshot counterfactual labels

A causal snapshot audit was added on top of the same persistent online
environment. The baseline policy advances the live environment to a selected
decision epoch; every legal action is then evaluated in a deep-copied branch
using the candidate action followed by keep for the remaining lookahead. Labels
contain RMSE reduction relative to keep plus messages, replay,
resynchronization and switch costs. The policy observation never receives these
future labels.

For Walker seed 0 at decision 23 with two-step lookahead and Top-K=2, 53 legal
actions were evaluated. Twenty-five improved RMSE relative to keep, 27 degraded
it and keep ranked 26th. The best action was a swap with RMSE reduction
0.006513. The deterministic information rule selected an add ranked 28th with
reduction -0.000226. This provides direct evidence that the present graph score
does not fully predict causal filter value and that learnable action-value
structure exists.

The reward-scale caveat is important: a switch penalty of 0.02 expressed in
physical RMSE units selected keep at this snapshot, while the deterministic
rule's churn weight of 1.0 acts on a differently normalized graph-information
score. These values must not be treated as interchangeable.

---

# V15 GraphObservation 特征审计（中文版）

## 1. 分类定义

- **D——部署可用**：决策时刻能够由估计器、传感器或通信系统获得；
- **S——仅仿真/评价可用**：可以用于奖励与验证，但禁止进入策略观测；
- **P——部分可用**：接口已有表达，但部分构建器缺失该量，或使用了仅仿真可用的替代来源；
- **M——缺失**：设计基线需要，但当前尚未表达。

## 2. 核心审计结论

当前 `GraphObservation` 已经覆盖状态、协方差、候选边、量测模态、通信可用性、丢包、时延、NIS、量测年龄和历史拓扑，足以支持 V14 的规则策略和反事实研究。但它还不是可以直接交给 V15 学习策略的部署观测契约，主要原因有三点：

1. **来源不透明**：相同的 `state` 或 `distance` 字段可能来自滤波估计，也可能来自仿真真值；
2. **在线资源状态不完整**：历史占用、重新同步、最后接收时间、连续失联、资源余量和可信新鲜度尚未系统进入图观测；
3. **张量化尚未部署化**：当前张量仍使用未归一化 ECI 状态、距离和协方差，并会丢弃部分 estimator/global metrics。

尤其需要注意：

- `short_horizon_topology_counterfactual` 的节点状态来自估计结果，但候选边距离由真值计算；它适合受控评价，不适合作为部署观测构建器；
- `v14_walker_dynamic_topology` 直接使用 Walker 真值构建规划图，它是几何/规划基线，不能未经适配直接连接学习策略；
- 未来策略输入中禁止出现真值、未来噪声、未来 RMSE、NEES 或反事实动作收益。

## 3. V15 必须补充的特征

节点级：

- 历史窗口占用、钉住历史数量和资源压力；
- 是否需要或正在等待重新同步；
- 窗口内发送、丢弃、拒绝消息率；
- 节点健康、通信预算或资源余量；
- 协方差来源、新鲜度和可信摘要。

边级：

- 与具体量测模态独立的相对速度/距离率；
- 预计剩余可见时间；
- 最后接收时间和连续失联时长；
- 丢包/时延估计是否可用及其样本数；
- 预计回放与重新同步代价；
- 链路资源占用和剩余容量；
- 拓扑版本与协方差来源可信度；
- 观测者与接收者不对称时的有向特征。

全局级：

- 链路预算、当前有效边数和候选边数；
- 平均/最大时延与信息年龄；
- 回放、重新同步和历史资源汇总；
- 近期拓扑切换率；
- 明确的观测模式与版本号。

## 4. 进入环境实现前的接口要求

1. 增加观测来源和模式描述，至少包括 `schema_version`、`state_source` 和 `geometry_source`；
2. 为 V15 策略路径定义带版本的节点、边和全局指标词表，同时保持旧接口兼容；
3. 对可选协方差、通信统计和估计量增加显式可用性掩码；
4. 只使用在线 orchestrator 状态补充资源、历史、重新同步和新鲜度特征；
5. 新建基于估计器的在线图观测适配器，禁止读取真值历史和未来观测；
6. 增加部署观测验证器，在来源属于仿真真值时拒绝交给策略或环境；
7. 新建归一化的 V15 张量模式，保留 V14 张量器作为回归与研究数据基线。

## 5. 推荐实施顺序

1. 先实现来源/模式元数据与验证测试；
2. 再实现当前可用字段的在线估计器/orchestrator适配器；
3. 补充缺失值掩码与归一化规范；
4. 导出资源、重新同步和新鲜度特征；
5. 增加 V15 张量化测试，包括节点置换、尺度和真值泄漏检查；
6. 最后实现 `TopologyControlEnvironment.reset/step`。

此次审计不冻结最终最小 GNN 特征集合。第一版环境应先保证信号可信、含义独立且可追溯，再通过消融实验决定学习策略真正需要哪些特征。
