# Orbital Refactor

`orbital-refactor` is a model-driven satellite state-estimation and
cooperative-estimation research framework. The completed V14 foundation targets
distributed, multi-modal, visibility-aware state estimation for satellite
swarms under communication delay, packet loss, link failure, and dynamic
topology. The V15 development line builds a safe graph-learning control layer
on top of that trusted estimator.

The project retains the original scripts in `legacy/` as numerical-regression
references while exposing reusable orbital dynamics, measurement, filtering,
communication, topology, and experiment components. Legacy EKF, covariance
intersection (CI), NIS gating, and export behavior is preserved where
regression compatibility is required.

Current package version: `0.4.0` (Python 3.10 or newer).

## V14 foundation and V15 transition

The architecture is organized into three layers:

```text
trusted distributed estimation
        -> deterministic cooperative topology policies
        -> GNN/RL topology and resource decisions
```

The filter remains responsible for physically meaningful state estimation.
GNNs are intended to encode dynamic inter-satellite relationships, and RL is a
future decision layer for selecting communication and observation resources;
neither is intended to replace orbital dynamics or the estimator.

### Implemented estimation foundation

- Two-body and J2 orbital propagation, RK4 integration, differential-orbit
  fleets, and Python-generated Walker constellations.
- Absolute ECI position navigation and inter-satellite `RANGE`, `RANGE_RATE`,
  `AZ_EL`, `RADAR`, `INFRARED`, and `OPTICAL` measurement semantics.
- ECI- and BODY-frame angular measurements, attitude histories, attitude
  uncertainty, Earth occultation, range/FOV visibility, and temporal
  visibility stabilization.
- Centralized Fleet EKF validation baselines and distributed local filters.
- Multi-neighbor Schmidt states containing one active local state, consider
  neighbor states, and their joint/cross covariance blocks.
- Observer-local relative-observation use by default; optional shared-delivery
  paths remain available for controlled comparisons.
- NIS integrity policies, soft covariance inflation, hard rejection, NEES,
  RMSE, covariance, and per-node/per-modality diagnostics.

CI is used only for multiple correlated estimates of the same physical state.
It must not be used to directly fuse the different physical states of distinct
satellites. The centralized Fleet EKF is a validation and small-scale baseline,
not the intended final distributed architecture.

### Covariance-safe communication and topology changes

The current distributed path supports:

- state/covariance messages with information provenance and lineage IDs;
- exact covariance transport and multi-neighbor Schmidt event replay;
- delay, packet loss, acknowledgements, and out-of-sequence events;
- bounded replay history, pinned checkpoints, event/resource limits, and
  explicit resynchronization after history exhaustion;
- topology versions, inactive links, link restoration, message rejection
  reasons, replay/resource peaks, and online topology changes.

These mechanisms establish covariance-consistent communication behavior. They
do not by themselves prove that every multi-modal measurement combination is
statistically consistent.

### Topology-policy and learning infrastructure

The repository now provides unified `GraphObservation`, `TopologyAction`, and
`TopologyPolicy` interfaces. A graph observation can describe node estimator
health and edge geometry, modality, visibility, communication availability,
packet loss, delay, information age, NIS, and resource state.

The current deterministic baseline is `LowChurnConnectedTreePolicy`. It is an
interpretable, low-switching connected-topology baseline, not a final optimized
policy.

The experimental learning infrastructure includes:

- legal `keep`, `add`, `remove`, and `swap` topology actions;
- fixed-prefix short-horizon counterfactual rollouts;
- conditional Monte Carlo future-noise branches;
- fleet- and node-level RMSE, covariance, NEES, NIS, communication, replay,
  and resynchronization targets;
- confidence intervals, positive/safe-positive gain probabilities, severe-loss
  probability, P10, and lower-tail metrics;
- graph tensorization, feature separability, label stability, action
  learnability, supervised GNN scoring, and safe-action classification.
- deployment-safe normalized V15 policy tensors, Top-K candidate pruning, and
  legal masked `keep`/`add`/`swap`/`remove` action spaces;
- a reproducible Walker-20 multi-step control environment with visibility,
  topology dwell, replay, resynchronization, communication and switch costs;
- restartable counterfactual snapshot collection, compressed pickle-free NPZ
  shards, schema-checked shard merging, and strict seed-disjoint splitting;
- PyTorch graph-action pretraining with regression/ranking, decision-oriented
  listwise, and hierarchical action-type/within-type objectives;
- an online hierarchical GNN policy that reads no truth or future labels and
  always maps its decision back through the current legal-action set.

The GNN remains a research prototype and RL initialization candidate, not a
production policy or an independent topology optimizer. Its current purpose is
to provide a safe, useful initial graph representation and actor policy that RL
can refine for long-horizon precision/resource trade-offs.

## Current validation findings

The V14 consistency path now separates source prediction and update events,
preserves event identity and covariance provenance, supports multi-neighbor
Schmidt replay, bounds and pins history, and explicitly resynchronizes after
history exhaustion. Online experiments with delay, packet loss and topology
changes report zero protocol rejection; delayed messages from obsolete topology
versions remain visible as a separate diagnostic class.

Five-node online counterfactual studies evaluate 27 legal/diagnostic local
actions under heterogeneous per-edge loss and delay. A single future trajectory
is not a stable action label: mean RMSE improvement can coexist with NEES
degradation or negative lower-tail gain. Robust opportunity analysis therefore
retains confidence bounds, safe-positive probability, consistency probability,
P10, lower-tail mean and severe-loss probability.

The adaptive Monte Carlo evaluator screens all actions with five future-noise
branches and extends only borderline actions to twenty while reusing the first
five rollouts. In the current two-prefix five-node study, 7 of 54 actions were
extended, neither group admitted a robust non-keep action, and the executed
action-branch count fell from 1080 to 405 (62.5%). These results justify a safe
`keep` fallback but do not establish that dynamic topology is generally
unnecessary.

Covariance reduction alone is still not accepted as valid information gain.
Future dense feedback must be qualified by observable consistency, provenance,
freshness and resynchronization diagnostics.

A three-seed Walker-20 deterministic comparison over 60 filter epochs contrasts
always-keep with the current cost-aware information-greedy policy under all
three relative modalities, 10% packet loss, and one-second delay. Information
greedy reduced mean final position RMSE from 0.68140 m to 0.67230 m (1.33%), but
added 28 transmitted messages and 26 replays on average, together with 4.33
topology switches and 4.67 resynchronizations. With the frozen switch penalty,
its penalized return was 0.07756 lower than keep. The policy is therefore a
useful deterministic accuracy/resource baseline rather than a universally
better controller. The paired four-panel presentation plot is stored as
`results/v15_walker20_deterministic_policy_overview.png`.

The current Walker-20 pilot dataset contains 48 online snapshots and 2544
counterfactual legal-action labels over seeds 0--7 and decision epochs
`0, 5, 10, 15, 20, 25`. Seeds 0--5 are used for training and 6--7 for strict
validation. The labels are balanced between positive and negative RMSE changes,
although the action set is dominated by `swap` combinations.

Flat action-value GNNs tended to collapse to a single action type. A
hierarchical PyTorch model that first chooses the action type and then ranks
actions within that type achieved positive mean held-out gain, 66.7% positive
selection rate and 83.3% action-type agreement. It exceeded always-keep and the
current shortest-added-edge diagnostic baseline, but still remained far below
the short-horizon oracle and did not reliably identify the exact best swap.

The hierarchical checkpoint was therefore evaluated as an RL initializer,
rather than tuned as a stand-alone oracle approximation. In ten-step online
Walker episodes on unseen seeds 8--10 it produced no illegal-action fallback or
filter divergence, selected both `keep` and `swap`, and changed final RMSE by
-0.48% on average relative to always-keep; the worst observed degradation was
0.31%. Each episode made four topology switches and incurred the associated
resynchronizations, which must be explicitly penalized during RL training.

## V15 development roadmap

The design baseline is documented in
[`docs/gnn_rl_design_baseline.md`](docs/gnn_rl_design_baseline.md), including a
complete Chinese version. The implementation-oriented current draft is
[`docs/gnn_rl_algorithm_baseline_v1_zh.md`](docs/gnn_rl_algorithm_baseline_v1_zh.md);
it freezes the proposed MPNN, hierarchical PPO, reward/cost and validation
framework before PPO code is started. V15 proceeds in this order:

1. Audit `GraphObservation` features as available, missing, simulation-only or
   deployment-available, and freeze normalization/missing-value semantics.
2. Implement a reproducible multi-step `TopologyControlEnvironment` around the
   online orchestrator with legal-action masks, decomposed reward, constraint
   costs and safety fallback.
3. Compare always-keep, random-legal, deterministic, information-greedy and
   counterfactual-oracle baselines; enrich scenarios if the oracle finds no
   meaningful control opportunity.
4. Calibrate consistency-qualified information gain and communication, delay,
   replay, resynchronization, switching and tail-risk constraints.
5. Use counterfactual supervision and, where useful, physically valid
   self-supervised tasks to obtain a GNN that meets RL-initialization criteria;
   do not require supervised exact-Oracle action matching.
6. Introduce masked, constrained and risk-sensitive RL from the qualified
   hierarchical checkpoint; include accuracy, communication, switching,
   replay and resynchronization terms, and retain deterministic safety gates
   plus a legal `keep` fallback.
7. Validate by held-out physical scenario, then move from five-node structured
   actions to hierarchical ten- and twenty-satellite Walker decisions.

The project has entered step 6. Hierarchical masked PPO and a frozen
LCB-reference advantage gate have been implemented and exercised in multi-step
closed-loop studies. Noise-augmented long-window supervision improves mean
gain and training-seed robustness, but the current gate has not established
condition-wise non-degradation: one unseen condition remained harmful for both
the reference and gated policies.

Robust snapshot datasets now preserve gain mean and standard deviation in
addition to their composite LCB. A separate offline moment model can apply
per-action emphasis and mean-sign calibration, but the first reference-action
safety study remained initialization-sensitive. It is therefore a diagnostic
research interface, not a deployed absolute `keep` gate. The operational
baseline remains the relative PPO-versus-LCB advantage gate plus legal `keep`
fallback; the next qualification step is broader condition-distribution
coverage without retuning on previously held-out seeds.

The compact five-node environment also provides an explicit opt-in
`undirected_independent` communication distribution. It samples packet loss
and delay per undirected candidate link while preserving the original
homogeneous distribution and all filter/GNN interfaces. A paired four-condition
pre-scan retained robust non-keep oracle opportunities but exposed a clear
generalization failure of the homogeneous LCB checkpoint. This distribution is
therefore the next training dimension; pre-scan conditions 76--79 are excluded
from subsequent fitting and formal evaluation.

The first geometry-randomized five-node curriculum now samples three physically
parameterized orbit families: compact along-track, differential along-track,
and a two-plane local cluster. Initial truth states are generated from valid
Keplerian elements, while condition and observation-noise seeds remain
separate. An eight-condition pre-scan found robust non-keep opportunities in
all three families; conditions 112--119 are reserved as development evidence
and excluded from training and formal confirmation.
The randomized-physical condition split is frozen in code: conditions 200--223
are training, 224--231 model selection, 232--239 development, and 240--247
one-time formal confirmation. Conditions 112--119 remain quarantined as
inspected pre-scan evidence.
The inspected pre-scan remains reproducible with random family assignment.
Formal fitting uses a separate seed-cycled configuration so the 24 training
conditions contain eight examples of each family and every eight-condition
evaluation block contains all three families without reward-based seed choice.
The frozen hierarchical GNN seed 0 was fitted only on conditions 200--223 and
selected on 224--231. In paired closed-loop selection runs it improved final
RMSE over always-keep by 0.01429 m on average (26/32 runs improved). Its first
use on development conditions 232--239 retained a 0.00767 m mean improvement
(20/32 improved), with no illegal fallback or divergence. The policy always
used one initial add followed by keep, so this passes only the PPO-initializer
gate; it is not evidence of a complete adaptive topology policy. Conditions
240--247 were subsequently consumed exactly once for formal confirmation.
A controlled PPO ablation preserved the supervised action-type head while
keeping the 96-episode budget, policy seed, rewards, and conditions unchanged.
Batch diagnostics showed PPO consistently increased the add-type probability
(ending near 0.579), while normalized add/swap advantages were positive and
keep advantages negative. The apparent all-keep result was instead caused by
deterministic decoding: global joint-probability argmax favored the singleton
keep action after add probability was divided among several candidate edges.
Hierarchical mode now selects the most probable type first and then its best
member; sampling and PPO log-probabilities are unchanged. Re-evaluating the same
saved weights gave the warm start a 0.00636 m development improvement (19/32
runs improved), whereas random initialization degraded by 0.02239 m on average
(5/32 improved, worst degradation 0.14876 m). GNN pretraining therefore improves
the current PPO initializer, although weak Critic explained variance and the
warm policy's fixed initial-add behavior remain open limitations.
The corrected-mode PPO comparison was then repeated for policy seeds 0, 1,
and 2 without changing data, rewards, or budget. All three preserved-head warm
starts improved over keep by 0.00636, 0.00621, and 0.00805 m respectively;
their pooled improvement was 0.00687 m with 59/96 episodes improved and worst
degradation 0.03266 m. Random initialization averaged -0.00786 m, improved only
20/96 episodes, and degraded by as much as 0.14876 m. This passes the frozen
multi-seed initializer gate. Configuration and all three seeds must now remain
fixed before the one-time 240--247 formal confirmation.
That frozen confirmation reproduced the development conclusion. Across policy
seeds 0, 1, and 2, warm-start PPO improved over keep by 0.00772, 0.00586, and
0.00636 m respectively. The pooled improvement was 0.00665 m, with 60/96
episodes improved, no illegal fallback, and worst degradation 0.02388 m.
Random initialization averaged -0.00435 m, improved 28/96 episodes, and
degraded by as much as 0.24337 m. Every warm-start seed retained one initial
add followed by keep. The formal result therefore confirms the GNN as a stable
PPO initializer, but does not establish a mature adaptive-topology policy.
Conditions 240--247 are now consumed and must not be used for tuning or another
confirmation claim.

The next, deliberately more aggressive curriculum keeps the frozen estimator
and policy interfaces but samples one shared graph Actor across 5, 10, and 20
nodes in a fixed 40/35/25 condition cycle. Episodes use 30 filter epochs, ten
decision points, Top-K=3 candidate pruning, and at most five topology changes.
Five-node cases retain the stratified randomized physical families; ten-node
cases initially use a physically connected Walker 10/1/0 ring, while twenty-
node cases use Walker 20/5/3. Link degradation windows are sampled only from
the physical candidate graph and navigation dropout remains time varying.
This curriculum is an opt-in V15 experiment and does not alter the consumed
five-node confirmation configuration.

A three-condition full-horizon smoke test confirmed that the same frozen GNN
Actor executes without schema changes or illegal fallback on all three graph
sizes. It selected add/keep actions only: five and twenty nodes exhausted the
five-switch budget and slightly degraded relative to keep, while ten nodes used
one add and improved by about 0.01578 m. These three episodes establish runtime
compatibility, not policy quality. They also confirm that the old initializer
must be retrained by the mixed-scale PPO curriculum before adaptive behavior is
claimed.

A bounded two-decision counterfactual prescan then audited conditions 320, 321,
322, 323, 324, and 326 at decision indices 0, 3, 6, and 9. At most two legal
members of each non-keep type were evaluated along the same always-keep state
trajectory, so this is a local opportunity audit rather than an exhaustive
Oracle. Positive non-keep opportunities appeared at 8/8 audited states for
both 5 and 10 nodes and 7/8 states for 20 nodes; after the initial decision the
counts were 6/6, 6/6, and 5/6. Mean best sampled gains were about 0.01452,
0.01190, and 0.01030 m respectively, with swap selected more often than add at
every scale. The frozen five-node reference GNN was positive at 6/8, 7/8, and
only 3/8 states, with mean 20-node gain -0.000875 m. The curriculum therefore
contains genuine later topology-control opportunities while exposing a clear
distribution shift for the old initializer. These conditions are development
prescan evidence and must not become a later formal-confirmation set.

The first shared mixed-scale PPO pilot used six training episodes from
conditions 400--405 and one update batch containing 60 transitions: 30 from
five nodes, 20 from ten nodes, and 10 from twenty nodes. Communication and
resynchronization penalties are normalized by fleet size (and communication
also by decision-window length), while RMSE reward and switch cost retain their
existing semantics. The update had nonzero gradient norm 1.5846, approximate KL
0.000317, and did not stop early. Sampled training actions included swap and
remove at every scale. On the first untouched development conditions 420--422,
the deterministic policy still used only add/keep but improved over keep by
0.01497, 0.00822, and 0.00278 m for 5, 10, and 20 nodes. This establishes an
end-to-end mixed-scale PPO training path, not statistical performance; the
three development conditions must not be used to tune a larger pilot.

The first complete 20-condition scale cycle then trained random and warm-start
PPO on the identical 440--459 schedule (8 five-node, 7 ten-node, and 5 twenty-
node episodes). On six untouched development conditions 460, 461, 462, 463,
464, and 466, random initialization improved 5/6 episodes by 0.01365 m on
average with worst degradation 0.01044 m. Warm start improved 4/6 by 0.02132 m
on average but degraded by as much as 0.02990 m: both five-node cases regressed,
while both twenty-node cases improved strongly. Both deterministic policies
still selected only add/keep, despite sampled training trajectories containing
swap/remove. Critic explained variance remained weak (about 0.014 random and
0.009 warm). Warm start therefore improves mean cross-scale performance but is
not yet the safer initializer. A single 20-episode batch provides only one PPO
update; the next experiment must use multiple rollout/update batches so changed
policies generate subsequent experience. These six conditions are now consumed
development diagnostics and cannot be reused for model selection.

Post-run architecture review found that the historical random comparator used
`explicit_action_pairing=False`, while the supervised warm-start checkpoint
used `True`. That 20-condition result is therefore exploratory and cannot be
interpreted as an initialization-only ablation. Variable-scale PPO now exposes
the pairing setting in its frozen configuration, defaults both branches to
`True`, and rejects a warm-start checkpoint whose Actor structure disagrees.
Regression tests require identical Actor parameter keys and tensor shapes for
random and warm initialization. All future multi-batch comparisons must use
this corrected structural baseline; the consumed 440--466 evidence will not be
rerun or relabeled.

With Actor structures aligned, a strict 60-episode, six-update comparison used
training conditions 500--559 and untouched evaluation conditions 580, 581,
582, 583, 584, and 586. Random initialization converged to deterministic keep
on all six evaluations, exactly matching the keep baseline. Warm start averaged
0.02081 m improvement but improved only 3/6 cases and degraded by as much as
0.01654 m. Its five-node cases improved strongly (mean 0.06989 m), while both
ten-node cases regressed and only one twenty-node case improved. The warm
deterministic policy still used add/keep only; no swap/remove survived into
deterministic evaluation. Critic explained variance deteriorated across the six
batches to about -0.11 for both branches. Multiple updates therefore remove the
old one-update limitation but do not solve cross-scale credit assignment. The
next work must diagnose scale-conditioned value targets and per-type advantages
before increasing the episode budget. Conditions 580--586 used here are now
consumed development evidence.

A frozen-Critic audit on new conditions 600, 601, 602, 603, 604, and 606
compared predictions with Critic-independent discounted Monte Carlo returns.
Across both branches, return standard deviation was about 0.67 while predicted
value standard deviation was only 0.06--0.07; explained variance was about
-0.12 and target/value correlation was negative. More importantly, mean return
fell from about 2.18 at decision 0 to -0.03 at decision 9, while random Critic
value rose from 0.267 to 0.416 and warm Critic value from 0.210 to 0.342. The
Critic has therefore learned the horizon direction backwards, rather than only
missing a fleet-scale offset. Variable-scale training now labels its artificial
window boundary as truncated and bootstraps the final value; legacy experiments
retain terminal boundaries by default. This changes the return construction,
not the environment physics or reward. A fresh controlled retraining is still
required before claiming that it improves Critic calibration. Conditions
600--606 are consumed diagnostic evidence.

A 30-episode controlled retraining on fresh conditions 620--649 tested that
change, with independent policy evaluation on 680--686 and a frozen-Critic
audit on 700--706. Training explained variance still declined to about -0.04.
The independent audit remained negative (about -0.104 random and -0.099 warm),
with target/value correlations of -0.62 and -0.69. Warm start improved 5/6
evaluation episodes by 0.00337 m on average, while random initialization again
collapsed to keep. Truncation is therefore the correct continuing-window
semantics but is not sufficient to identify the strongly transient filter
return. The next minimal ablation should expose normalized episode phase to the
Critic only, without changing the Actor, reward, or environment physics.

That Critic-only phase ablation was then run as a paired 30-episode comparison
on conditions 720--749. A zero-initialized one-dimensional phase projection
preserved every baseline Critic weight and produced identical first-batch
explained variance. It did not help thereafter: final explained variance was
-0.03482 versus -0.03541 for random initialization and -0.04899 versus -0.04983
for warm start (baseline versus phase). Deterministic policy results were also
identical. Since the Critic already observes `log1p_timestamp`, another linear
time representation is not the limiting factor. Further time-feature expansion
is deferred; the next diagnosis should separate filter-initialization transient
reward from topology-dependent return targets. Conditions 720--749 and 780--786
are consumed development evidence.

A same-state, same-noise one-step counterfactual reward audit on conditions
820--825 then separated always-keep filter improvement from topology-dependent
gain. Keep reward RMS was 0.5037 m, whereas all evaluated non-keep gains over
keep had RMS 0.02084 m, only 4.14% as large (2.56% for 10 nodes and 1.67% for
20 nodes). At decision zero, keep improvement averaged 1.485 m. The supervised
reference policy gained only 0.00186 m on average and was positive in 55% of
audited states, although the bounded oracle best non-keep action was positive
in 86.7%. The current absolute-RMSE reward is therefore dominated by common
filter initialization transients. The next controlled training experiment
should use selected-action reward minus a same-state keep counterfactual,
while retaining absolute task metrics for evaluation.

An exploratory 30-episode paired replay on the already-consumed 720--749
conditions then enabled that counterfactual reward. Warm-start Critic explained
variance changed from 0.023/-0.013/-0.049 across the three absolute-reward
batches to 0.115/0.435/0.338; mean deterministic RMSE improvement increased
from 0.00211 m to 0.00455 m. On fresh audit conditions 920--926, overall frozen
Critic explained variance was 0.726 random and 0.519 warm, with positive target
correlations of 0.855 and 0.722. However, within-scale explained variance
remained negative for 10 and 20 nodes (warm was positive only at 5 nodes).
Counterfactual reward is therefore a promising development option, not yet the
formal default: it removes the common filter transient and recovers aggregate
value structure, but within-scale action credit still needs confirmation.

Fresh paired confirmation on training conditions 940--969 and evaluation
conditions 1000--1006 preserved the warm-start policy benefit: mean RMSE
improvement rose from 0.00431 m with absolute reward to 0.00703 m with
counterfactual reward, mainly through the 20-node cases (0.00671 m to
0.01488 m). Training Critic explained variance ended at 0.225 instead of
-0.033. Yet a frozen audit on new conditions 1040--1046 did not confirm value
calibration: warm overall explained variance was -0.243 (absolute: -0.080),
although correlation improved from -0.616 to +0.211 and value RMSE fell from
0.813 to 0.0215 because the difference-return scale is much smaller. Every
within-scale explained variance remained negative. Counterfactual reward stays
optional; the next issue is centering/scaling small difference returns across
fleet scales, not another expansion of Actor capacity.

A fixed per-scale RMS normalization was tested next on the same 940--969
development conditions, using independently audited non-keep gain RMS values
of 0.02464/0.01542/0.00745 m for 5/10/20 nodes. Task gain and resource penalty
were scaled together. This naive normalization failed: warm final explained
variance fell from 0.225 unscaled to -0.0148, value loss rose from about
5e-4 to 1.48, and deterministic mean RMSE improvement fell from 0.00703 m to
0.00542 m. It is retained only as an optional negative ablation. Further work
should normalize Critic targets or calibrate values by scale without changing
the Actor's physical reward/cost balance.

Identity-initialized per-scale affine Critic heads were then evaluated without
changing rewards or Actor inputs. On the paired 940--969 training run they
improved warm final explained variance from 0.225 to 0.286 while leaving the
deterministic policy exactly unchanged. Frozen audits repeated the aggregate
calibration benefit: warm EV changed from -0.243 to -0.007 on 1040--1046 and
from 0.153 to 0.292 on fresh 1080--1086; correlations also improved from
0.211 to 0.358 and from 0.521 to 0.595. Within-scale EV remained negative and
mostly unchanged. The affine heads are therefore a useful optional cross-scale
Critic calibration mechanism, not a solution to within-scale action credit or
a source of demonstrated policy gain.

A held-out linear feature audit then tested the exact pooled state inputs used
by the Critic within each fleet scale. It used 15 disjoint training conditions
(1120--1134) and six test conditions (1180--1186), with ridge penalties from
0.1 to 100. Strong regularization recovered positive held-out structure at all
scales: best EV/correlation was 0.507/0.740 for 5 nodes (global features),
0.623/0.789 for 10 nodes (full features), and 0.383/0.627 for 20 nodes (full
features). Timestamp alone was ineffective for 5 and 20 nodes. Current state
features therefore contain useful within-scale return signal; the more likely
bottleneck is small-sample Critic fitting and regularization, not immediate
need for an action-conditioned Q-Critic or a larger observation schema.

A Critic-only Adam weight decay of 1e-3 was then paired against the calibrated
counterfactual model. It left Actor policy and deterministic RMSE unchanged.
On fresh 1200--1206 audit conditions, warm within-scale EV improved modestly
for all scales (5: -0.680 to -0.475, 10: -2.624 to -2.566, 20: 0.401 to
0.431) and RMSE fell from 0.02011 to 0.01858 overall, but aggregate EV fell
from 0.320 to 0.252. Weight decay remains optional: it slightly regularizes
within-scale fitting but does not solve it and weakens aggregate calibration.
The accompanying episode-aware plot is stored as
`results/v15_variable_scale_ppo_regularized_critic_training_overview.png`.

A longer single-seed development run then trained both aligned branches for
120 episodes in twelve ten-episode rollout/update batches on fresh conditions
1300--1419. Critic scale calibration, counterfactual keep reward, and Critic-only
weight decay of 1e-3 were retained. Random initialization again decoded to keep
for all six evaluation conditions 1450--1456. The warm-start branch improved
4/6 conditions, averaged 0.01841 m RMSE improvement, and had worst degradation
0.00516 m. Mean improvement by scale was 0.02807, 0.00063, and 0.01281 m for
5, 10, and 20 nodes. Final training explained variance reached 0.670 random and
0.814 warm, a substantial change from the earlier negative-Critic runs. The
warm deterministic policy nevertheless retained only add/keep actions; this is
single-policy-seed development evidence, not a multi-seed generalization or
condition-wise safety claim. The summary and six-panel plot are stored as
`results/v15_variable_scale_ppo_long120_wd1e3_conditions1300_1419.json` and
`results/v15_variable_scale_ppo_long120_training_overview.png`.

Long variable-scale runs now optionally save a complete checkpoint after every
rollout/update batch. The checkpoint contains the model, optimizer, policy and
minibatch generator state, global PyTorch RNG state, next episode, and all
completed diagnostics. Resume rejects a changed configuration. An interrupted
two-batch regression test produces exactly the same diagnostics and final model
parameters as uninterrupted training. The multibatch CLI exposes
`--checkpoint-directory`, `--resume-random`, and `--resume-warm` so either
branch can continue without repeating completed batches.

The frozen 120-episode comparison was repeated for policy seeds 0, 1, and 2
without changing training conditions 1300--1419, evaluation conditions
1450--1456, or any hyperparameter. Warm start averaged 0.01804 m improvement
with a between-seed standard deviation of 0.00169 m, improved 13/18 evaluation
episodes, and had worst degradation 0.01213 m. Mean improvement by scale was
0.03031, 0.00063, and 0.00834 m for 5, 10, and 20 nodes. Random initialization
averaged only 0.00253 m, improved 4/18 episodes, and degraded by as much as
0.01936 m. All three warm deterministic policies made exactly the same 34 keep
and 26 add decisions, with no swap or remove, so Actor behavior is seed-stable
but remains structurally narrow. Final warm Critic explained variance was
0.814, -0.001, and 0.037 across seeds; policy benefit therefore does not imply
stable value calibration. This passes a development multi-seed initializer
check, not a mature adaptive-topology or condition-wise safety criterion. The
machine-readable aggregate and figure are stored as
`results/v15_variable_scale_ppo_long120_multiseed_summary.json` and
`results/v15_variable_scale_ppo_long120_multiseed_overview.png`.

Variable-scale PPO summaries can be rendered with:

```bash
python -m experiments.run_v15_variable_scale_ppo_visualization \
  --summary results/v15_variable_scale_ppo_calibrated_counterfactual_conditions940_969.json \
  --output results/v15_variable_scale_ppo_calibrated_training_overview.png
```

The six-panel figure reports batch explained variance, PPO losses, approximate
KL, training return, scale-wise deterministic RMSE improvement, and evaluation
action counts. New multibatch summaries also include per-episode diagnostics;
older summaries remain supported through their scale-aggregated training data.
Plotting requires Matplotlib and can use the `state_estimate` environment even
when training uses `state_estimate_gnn`.

### V15 snapshot collection and GNN environment

Install the optional PyTorch dependency or use the `state_estimate_gnn` Conda
environment. Collect one restartable Walker shard with:

```bash
python -m experiments.run_v15_snapshot_collection \
  --seeds 0 --epochs 0 5 10 15 20 25 --lookahead 2 \
  --output results/v15_walker_snapshot_seed00.npz
```

The checked-in pilot dataset is
`results/v15_walker_snapshot_pilot_seed00_07.npz`. The corresponding qualified
initialization candidate is
`results/v15_walker_gnn_hierarchical_seed00_05_val06_07.pt`; its held-out and
closed-loop metrics are stored alongside it as JSON. PyTorch is optional for
the estimator and is only required for GNN training or inference.

## Repository layout

```text
orbital_core/   dynamics, measurement models, EKF/CI, integrity, and metrics
cooperative/    distributed filters, messages, replay, transport, and topology
scenarios/      orbit, fleet, attitude, visibility, and Walker truth generation
interfaces/     stable task, observation, state, and attitude data objects
pipelines/      reusable single-satellite and fleet pipelines
experiments/    V14 validation, counterfactual, Walker, topology, and GNN studies
tests/          regression, numerical, communication, topology, and study tests
legacy/         original numerical-reference scripts
```

## Main interfaces

Run a state-awareness task through the stable module interface:
 
```python
output = StateAwarenessModule().run(module_input)
```

Run a complete observation history with:

```python
output = StateAwarenessModule().run_history(module_input)
```

Select the single-satellite fusion architecture through the filter
configuration:

```python
config["filter"]["architecture"] = "centralized"    # Centralized EKF
# or
config["filter"]["architecture"] = "federated_ci"   # Federated EKF + CI (default)
```

Run the reusable multi-node cooperative pipeline with:

```python
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
```

Export a run to CSV, NPZ, and JSON with:

```python
from exporters import export_run_bundle

paths = export_run_bundle(
    history,
    module_output,
    output_directory="outputs/run_001",
    stem="centralized_result",
)
```

## Data flow

Precomputed neural-network predictions can be used without checkpoint files:

```text
SHIRT JSON + predictions.npz
        -> data-source adapters
        -> ModuleInput / Observation
        -> StateAwarenessModule
        -> ModuleOutput
```

Checkpoint files are required only by the separate neural-network inference
and training tools.

## Examples

Run examples from the project root:

```bash
python examples/run_standard_interface.py
python examples/run_multi_sat_scenario.py
python examples/run_multi_sat_interface.py
python examples/run_v13_1_baseline.py
python examples/run_v13_2_fleet_ci.py
python examples/run_v13_4_attitude_coupling.py
```

Run the v13.4 attitude-coupling Monte Carlo validation with:

```bash
python examples/run_v13_4_attitude_monte_carlo.py --seeds 30
```

Validate the independent MEKF covariance before tuning orbit/attitude
interface weights:

```bash
python examples/run_v13_4_attitude_consistency_monte_carlo.py --seeds 30
```

Run distributed Fleet-State CI validation across packet-loss and delay grids:

```bash
python examples/run_v13_4_distributed_monte_carlo.py --seeds 30 \
  --packet-loss-rates 0.0 0.2 --delays 0.0 4.0
```

Run the configurable noise and process Monte Carlo sweep with:

```bash
python examples/run_parameter_sweep.py --seeds 5 \
  --modes combined --angle-sigmas-deg 1.0 2.0 \
  --process-noises 1e-7
```

The sweep exports one raw CSV row per seed and algorithm, plus an aggregated
`*_summary.csv` containing means and standard deviations.

## Tests

Run the complete test suite with:

```bash
python -m pytest
```

---

# Version Change Log

## v13.4 - Attitude-aware BODY angle fusion

### Added

- Added BODY-frame inter-satellite azimuth/elevation prediction using explicit
  source-satellite `wxyz` inertial-to-body quaternions.
- Added BODY-angle Jacobians with respect to the orbital states and the MEKF
  left-multiplicative small-angle attitude error.
- Added propagation of attitude uncertainty into the angle covariance through
  `R_effective = R_sensor + J_attitude P_attitude J_attitude^T`.
- Connected epoch- and satellite-matched `AttitudeEstimate` objects to the
  centralized Fleet EKF and distributed Fleet-State CI paths.
- Added a reproducible comparison of truth attitude, MEKF attitude with
  covariance, and MEKF attitude without covariance.
- Added multi-seed Monte Carlo export with per-run metrics, aggregate
  statistics, percentiles, failure rates, and paired covariance-propagation
  win rates.
- Added distributed Fleet-State CI Monte Carlo validation across configurable
  packet-loss and communication-delay grids while keeping attitude as an
  external or independently estimated auxiliary input.
- Added Pre-CI and Post-CI state/covariance histories, per-epoch CI gain
  diagnostics, and a local-only control mode for separating measurement-update
  behavior from communication-fusion effects.
- Added independent attitude NEES, gyro NIS, and star-tracker NIS Monte Carlo
  diagnostics so MEKF covariance calibration can be assessed before any
  orbit/attitude interface scaling is introduced.

### Validation

- The 120-second comparison produces a mean MEKF attitude error of about
  `0.051 deg`.
- Truth-attitude BODY angles produce a mean two-dimensional angle NIS near the
  expected value of `2`.
- Propagating MEKF attitude covariance prevents the strongly overconfident NIS
  observed when the same attitude estimate is used with zero attitude
  covariance.

## v13.3 - Attitude MEKF migration

### Added

- Added `orbital_core/attitude.py` with explicit `wxyz` quaternion operations,
  rigid-body angular dynamics, RK4 angular-rate integration, and attitude-error
  utilities migrated from the original thesis implementation.
- Added `orbital_core/attitude_filter.py` with a nine-dimensional
  multiplicative error state:

  ```text
  [attitude error, angular-velocity error, gyro-bias error]
  ```

- Added gyro, star-tracker, and attitude-estimate interface objects.
- Added reproducible attitude truth, gyro white noise, gyro-bias random walk,
  and star-tracker measurement simulation.
- Added rigid-body angular-rate propagation, star-tracker quaternion updates,
  body-vector updates, NIS output, and Joseph covariance updates.
- Added quaternion-convention, sensor-reproducibility, MEKF-convergence,
  gyro-bias, and covariance-positive-semidefinite tests.

### Changed

- Extended the original six-dimensional attitude and angular-rate MEKF with an
  explicit three-dimensional gyro-bias state.
- Documented attitude estimation as an independent auxiliary subsystem. It is
  not yet coupled to the orbital angle-measurement update; a future integration
  can provide the estimated inertial-to-body DCM and attitude covariance.

### Validation

- Attitude migration tests: `3 passed`.
- Regression suite excluding two sandbox-incompatible temporary-directory test
  files: `72 passed`.

## v13.2 - Fleet-State CI and statistical validation

### Added

- Added a distributed validation path in which every satellite maintains a
  local copy of the same `6N` fleet state.
- Added validated `FleetStateMessage` exchange between nodes.
- Restricted CI to messages with identical state dimensions and satellite
  ordering.
- Added optional absolute ECI position anchors to remove the global drift mode
  of a relative-only constellation.
- Added symmetric fleet scenarios, a centralized `6N` EKF, distributed
  Fleet-State CI, RTN angle measurements, RMSE/NIS/NEES evaluation, and
  configurable Monte Carlo parameter sweeps.
- Added reproducible Gaussian white noise for inter-satellite range,
  range-rate, RTN azimuth/elevation, and low-rate GNSS position anchors.
- Matched generated covariance matrices to the configured measurement-noise
  standard deviations.

### Fixed

- Included neighbor-state covariance in distributed inter-satellite updates.
- Rejected direct CI between absolute states of different physical satellites
  in the primary fleet-filter path.

## v13.1 - Symmetric three-satellite baseline

### Added

- Added `scenarios/fleet_scenario.py` for symmetric N-satellite truth
  scenarios.
- Added `orbital_core/fleet_centralized_ekf.py` for the centralized `6N` state.
- Added `pipelines/fleet_centralized.py` for centralized inter-satellite EKF
  execution.
- Added per-satellite absolute orbit propagation, range/range-rate observation
  updates, fully connected topology support, and three-satellite acceptance
  examples.
- Included neighbor covariance in distributed inter-satellite updates through
  `H_j P_j H_j^T`.
- Rejected direct CI between different satellites' absolute states.

## v12.4 - Cooperative experiment framework

### Added

- Added a unified cooperative experiment entry script.
- Added JSON-based cooperative experiment configuration.
- Added cooperative result-summary export.

### Purpose

- Improved reproducibility and prepared delay, dropout, and packet-loss
  comparative experiments.

## v12.3 - Age-aware CI weighting

### Added

- Added information-age-based covariance inflation.
- Added optional age-aware CI weighting for asynchronous fusion.

### Changed

- Extended cooperative pipeline interfaces with:
  - `age_aware`
  - `age_penalty`

### Purpose

- Reduced the influence of older asynchronous information during CI fusion.

## v12.2 - Orbital-dynamics time alignment

### Added

- Replaced constant-velocity delayed-state alignment with two-body + J2 RK4
  propagation.
- Added an independent message-buffer lifecycle for each fusion run.

### Fixed

- Prevented communication-buffer state from leaking between repeated
  experiments.

## v12.1 - Asynchronous time-alignment fusion

### Added

- Added `cooperative/time_alignment.py`.
- Added delayed-state alignment before CI fusion.
- Added a time-propagation interface for future orbital-dynamics-based
  compensation.

### Changed

- Updated the asynchronous fusion pipeline so that:
  - `MessageBuffer` releases delayed reports;
  - released reports are propagated to the current fusion epoch;
  - aligned reports are used for CI fusion.
- Kept example entry scripts on the project-root path-insertion pattern.

### Notes

- This version used a constant-velocity transition model as the initial
  time-alignment baseline.

## v12 - Asynchronous fusion base

### Added

- Added message-buffer-based delayed communication processing.
- Added arrival-timestamp-based report release.

## v11.1 - Communication logging extension

### Added

- Added delay-configuration output to `run_multi_sat_delay.py`.
- Added received-node-history inspection and recording.
- Added source and arrival timestamp output for delay debugging.

### Notes

- This version did not yet include time-alignment compensation.

## v10 - Communication extension

### Added

- Added `cooperative/communication_channel.py`.
- Added stochastic packet-loss simulation.
- Extended the CI pipeline with `communication_channel`.
- Added `received_node_history`.
- Added `run_multi_sat_packet_loss.py`.

### Notes

- Node dropout remained supported through `CommunicationConfig`.

## v9 - Preliminary communication changes

### Existing in v8

- `NodeReport`.
- `active_node_history`.
- `validity_history_by_node`.

### Added

- Added `cooperative/communication.py`.
- Added `examples/run_multi_sat_node_dropout.py`.

### Notes

- The communication layer supported deterministic dropout.
- Packet loss, delay buffering, and asynchronous fusion were identified as
  subsequent extensions.

## v8 - Multi-satellite cooperative estimation

### Added

- Added an end-to-end path from generated orbital truth through local filtering
  to epoch-wise multi-node covariance intersection.
- Added absolute two-body + J2 orbit propagation.
- Added Keplerian-elements-to-ECI initialization.
- Added per-observer RTN/PRI frame histories.
- Added target and observer absolute and relative trajectory generation.
- Added synthetic optical, infrared, radar, and NN-surrogate observations.
- Added `StateAwarenessModule.run_history()` while preserving the stable
  `run()` API.
- Added multi-node history execution helpers.
- Added conversion from node-relative estimates to a common target absolute ECI
  state.
- Added simultaneous covariance-intersection fusion for one to three nodes.
- Added complete communication-outage handling by holding the last cooperative
  posterior after initialization.
- Added `node_validity_by_node` support for node and link failure simulation.

### Example workflow

The complete multi-satellite interface example:

1. Propagates one target and three observer satellites with the two-body + J2
   model.
2. Generates heterogeneous infrared, radar, and synthetic learning-enhanced
   observations.
3. Builds an independent `ModuleInput` and single-node filter for each
   observer.
4. Converts every local target-relative estimate into a target absolute ECI
   state.
5. Fuses available node reports by covariance intersection at each epoch.
6. Reports local and cooperative position/velocity RMSE and final node weights.

## v6 - Centralized fusion and unified result export

### Added

- Added a centralized multimodal EKF pipeline while preserving the external
  `ModuleInput` and `ModuleOutput` interfaces.
- Added `orbital_core/centralized_filter.py` with centralized prediction,
  per-modality NIS prechecks, and joint measurement updates.
- Added `pipelines/centralized.py` with the centralized execution loop,
  abnormal-event handling, and standard output packaging.
- Added `exporters/result_exporter.py` for unified CSV, NPZ, and JSON export.
- Extended `adapters/module_input_adapter.py` with centralized input adaptation
  through the existing standard interface.
- Added support for traditional optical, infrared, and radar observations, as
  well as learning-enhanced optical observations.

### Preserved behavior

- Preserved the legacy fixed finite-difference step, pseudoinverse, Joseph
  covariance update, and NIS soft/hard gating behavior.

### Output

- `centralized_result.csv`: per-epoch position, velocity, acceleration, NIS,
  gating, and CI weights when applicable.
- `centralized_result.npz`: complete state, covariance, and diagnostic history.
- `centralized_result.json`: standard `ModuleOutput` for upstream systems and
  visualization.

### Validation

- Regression and interface tests: `20 passed`.
- Centralized trajectories and covariances matched the legacy scripts, with
  only floating-point differences of approximately `1e-8` to `1e-7`.

## v5 - SHIRT and NN-prediction data adapters

### Added

- Added a data-source layer that converts legacy experiment files into the
  documented interface objects without changing the filter or CI
  implementation.
- Added `adapters/shirt_data_adapter.py` to:
  - load SHIRT `metadata.json` and `roe*.json`;
  - preserve the verified legacy quaternion and coordinate convention;
  - construct runtime auxiliary histories and a standard `ModuleInput`.
- Added `adapters/nn_prediction_adapter.py` to:
  - read `predictions.npz` through the existing `image_path` / `t_pred`
    contract;
  - align predictions with SHIRT filenames;
  - optionally build pseudo velocity;
  - emit learning-enhanced optical `Observation` objects.
- Added `adapters/synthetic_measurement_adapter.py` to:
  - create infrared azimuth/elevation and radar range/range-rate observations;
  - support explicit dropout windows;
  - emit standard traditional `Observation` objects.

### Validation

- Regression and adapter tests: `18 passed`.

## v4 - Standard input adaptation and learning-enhanced observations

### Added

- Added `adapters/module_input_adapter.py`.
- Added the unified `interfaces/state_awareness_module.py` entry point.
- Added automatic mapping of
  `Observation(modality="OPTICAL", source_type="LEARNING")` to `nn`.
- Added three-dimensional position observations in ECI and SPRI frames.
- Added six-dimensional position and pseudo-velocity observations in ECI and
  SPRI frames.
- Compared NN EKF numerical behavior with the original
  `federated_ci3_nn_ir_rad_fusion_ekf.py`.

### Implementation notes

- The adapter converts documented `ModuleInput` and `Observation` objects into
  arrays and local filters used by the federated CI pipeline.
- Primary-satellite state history and `q_eci2pri` history remain auxiliary
  model data in `config.runtime` to minimize algorithm changes.
- A modality uses a fixed measurement covariance within one task to maintain
  compatibility with the legacy implementation.
- Time-varying measurement covariance and confidence-based dynamic scaling were
  deferred as optional features that would not change default behavior.

## v3 - Federated CI pipeline and standard output

### Added

- Added `pipelines/federated_ci.py`.
- Organized traditional optical, infrared, radar, and other local EKFs into a
  unified federated pipeline.
- Preserved valid-posterior selection for CI, holding the previous fused result
  when every modality is missing, and optional feedback behavior.
- Added local states, fused states, covariance, NIS, gating flags, CI weights,
  and statistics to pipeline output.
- Converted modality loss, hard-gate rejection, and soft-gate down-weighting
  into `AbnormalEvent` objects.
- Added direct generation of `LocalEstimate`, `SingleFusionResult`, and
  `ModuleOutput`.

### Validation

- Regression tests directly called
  `legacy/federated_ci_dynamics_fusion_ekf.py` and compared fused states, fused
  covariances, and every local-filter result.

## v2 - Shared filters and CI utilities

### Added

- Added `orbital_core/filters.py` to unify single-modality and federated local
  orbital EKFs.
- Added `orbital_core/ci_fusion.py` to unify two-way and three-way covariance
  intersection and return named weights.
- Added `orbital_core/quality.py` with non-intrusive quality-scoring helpers.
- Added `pipelines/single_modal.py` to extract the single-modality execution
  loop and return history plus a standard `LocalEstimate`.
- Added regression tests that load the legacy `DynamicsEKF` directly and verify
  identical prediction and update results.

### Preserved behavior

- Preserved the fixed-step numerical Jacobian, pseudoinverse, Joseph covariance
  update, and NIS soft/hard gating behavior.
- Kept legacy scripts unchanged.

### Validation

- Test suite at this stage: `8 passed`.

## v1 - Initial engineering refactor

### Added

- Preserved the original scripts under `legacy/` as regression references.
- Extracted Earth constants, coordinate transformations, orbital dynamics,
  measurement models, and metric calculations.
- Added interface objects including `ModuleInput`, `Observation`,
  `LocalEstimate`, `SingleFusionResult`, `NodeReport`, and `ModuleOutput`.
- Added foundational unit tests.

### Preserved behavior

- Did not change the numerical logic of the original EKF, CI, NIS, or experiment
  workflows.
