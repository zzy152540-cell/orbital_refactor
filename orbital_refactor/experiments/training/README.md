# Training package

This package owns model definition and optimization code only.  Environment
physics, counterfactual label generation, diagnostics, and runnable experiment
entry points remain one level above in `experiments/`.

| Module | Responsibility |
|---|---|
| `graph_action_gnn.py` | Graph encoders, hierarchical action heads, supervised losses, and checkpoint loading |
| `topology_ppo.py` | Hierarchical Actor-Critic, rollout preparation, GAE, and PPO updates |
| `topology_ppo_stage0.py` | Minimal topology-control PPO integration stage |
| `topology_ppo_stage1.py` | Physical scenario PPO configuration, penalties, and evaluation helpers |
| `variable_scale_topology_ppo.py` | Shared 5/10/20-node PPO training and Walker curriculum integration |
| `variable_scale_critic_fitting.py` | Frozen-Actor Critic learnability fitting with disjoint validation and best-EV selection |
| `variable_scale_ppo_multiseed.py` | Frozen-configuration aggregation across policy seeds |

New code should import these modules through `experiments.training`.  The old
top-level module names are intentionally small compatibility shims so saved
commands and downstream callers continue to work during migration.
