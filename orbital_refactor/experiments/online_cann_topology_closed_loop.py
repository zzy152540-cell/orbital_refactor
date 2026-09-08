from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.v15_policy_tensor import NODE_FEATURE_NAMES
from experiments.topology_control_environment import TopologyControlEnvironment


@dataclass(frozen=True)
class OnlineCANNTopologyClosedLoopResult:
    epoch_count: int
    base_filter_identical: bool
    base_tensor_prefix_identical: bool
    rewards_identical: bool
    costs_identical: bool
    all_actions_legal: bool
    cann_features_changed: bool
    disabled_runtime_seconds: float
    enabled_runtime_seconds: float


def run_online_cann_topology_closed_loop(
    *, seed=0, node_count=3, episode_epochs=6, dt=2.0,
):
    """Paired keep-policy run proving the optional CANN path is side-effect free."""
    common = dict(
        node_count=node_count, episode_epochs=episode_epochs, dt=dt,
        decision_interval_epochs=1, relative_modalities=("RANGE",),
    )
    disabled, disabled_runtime = _run(
        TopologyControlEnvironment(**common, cann_policy_features=False), seed,
    )
    enabled, enabled_runtime = _run(
        TopologyControlEnvironment(**common, cann_policy_features=True), seed,
    )
    base_width = len(NODE_FEATURE_NAMES)
    return OnlineCANNTopologyClosedLoopResult(
        epoch_count=len(disabled),
        base_filter_identical=all(
            _same_observation(left[0].observation, right[0].observation)
            for left, right in zip(disabled, enabled)
        ),
        base_tensor_prefix_identical=all(np.array_equal(
            left[0].policy_tensor.node_features,
            right[0].policy_tensor.node_features[:, :base_width],
        ) for left, right in zip(disabled, enabled)),
        rewards_identical=all(left[1] == right[1]
                              for left, right in zip(disabled, enabled)),
        costs_identical=all(left[2] == right[2]
                            for left, right in zip(disabled, enabled)),
        all_actions_legal=all(left[3] and right[3]
                              for left, right in zip(disabled, enabled)),
        cann_features_changed=any(not np.array_equal(
            enabled[0][0].policy_tensor.node_features[:, base_width:],
            item[0].policy_tensor.node_features[:, base_width:],
        ) for item in enabled[1:]),
        disabled_runtime_seconds=disabled_runtime,
        enabled_runtime_seconds=enabled_runtime,
    )


def _run(environment, seed):
    started = perf_counter()
    state = environment.reset(seed=seed)
    records = [(state, None, None, True)]
    while True:
        step = environment.step(0)
        records.append((
            step.state, step.reward_terms, step.constraint_costs,
            not step.action_resolution.used_fallback,
        ))
        if step.terminated or step.truncated:
            break
    return records, perf_counter() - started


def _same_observation(left, right):
    left_nodes = {node.node_id: node for node in left.nodes}
    right_nodes = {node.node_id: node for node in right.nodes}
    if set(left_nodes) != set(right_nodes):
        return False
    return all(
        left_nodes[node].state == right_nodes[node].state
        and left_nodes[node].covariance_diagonal
        == right_nodes[node].covariance_diagonal
        for node in left_nodes
    )


if __name__ == "__main__":
    print(run_online_cann_topology_closed_loop())
