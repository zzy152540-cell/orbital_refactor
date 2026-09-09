from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from cooperative.v15_policy_tensor import NODE_FEATURE_NAMES
from experiments.topology_control_environment import TopologyControlEnvironment


@dataclass(frozen=True)
class OnlineCANNUpdateModeAblation:
    every_epoch_runtime_seconds: float
    decision_epoch_runtime_seconds: float
    every_epoch_reset_seconds: float
    decision_epoch_reset_seconds: float
    every_epoch_step_seconds: float
    decision_epoch_step_seconds: float
    runtime_reduction_fraction: float
    base_filter_identical: bool
    base_tensor_prefix_identical: bool
    mean_cann_feature_difference: float
    maximum_cann_feature_difference: float


def run_online_cann_update_mode_ablation(
    *, seed=0, node_count=3, episode_epochs=12,
    decision_interval_epochs=3, dt=2.0,
):
    common = dict(
        node_count=node_count, episode_epochs=episode_epochs,
        decision_interval_epochs=decision_interval_epochs, dt=dt,
        relative_modalities=("RANGE",), cann_policy_features=True,
    )
    every, every_runtime, every_reset, every_step = _run(TopologyControlEnvironment(
        **common, cann_update_mode="every_epoch",
    ), seed)
    decision, decision_runtime, decision_reset, decision_step = _run(TopologyControlEnvironment(
        **common, cann_update_mode="decision_epoch",
    ), seed)
    base_width = len(NODE_FEATURE_NAMES)
    differences = np.concatenate([
        np.abs(left.policy_tensor.node_features[:, base_width:]
               - right.policy_tensor.node_features[:, base_width:]).reshape(-1)
        for left, right in zip(every, decision)
    ])
    return OnlineCANNUpdateModeAblation(
        every_epoch_runtime_seconds=every_runtime,
        decision_epoch_runtime_seconds=decision_runtime,
        every_epoch_reset_seconds=every_reset,
        decision_epoch_reset_seconds=decision_reset,
        every_epoch_step_seconds=every_step,
        decision_epoch_step_seconds=decision_step,
        runtime_reduction_fraction=float(
            1.0 - decision_runtime / every_runtime
        ),
        base_filter_identical=all(_same_filter(left, right)
                                  for left, right in zip(every, decision)),
        base_tensor_prefix_identical=all(np.array_equal(
            left.policy_tensor.node_features[:, :base_width],
            right.policy_tensor.node_features[:, :base_width],
        ) for left, right in zip(every, decision)),
        mean_cann_feature_difference=float(np.mean(differences)),
        maximum_cann_feature_difference=float(np.max(differences)),
    )


def _run(environment, seed):
    started = perf_counter()
    states = [environment.reset(seed=seed)]
    after_reset = perf_counter()
    while True:
        step = environment.step(0)
        states.append(step.state)
        if step.terminated or step.truncated:
            break
    finished = perf_counter()
    return (
        states, finished - started, after_reset - started,
        finished - after_reset,
    )


def _same_filter(left, right):
    left_nodes = {node.node_id: node for node in left.observation.nodes}
    right_nodes = {node.node_id: node for node in right.observation.nodes}
    return set(left_nodes) == set(right_nodes) and all(
        left_nodes[node].state == right_nodes[node].state
        and left_nodes[node].covariance_diagonal
        == right_nodes[node].covariance_diagonal
        for node in left_nodes
    )


if __name__ == "__main__":
    print(run_online_cann_update_mode_ablation())
