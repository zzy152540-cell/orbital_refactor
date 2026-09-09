from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from brain_inspired.online_navigation_graph_features import (
    OnlineNavigationGraphFeatureConfig,
)
from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.ring_cann import RingCANNConfig
from cooperative.v15_policy_tensor import NODE_FEATURE_NAMES
from experiments.topology_control_environment import TopologyControlEnvironment


@dataclass(frozen=True)
class OnlineCANNRingNumericsResult:
    label: str
    num_neurons: int
    internal_dt: float
    runtime_seconds: float
    runtime_reduction_fraction: float
    base_filter_identical: bool
    base_tensor_prefix_identical: bool
    mean_cann_feature_difference: float
    maximum_cann_feature_difference: float


def run_online_cann_ring_numerics_scan(
    *, seed=0, node_count=3, episode_epochs=7, dt=2.0,
):
    settings = (
        ("reference_180_1ms", 180, 0.001),
        ("stable_180_2ms", 180, 0.002),
        ("compact_90_2ms", 90, 0.002),
        ("compact_60_2ms", 60, 0.002),
    )
    runs = []
    for label, neurons, internal_dt in settings:
        ring = replace(
            RingCANNConfig(), num_neurons=neurons,
            internal_dt=internal_dt,
        )
        config = OnlineNavigationGraphFeatureConfig(
            direction_config=OrbitalDirectionConfig(ring=ring),
        )
        runs.append((label, neurons, internal_dt, *_run(
            TopologyControlEnvironment(
                node_count=node_count, episode_epochs=episode_epochs,
                decision_interval_epochs=1, dt=dt,
                relative_modalities=("RANGE",), cann_policy_features=True,
                cann_feature_config=config,
            ),
            seed,
        )))

    _, _, _, reference_runtime, reference_states = runs[0]
    base_width = len(NODE_FEATURE_NAMES)
    results = []
    for label, neurons, internal_dt, runtime, states in runs:
        differences = np.concatenate([
            np.abs(
                reference.policy_tensor.node_features[:, base_width:]
                - candidate.policy_tensor.node_features[:, base_width:]
            ).reshape(-1)
            for reference, candidate in zip(reference_states, states)
        ])
        results.append(OnlineCANNRingNumericsResult(
            label=label, num_neurons=neurons, internal_dt=internal_dt,
            runtime_seconds=runtime,
            runtime_reduction_fraction=float(1.0 - runtime / reference_runtime),
            base_filter_identical=all(
                _same_filter(left, right)
                for left, right in zip(reference_states, states)
            ),
            base_tensor_prefix_identical=all(np.array_equal(
                left.policy_tensor.node_features[:, :base_width],
                right.policy_tensor.node_features[:, :base_width],
            ) for left, right in zip(reference_states, states)),
            mean_cann_feature_difference=float(np.mean(differences)),
            maximum_cann_feature_difference=float(np.max(differences)),
        ))
    return tuple(results)


def _run(environment, seed):
    started = perf_counter()
    states = [environment.reset(seed=seed)]
    while True:
        step = environment.step(0)
        states.append(step.state)
        if step.terminated or step.truncated:
            break
    return perf_counter() - started, states


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
    for item in run_online_cann_ring_numerics_scan():
        print(item)
