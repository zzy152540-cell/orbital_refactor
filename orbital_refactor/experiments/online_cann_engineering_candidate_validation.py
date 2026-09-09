from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from time import perf_counter

import numpy as np

from brain_inspired.online_navigation_graph_features import (
    OnlineNavigationGraphFeatureConfig,
)
from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.ring_cann import RingCANNConfig
from cooperative.v15_policy_tensor import NODE_FEATURE_NAMES
from experiments.counterfactual_physical_scenarios import (
    FIVE_NODE_PHYSICAL_FAMILIES,
)
from experiments.topology_control_environment import (
    CompactFleetScenarioDistribution,
    TopologyControlEnvironment,
)


@dataclass(frozen=True)
class OnlineCANNEngineeringCaseResult:
    case_name: str
    node_count: int
    epoch_count: int
    reference_runtime_seconds: float
    candidate_runtime_seconds: float
    runtime_reduction_fraction: float
    base_filter_identical: bool
    base_tensor_prefix_identical: bool
    mean_cann_feature_difference: float
    maximum_cann_feature_difference: float


def run_online_cann_engineering_candidate_validation(
    *, seed=11, episode_epochs=30, dt=2.0, include_walker=True,
):
    degraded = CompactFleetScenarioDistribution(
        packet_loss_range=(0.1, 0.35),
        communication_delay_range=(0.0, 4.0),
        navigation_dropout_node_count=1,
        initial_topology_types=("chain", "ring", "star"),
        link_condition_mode="undirected_independent",
        dynamic_link_event_count=1,
    )
    diverse_five = replace(
        degraded,
        physical_scenario_families=tuple(FIVE_NODE_PHYSICAL_FAMILIES),
        physical_family_assignment_mode="seed_cycle",
    )
    cases = [
        ("three_nominal", dict(
            node_count=3, randomize_stage1_conditions=False,
            compact_scenario_distribution=CompactFleetScenarioDistribution(),
        )),
        ("three_degraded", dict(
            node_count=3, randomize_stage1_conditions=True,
            compact_scenario_distribution=degraded,
        )),
        ("five_nominal", dict(
            node_count=5, randomize_stage1_conditions=False,
            compact_scenario_distribution=CompactFleetScenarioDistribution(),
        )),
        ("five_diverse_degraded", dict(
            node_count=5, randomize_stage1_conditions=True,
            compact_scenario_distribution=diverse_five,
        )),
    ]
    if include_walker:
        cases.append(("walker_twenty", dict(
            node_count=20, scenario_type="walker_20_5_3",
            randomize_stage1_conditions=False,
        )))
    reference_config = _feature_config(180, 0.001)
    candidate_config = _feature_config(90, 0.002)
    results = []
    for offset, (name, case_options) in enumerate(cases):
        options = dict(
            episode_epochs=episode_epochs, decision_interval_epochs=1, dt=dt,
            relative_modalities=("RANGE",), cann_policy_features=True,
        )
        options.update(case_options)
        case_seed = int(seed + offset)
        reference_runtime, reference = _run(TopologyControlEnvironment(
            **options, cann_feature_config=reference_config,
        ), case_seed)
        candidate_runtime, candidate = _run(TopologyControlEnvironment(
            **options, cann_feature_config=candidate_config,
        ), case_seed)
        base_width = len(NODE_FEATURE_NAMES)
        differences = np.concatenate([
            np.abs(
                left.policy_tensor.node_features[:, base_width:]
                - right.policy_tensor.node_features[:, base_width:]
            ).reshape(-1)
            for left, right in zip(reference, candidate)
        ])
        results.append(OnlineCANNEngineeringCaseResult(
            case_name=name, node_count=int(case_options["node_count"]),
            epoch_count=episode_epochs,
            reference_runtime_seconds=reference_runtime,
            candidate_runtime_seconds=candidate_runtime,
            runtime_reduction_fraction=float(
                1.0 - candidate_runtime / reference_runtime
            ),
            base_filter_identical=all(
                _same_filter(left, right)
                for left, right in zip(reference, candidate)
            ),
            base_tensor_prefix_identical=all(np.array_equal(
                left.policy_tensor.node_features[:, :base_width],
                right.policy_tensor.node_features[:, :base_width],
            ) for left, right in zip(reference, candidate)),
            mean_cann_feature_difference=float(np.mean(differences)),
            maximum_cann_feature_difference=float(np.max(differences)),
        ))
    return tuple(results)


def _feature_config(neurons, internal_dt):
    ring = replace(
        RingCANNConfig(), num_neurons=int(neurons),
        internal_dt=float(internal_dt),
    )
    return OnlineNavigationGraphFeatureConfig(
        direction_config=OrbitalDirectionConfig(ring=ring),
    )


def _run(environment, seed):
    started = perf_counter()
    states = [environment.reset(seed=seed, condition_seed=seed)]
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
    for item in run_online_cann_engineering_candidate_validation():
        print(item)
