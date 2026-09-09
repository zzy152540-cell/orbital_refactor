from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellConfig,
)
from brain_inspired.line_cann import LineCANNConfig
from brain_inspired.navigation_place_cells import NavigationPlaceCellConfig
from brain_inspired.online_navigation_graph_features import (
    OnlineNavigationGraphFeatureConfig,
)
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
from cooperative.v15_cann_policy_tensor import CANN_NODE_FEATURE_NAMES
from cooperative.v15_policy_tensor import NODE_FEATURE_NAMES
from experiments.topology_control_environment import TopologyControlEnvironment


@dataclass(frozen=True)
class OnlineCANNStressCalibrationResult:
    case_name: str
    epoch_count: int
    event_count_by_feature: tuple[tuple[str, int], ...]
    changing_feature_count: int
    base_filter_identical_to_nominal: bool


def run_online_cann_policy_stress_calibration(
    *, seed=31, node_count=3, episode_epochs=12, dt=2.0,
):
    cases = (
        ("nominal", OnlineNavigationGraphFeatureConfig()),
        ("narrow_rt_reject", _narrow_rt_config(rolling=False)),
        ("narrow_rt_rebase", _narrow_rt_config(rolling=True)),
        ("narrow_place", _narrow_place_config()),
    )
    runs = []
    for name, config in cases:
        states = _run(TopologyControlEnvironment(
            node_count=node_count, episode_epochs=episode_epochs, dt=dt,
            decision_interval_epochs=1, relative_modalities=("RANGE",),
            cann_policy_features=True, cann_feature_config=config,
        ), seed)
        runs.append((name, states))
    nominal = runs[0][1]
    base_width = len(NODE_FEATURE_NAMES)
    event_names = {
        "cann_boundary_saturated", "cann_anchor_rejected",
        "cann_reference_rebased", "place_boundary_saturated",
        "place_scale_transition",
    }
    results = []
    for name, states in runs:
        values = np.stack([
            state.policy_tensor.node_features[:, base_width:base_width + 12]
            for state in states
        ])
        counts = tuple(
            (feature, int(np.count_nonzero(values[:, :, index])))
            for index, feature in enumerate(CANN_NODE_FEATURE_NAMES)
            if feature in event_names
        )
        results.append(OnlineCANNStressCalibrationResult(
            case_name=name, epoch_count=len(states),
            event_count_by_feature=counts,
            changing_feature_count=int(sum(
                np.ptp(values[:, :, index]) > 0.0
                for index in range(values.shape[2])
            )),
            base_filter_identical_to_nominal=all(
                _same_filter(left, right)
                for left, right in zip(nominal, states)
            ),
        ))
    return tuple(results)


def _narrow_rt_config(*, rolling):
    axis = LineCANNConfig(
        num_neurons=41, minimum_value=-1.0, maximum_value=1.0,
        tuning_width=0.1,
    )
    return OnlineNavigationGraphFeatureConfig(rt_config=OrbitalRTGridConfig(
        radial=axis, along_track=axis,
        maximum_anchor_innovation=(1.0e6 if rolling else 0.01),
        rolling_reference_enabled=rolling,
        rolling_reference_trigger_fraction=0.5,
    ))


def _narrow_place_config():
    coarse = NavigationPlaceCellConfig(
        radial_centers_m=(-1.0, 0.0, 1.0),
        along_track_centers_m=(-1.0, 0.0, 1.0),
        radial_sigma_m=0.25, along_track_sigma_m=0.25,
    )
    fine = NavigationPlaceCellConfig(
        radial_centers_m=(-0.1, 0.0, 0.1),
        along_track_centers_m=(-0.1, 0.0, 0.1),
        radial_sigma_m=0.025, along_track_sigma_m=0.025,
    )
    return OnlineNavigationGraphFeatureConfig(
        place_config=HierarchicalNavigationPlaceCellConfig(
            coarse=coarse, fine=fine,
        )
    )


def _run(environment, seed):
    states = [environment.reset(seed=seed)]
    while True:
        step = environment.step(0)
        states.append(step.state)
        if step.terminated or step.truncated:
            return states


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
    for item in run_online_cann_policy_stress_calibration():
        print(item)
