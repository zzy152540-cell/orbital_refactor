"""Distributed per-satellite Consensus-CI experiment.

This v13 example exercises the new fleet-shaped data flow:
each satellite owns its own state history, exchanges reports only with topology
neighbors, and receives a consensus-updated local history. It is intentionally
separate from the existing target-tracking cooperative CI baseline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.communication_channel import CommunicationChannel
from cooperative.consensus_runner import (
    DistributedConsensusHistory,
    run_distributed_consensus_history,
)
from cooperative.delay_channel import DelayChannel
from cooperative.topology import chain_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.metrics import compute_rmse
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate
from orbital_core.orbit_elements import keplerian_to_eci


@dataclass(frozen=True)
class DistributedConsensusDemoCase:
    timestamps: np.ndarray
    truth_state_history_by_node: dict[str, np.ndarray]
    local_state_history_by_node: dict[str, np.ndarray]
    local_covariance_history_by_node: dict[str, np.ndarray]


@dataclass(frozen=True)
class FleetRMSE:
    position_rmse_by_node: dict[str, float]
    velocity_rmse_by_node: dict[str, float]
    fleet_position_rmse: float
    fleet_velocity_rmse: float


def build_demo_case(seed: int = 7) -> DistributedConsensusDemoCase:
    timestamps = np.arange(0.0, 301.0, 1.0)
    altitude = 700e3
    inclination = np.deg2rad(23.0)
    base_state = keplerian_to_eci(
        R_EARTH + altitude,
        0.001,
        inclination,
        0.0,
        0.0,
        0.0,
    )
    initial_states = {
        "sat_01": base_state + np.array([0.0, -20.0, 0.0, 0.02, 0.00, 0.00]),
        "sat_02": base_state + np.array([35.0, 0.0, 10.0, 0.00, -0.02, 0.01]),
        "sat_03": base_state + np.array([-30.0, 25.0, -10.0, -0.01, 0.02, 0.00]),
    }
    truth = {
        node_id: propagate_absolute_orbit(initial_state, timestamps)
        for node_id, initial_state in initial_states.items()
    }

    rng = np.random.default_rng(seed)
    local_states: dict[str, np.ndarray] = {}
    local_covariances: dict[str, np.ndarray] = {}
    node_biases = {
        "sat_01": np.array([45.0, -20.0, 30.0, 0.04, -0.02, 0.01]),
        "sat_02": np.array([-35.0, 50.0, -20.0, -0.03, 0.03, -0.02]),
        "sat_03": np.array([25.0, 25.0, 45.0, 0.02, 0.01, 0.03]),
    }
    covariance_scales = {"sat_01": 140.0, "sat_02": 90.0, "sat_03": 180.0}
    for node_id, states in truth.items():
        noise = rng.normal(
            loc=0.0,
            scale=np.array([8.0, 8.0, 8.0, 0.01, 0.01, 0.01]),
            size=states.shape,
        )
        local_states[node_id] = states + node_biases[node_id] + noise
        position_sigma = covariance_scales[node_id]
        velocity_sigma = covariance_scales[node_id] / 500.0
        covariance = np.diag(
            [
                position_sigma,
                position_sigma,
                position_sigma,
                velocity_sigma,
                velocity_sigma,
                velocity_sigma,
            ]
        ) ** 2
        local_covariances[node_id] = np.tile(covariance, (len(timestamps), 1, 1))

    return DistributedConsensusDemoCase(
        timestamps=timestamps,
        truth_state_history_by_node=truth,
        local_state_history_by_node=local_states,
        local_covariance_history_by_node=local_covariances,
    )


def evaluate_fleet_rmse(
    *,
    truth_state_history_by_node: dict[str, np.ndarray],
    estimated_state_history_by_node: dict[str, np.ndarray],
) -> FleetRMSE:
    position_by_node: dict[str, float] = {}
    velocity_by_node: dict[str, float] = {}
    for node_id, truth in truth_state_history_by_node.items():
        estimate = estimated_state_history_by_node[node_id]
        position_by_node[node_id] = compute_rmse(estimate[:, :3] - truth[:, :3])
        velocity_by_node[node_id] = compute_rmse(estimate[:, 3:] - truth[:, 3:])
    return FleetRMSE(
        position_rmse_by_node=position_by_node,
        velocity_rmse_by_node=velocity_by_node,
        fleet_position_rmse=float(np.mean(list(position_by_node.values()))),
        fleet_velocity_rmse=float(np.mean(list(velocity_by_node.values()))),
    )


def run_demo(
    *,
    packet_loss: dict[str, float] | None = None,
    delay_by_node: dict[str, float] | None = None,
    use_range_measurements: bool = True,
    range_sigma: float = 5.0,
    use_range_rate_measurements: bool = True,
    range_rate_sigma: float = 2.0,
) -> tuple[DistributedConsensusDemoCase, DistributedConsensusHistory]:
    case = build_demo_case()
    channel = (
        None
        if packet_loss is None
        else CommunicationChannel(packet_loss_rate=packet_loss, random_seed=42)
    )
    delay = None if delay_by_node is None else DelayChannel(delay_by_node=delay_by_node)
    inter_satellite_observations = (
        _build_range_observations(case, range_sigma=range_sigma)
        if use_range_measurements
        else None
    )
    if use_range_rate_measurements:
        range_rate_observations = _build_range_rate_observations(
            case,
            range_rate_sigma=range_rate_sigma,
        )
        inter_satellite_observations = (
            range_rate_observations
            if inter_satellite_observations is None
            else [*inter_satellite_observations, *range_rate_observations]
        )
    history = run_distributed_consensus_history(
        timestamps=case.timestamps,
        state_history_by_node=case.local_state_history_by_node,
        covariance_history_by_node=case.local_covariance_history_by_node,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        communication_channel=channel,
        delay_channel=delay,
        grid_points=31,
        age_aware=delay is not None,
        age_penalty=1e-2,
        inter_satellite_observations=inter_satellite_observations,
        inter_satellite_gate_enable=True,
        inter_satellite_gate_threshold=9.21,
        inter_satellite_gate_mode="soft",
        inter_satellite_soft_scale=25.0,
    )
    return case, history


def _build_range_observations(
    case: DistributedConsensusDemoCase,
    *,
    range_sigma: float,
    seed: int = 17,
) -> list[InterSatelliteObservation]:
    topology = chain_topology(["sat_01", "sat_02", "sat_03"])
    rng = np.random.default_rng(seed)
    observations: list[InterSatelliteObservation] = []
    for node_id in topology.node_ids:
        for neighbor_id in topology.neighbors(node_id):
            for index, timestamp in enumerate(case.timestamps):
                measurement = measure_relative_range(
                    case.truth_state_history_by_node[node_id][index],
                    case.truth_state_history_by_node[neighbor_id][index],
                    noise=float(rng.normal(0.0, range_sigma)),
                )
                observations.append(
                    InterSatelliteObservation(
                        timestamp=float(timestamp),
                        source_node_id=node_id,
                        target_node_id=neighbor_id,
                        modality="RANGE",
                        measurement=np.array([measurement], dtype=float),
                        covariance=np.array([[range_sigma**2]], dtype=float),
                        confidence=1.0,
                        valid_flag=True,
                    )
                )
    return observations


def _build_range_rate_observations(
    case: DistributedConsensusDemoCase,
    *,
    range_rate_sigma: float,
    seed: int = 23,
) -> list[InterSatelliteObservation]:
    topology = chain_topology(["sat_01", "sat_02", "sat_03"])
    rng = np.random.default_rng(seed)
    observations: list[InterSatelliteObservation] = []
    for node_id in topology.node_ids:
        for neighbor_id in topology.neighbors(node_id):
            for index, timestamp in enumerate(case.timestamps):
                measurement = measure_relative_range_rate(
                    case.truth_state_history_by_node[node_id][index],
                    case.truth_state_history_by_node[neighbor_id][index],
                    noise=float(rng.normal(0.0, range_rate_sigma)),
                )
                observations.append(
                    InterSatelliteObservation(
                        timestamp=float(timestamp),
                        source_node_id=node_id,
                        target_node_id=neighbor_id,
                        modality="RANGE_RATE",
                        measurement=np.array([measurement], dtype=float),
                        covariance=np.array([[range_rate_sigma**2]], dtype=float),
                        confidence=1.0,
                        valid_flag=True,
                    )
                )
    return observations


def main() -> None:
    packet_loss = {"sat_01": 0.0, "sat_02": 0.1, "sat_03": 0.2}
    delay_by_node = {"sat_01": 0.0, "sat_02": 2.0, "sat_03": 4.0}
    case, range_consensus = run_demo(
        packet_loss=packet_loss,
        delay_by_node=delay_by_node,
        use_range_rate_measurements=False,
    )
    _, range_rate_consensus = run_demo(
        packet_loss=packet_loss,
        delay_by_node=delay_by_node,
        use_range_rate_measurements=True,
    )

    local_metrics = evaluate_fleet_rmse(
        truth_state_history_by_node=case.truth_state_history_by_node,
        estimated_state_history_by_node=case.local_state_history_by_node,
    )
    range_metrics = evaluate_fleet_rmse(
        truth_state_history_by_node=case.truth_state_history_by_node,
        estimated_state_history_by_node=range_consensus.state_history_by_node,
    )
    range_rate_metrics = evaluate_fleet_rmse(
        truth_state_history_by_node=case.truth_state_history_by_node,
        estimated_state_history_by_node=range_rate_consensus.state_history_by_node,
    )

    print("Distributed Consensus-CI v13 demo")
    print("=" * 40)
    print("Local independent estimates:")
    _print_metrics(local_metrics)
    print("\nRange + Consensus estimates:")
    _print_metrics(range_metrics)
    print("\nRange + Range-rate + Consensus estimates:")
    _print_metrics(range_rate_metrics)

    stats = range_rate_consensus.communication_stats
    print("\nCommunication stats:")
    print(f"  attempted reports: {stats.attempted_report_count}")
    print(f"  received reports:  {stats.received_report_count}")
    print(f"  dropped reports:   {stats.dropped_report_count}")
    print(f"  pending reports:   {stats.pending_report_count}")
    print(f"  packet loss rate:  {stats.packet_loss_rate:.3f}")
    print(f"  average delay:     {stats.average_delay:.3f} s")
    print("\nFinal received neighbors:")
    final_index = len(range_rate_consensus.timestamps) - 1
    for node_id in range_rate_consensus.node_ids:
        print(
            f"  {node_id}: {range_rate_consensus.received_reports_by_node[node_id][final_index]}"
        )


def _print_metrics(metrics: FleetRMSE) -> None:
    for node_id in sorted(metrics.position_rmse_by_node):
        print(
            f"  {node_id}: position={metrics.position_rmse_by_node[node_id]:.3f} m, "
            f"velocity={metrics.velocity_rmse_by_node[node_id]:.6f} m/s"
        )
    print(
        f"  fleet:  position={metrics.fleet_position_rmse:.3f} m, "
        f"velocity={metrics.fleet_velocity_rmse:.6f} m/s"
    )


if __name__ == "__main__":
    main()
