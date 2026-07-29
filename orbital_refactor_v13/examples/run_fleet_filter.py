"""Closed-loop per-satellite fleet filter example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.fleet_filter_runner import run_fleet_filter
from cooperative.topology import chain_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.metrics import compute_rmse
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.orbit_elements import keplerian_to_eci


def build_case():
    timestamps = np.arange(0.0, 121.0, 2.0)
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0)
    initials = {
        "sat_01": base + np.array([0.0, -20.0, 0.0, 0.02, 0.00, 0.00]),
        "sat_02": base + np.array([35.0, 0.0, 10.0, 0.00, -0.02, 0.01]),
        "sat_03": base + np.array([-30.0, 25.0, -10.0, -0.01, 0.02, 0.00]),
    }
    truth = {
        node_id: propagate_absolute_orbit(initial_state, timestamps)
        for node_id, initial_state in initials.items()
    }
    initial_errors = {
        "sat_01": np.array([45.0, -20.0, 30.0, 0.04, -0.02, 0.01]),
        "sat_02": np.array([-35.0, 50.0, -20.0, -0.03, 0.03, -0.02]),
        "sat_03": np.array([25.0, 25.0, 45.0, 0.02, 0.01, 0.03]),
    }
    estimates0 = {
        node_id: truth[node_id][0] + initial_errors[node_id]
        for node_id in truth
    }
    covariances0 = {
        node_id: np.diag([120.0, 120.0, 120.0, 0.2, 0.2, 0.2]) ** 2
        for node_id in truth
    }
    return timestamps, truth, estimates0, covariances0


def build_inter_satellite_observations(
    *,
    timestamps,
    truth,
    range_sigma,
    range_rate_sigma,
    angle_sigma,
    include_range_rate,
    include_az_el,
    az_el_frame,
):
    topology = chain_topology(["sat_01", "sat_02", "sat_03"])
    rng = np.random.default_rng(20260723)
    observations = []
    for source in topology.node_ids:
        for target in topology.neighbors(source):
            for index, timestamp in enumerate(timestamps):
                observations.append(
                    InterSatelliteObservation(
                        timestamp=float(timestamp),
                        source_node_id=source,
                        target_node_id=target,
                        modality="RANGE",
                        measurement=np.array([
                            measure_relative_range(
                                truth[source][index],
                                truth[target][index],
                                noise=float(rng.normal(0.0, range_sigma)),
                            )
                        ]),
                        covariance=np.array([[range_sigma**2]]),
                        confidence=1.0,
                        valid_flag=True,
                    )
                )
                if include_az_el:
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="AZ_EL",
                            measurement=measure_relative_az_el(
                                truth[source][index],
                                truth[target][index],
                                frame=az_el_frame,
                                noise=rng.normal(0.0, angle_sigma, size=2),
                            ),
                            covariance=np.diag([angle_sigma, angle_sigma]) ** 2,
                            confidence=1.0,
                            valid_flag=True,
                        )
                    )
                if include_range_rate:
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="RANGE_RATE",
                            measurement=np.array([
                                measure_relative_range_rate(
                                    truth[source][index],
                                    truth[target][index],
                                    noise=float(rng.normal(0.0, range_rate_sigma)),
                                )
                            ]),
                            covariance=np.array([[range_rate_sigma**2]]),
                            confidence=1.0,
                            valid_flag=True,
                        )
                    )
    return observations


def main():
    timestamps, truth, initial_state, initial_covariance = build_case()
    baseline = {
        node_id: propagate_absolute_orbit(initial_state[node_id], timestamps)
        for node_id in initial_state
    }
    range_only = run_filter_case(
        timestamps=timestamps,
        truth=truth,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        include_range_rate=False,
    )
    range_rate = run_filter_case(
        timestamps=timestamps,
        truth=truth,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        include_range_rate=True,
    )

    print("Closed-loop fleet filter v13 demo")
    print("=" * 40)
    print("Propagated initial estimates:")
    print_metrics(truth, baseline)

    print("\nFleet filter estimates: RANGE only")
    print_metrics(truth, range_only.state_history_by_node)
    print_gate_stats(range_only)

    print("\nFleet filter estimates: RANGE + RANGE_RATE")
    print_metrics(truth, range_rate.state_history_by_node)
    print_gate_stats(range_rate)

    print("\nCommunication stats:")
    print_communication_stats(range_rate)


def run_filter_case(
    *,
    timestamps,
    truth,
    initial_state,
    initial_covariance,
    include_range_rate,
):
    observations = build_inter_satellite_observations(
        timestamps=timestamps,
        truth=truth,
        range_sigma=5.0,
        range_rate_sigma=0.02,
        angle_sigma=np.deg2rad(0.02),
        include_range_rate=include_range_rate,
        include_az_el=False,
        az_el_frame="RTN",
    )
    return run_fleet_filter(
        timestamps=timestamps,
        initial_state_by_node=initial_state,
        initial_covariance_by_node=initial_covariance,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        inter_satellite_observations=observations,
        process_noise_acceleration_std=1e-8,
        inter_satellite_gate_enable=True,
        inter_satellite_gate_threshold=9.21,
        inter_satellite_gate_mode="soft",
        inter_satellite_soft_scale=25.0,
        enable_state_consensus=False,
        inter_satellite_frame_by_modality={"AZ_EL": "RTN"},
        consensus_iterations=1,
        ci_grid_points=31,
    )


def print_metrics(truth, estimate):
    position = {}
    velocity = {}
    for node_id in sorted(truth):
        position[node_id] = compute_rmse(estimate[node_id][:, :3] - truth[node_id][:, :3])
        velocity[node_id] = compute_rmse(estimate[node_id][:, 3:] - truth[node_id][:, 3:])
        print(
            f"  {node_id}: position={position[node_id]:.3f} m, "
            f"velocity={velocity[node_id]:.6f} m/s"
        )
    print(
        f"  fleet:  position={np.mean(list(position.values())):.3f} m, "
        f"velocity={np.mean(list(velocity.values())):.6f} m/s"
    )


def print_gate_stats(result):
    total = 0
    gated = 0
    for history in result.inter_satellite_gate_history_by_node.values():
        for per_epoch in history:
            for key, value in per_epoch.items():
                if key.endswith(":BLOCK"):
                    total += 1
                    gated += int(value)
    print(f"  gated observation blocks: {gated}/{total}")


def print_communication_stats(result):
    stats = result.communication_stats
    print(f"  attempted reports: {stats.attempted_report_count}")
    print(f"  received reports:  {stats.received_report_count}")
    print(f"  dropped reports:   {stats.dropped_report_count}")
    print(f"  pending reports:   {stats.pending_report_count}")


if __name__ == "__main__":
    main()
