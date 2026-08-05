from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_orchestrator import (
    NetworkSchmidtOrchestrator,
    TransportSourceUpdate,
)
from experiments.v14_exact_transport_scale_scan import (
    _build_case,
    _topology_runtime_schedule,
    _validated_topology_inactive_windows,
)
from experiments.v14_three_satellite_local_observation import (
    _three_satellite_scenario,
)
from orbital_core.metrics import compute_nees_history, compute_rmse
from orbital_core.measurement_integrity import MeasurementIntegrityPolicy
from experiments.v14_observation_faults import apply_observation_faults
from scenarios.measurement_visibility import VisibilityConfig


@dataclass(frozen=True)
class OnlineTopologyResynchronizationSummary:
    run_count: int
    mean_position_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    minimum_joint_covariance_eigenvalue: float
    resynchronization_count: int
    rejected_message_count: int
    rejection_counts_by_reason: dict[str, int]
    transmitted_message_count: int
    dropped_message_count: int
    stale_topology_message_count: int
    protocol_rejected_message_count: int
    integrity_status_counts: dict[str, int]
    final_lineage_by_directed_link: dict[tuple[str, str], str]


def run_v14_online_topology_resynchronization_experiment(
    *, seeds: int = 5, duration: float = 120.0, dt: float = 2.0,
    inactive_edge: tuple[str, str] = ("sat_a", "sat_b"),
    inactive_window: tuple[float, float] = (20.0, 100.0),
    max_pinned_age: float = 20.0,
    topology_type: str = "ring", maximum_range: float = 5000.0,
    range_sigma: float = 2.0, range_rate_sigma: float = 0.05,
    az_el_sigma: float = np.deg2rad(0.05), optical_sigma: float = 1e-3,
    absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
    topology_inactive_windows_by_undirected_edge: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ] | None = None,
    packet_loss_rate: float = 0.0,
    communication_delay: float = 0.0,
    radar_actual_noise_scale: float = 1.0,
    infrared_outlier_bias: tuple[float, float] | None = None,
    infrared_outlier_window: tuple[float, float] | None = None,
    integrity_policy_by_modality: Mapping[
        str, MeasurementIntegrityPolicy
    ] | None = None,
) -> OnlineTopologyResynchronizationSummary:
    if radar_actual_noise_scale <= 0.0:
        raise ValueError("radar_actual_noise_scale must be positive.")
    if infrared_outlier_bias is not None and infrared_outlier_window is None:
        raise ValueError("infrared_outlier_bias requires a window.")
    metrics = []
    total_resync = 0
    total_rejected = 0
    total_transmitted = 0
    total_dropped = 0
    total_stale_topology = 0
    total_protocol_rejected = 0
    integrity_status_counts = {}
    rejection_counts = {}
    final_lineages = {}
    for seed in range(seeds):
        timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
        scenario = _three_satellite_scenario(timestamps)
        case = _build_case(
            seed=seed, duration=duration, dt=dt,
            range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
            az_el_sigma=az_el_sigma, optical_sigma=optical_sigma,
            absolute_sigma=absolute_sigma,
            process_noise_acceleration=process_noise_acceleration,
            packet_loss=0.0, delay=0.0, acknowledge_messages=True,
            node_count=3, topology_type=topology_type,
            truth_initial_state_by_node={
                node: history[0]
                for node, history in (
                    scenario.truth_state_history_by_node.items()
                )
            },
            visibility_by_modality={
                modality: VisibilityConfig(maximum_range=maximum_range)
                for modality in ("RADAR", "INFRARED", "OPTICAL")
            },
            relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
        )
        inactive = _validated_topology_inactive_windows(
            topology_inactive_windows_by_undirected_edge
            if topology_inactive_windows_by_undirected_edge is not None
            else {inactive_edge: (inactive_window,)},
            topology=case["topology"],
        )
        case["observations"] = apply_observation_faults(
            case["observations"], truth=case["truth"], seed=seed,
            radar_actual_noise_scale=radar_actual_noise_scale,
            infrared_outlier_bias=infrared_outlier_bias,
            infrared_outlier_window=infrared_outlier_window,
        )
        versions, active_neighbors = _topology_runtime_schedule(
            case["timestamps"], topology=case["topology"],
            inactive_windows=inactive,
        )
        source_updates = _source_updates_from_messages(
            case["transmitted_messages"], case["topology"].node_ids,
        )
        observations_by_time = _items_by_timestamp(case["observations"])
        absolute_by_time = _items_by_timestamp(case["absolute_observations"])
        orchestrator = NetworkSchmidtOrchestrator(
            initial_state_by_node=case["initial_states"],
            initial_covariance_by_node=case["initial_covariances"],
            topology=case["topology"], initial_timestamp=0.0,
            process_noise_acceleration=process_noise_acceleration,
            history_window=10.0,
            max_pinned_age=max_pinned_age,
            packet_loss_rate=packet_loss_rate,
            communication_delay=communication_delay,
            random_seed=20270000 + seed,
            integrity_policy_by_modality=integrity_policy_by_modality,
        )
        states = {
            node: np.zeros((len(case["timestamps"]), 6))
            for node in case["topology"].node_ids
        }
        covariances = {
            node: np.zeros((len(case["timestamps"]), 6, 6))
            for node in case["topology"].node_ids
        }
        minimum_eigenvalue = float("inf")
        for index, timestamp in enumerate(case["timestamps"]):
            timestamp = float(timestamp)
            result = orchestrator.step(
                timestamp, topology_version=versions[timestamp],
                active_neighbors_by_node=active_neighbors[timestamp],
                source_update_by_node={
                    node: source_updates[(node, timestamp)]
                    for node in case["topology"].node_ids
                },
                observations=observations_by_time.get(timestamp, ()),
                absolute_observations=absolute_by_time.get(timestamp, ()),
            )
            total_resync += len(result.resynchronized_links)
            total_rejected += result.rejected_message_count
            total_transmitted += result.transmitted_message_count
            total_dropped += result.dropped_message_count
            total_stale_topology += result.stale_topology_message_count
            total_protocol_rejected += (
                result.protocol_rejected_message_count
            )
            for reason, count in result.rejection_counts_by_reason.items():
                rejection_counts[reason] = rejection_counts.get(reason, 0) + count
            for node, step_result in result.result_by_node.items():
                states[node][index] = step_result.state.active_state
                covariances[node][index] = step_result.state.active_covariance
                minimum_eigenvalue = min(
                    minimum_eigenvalue,
                    float(np.linalg.eigvalsh(
                        step_result.state.joint_covariance
                    ).min()),
                )
        metrics.append(_metrics(states, covariances, case["truth"]))
        metrics[-1]["minimum_eigenvalue"] = minimum_eigenvalue
        final_lineages = {
            (receiver, source): lifecycle.lineage_id
            for receiver, session in orchestrator.sessions.items()
            for source, lifecycle in session.link_by_neighbor.items()
        }
        for session in orchestrator.sessions.values():
            for integrity in (
                session.coordinator.integrity_by_information_id.values()
            ):
                integrity_status_counts[integrity.status] = (
                    integrity_status_counts.get(integrity.status, 0) + 1
                )
    return OnlineTopologyResynchronizationSummary(
        run_count=seeds,
        mean_position_rmse=float(np.mean([m["position_rmse"] for m in metrics])),
        mean_nees=float(np.mean([m["nees"] for m in metrics])),
        mean_nees_95_coverage=float(np.mean([m["coverage"] for m in metrics])),
        minimum_joint_covariance_eigenvalue=float(min(
            m["minimum_eigenvalue"] for m in metrics
        )),
        resynchronization_count=total_resync,
        rejected_message_count=total_rejected,
        rejection_counts_by_reason=rejection_counts,
        transmitted_message_count=total_transmitted,
        dropped_message_count=total_dropped,
        stale_topology_message_count=total_stale_topology,
        protocol_rejected_message_count=total_protocol_rejected,
        integrity_status_counts=integrity_status_counts,
        final_lineage_by_directed_link=final_lineages,
    )


def _source_updates_from_messages(messages, node_ids):
    updates = {}
    for message in messages:
        source = str(message.source_node_id)
        for event in message.transport_events:
            key = (source, float(event.timestamp))
            updates.setdefault(key, TransportSourceUpdate(
                state=event.state_estimate,
                error_transition=message.error_transition,
                independent_process_noise=message.accumulated_process_noise,
                information_ids=event.information_ids,
                event_error_transition=event.error_transition,
                event_process_noise=event.independent_process_noise,
            ))
    missing_sources = set(node_ids) - {source for source, _ in updates}
    if missing_sources:
        raise RuntimeError("Source updates are unavailable for some nodes.")
    return updates


def _items_by_timestamp(items):
    result = {}
    for item in items:
        result.setdefault(float(item.timestamp), []).append(item)
    return result


def _metrics(states, covariances, truth):
    errors = []
    nees = []
    for node in truth:
        errors.append(states[node][:, :3] - truth[node][:, :3])
        nees.extend(compute_nees_history(
            states[node], truth[node], covariances[node]
        ))
    nees = np.asarray(nees)
    return {
        "position_rmse": compute_rmse(np.vstack(errors)),
        "nees": float(np.mean(nees)),
        "coverage": float(np.mean((nees >= 1.2373442458) & (nees <= 14.4493753354))),
    }
