from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import csv
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.message_transport import MessageChannel
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology, ring_topology, star_topology
from interfaces.data_objects import (
    AbsolutePositionObservation,
    ObservationMessage,
    StateMessage,
)
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_optical_uv,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.inter_satellite_model import body_angle_effective_covariance
from orbital_core.metrics import compute_nees_history, compute_rmse
from orbital_core.measurement_semantics import inter_satellite_semantic_metadata
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import generate_fleet_scenario
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    VisibilityTemporalFilterConfig,
    generate_inter_satellite_observation_opportunities,
    summarize_observation_opportunities,
    stabilize_observation_opportunities,
)

Array = np.ndarray
NEES_95_DOF6 = (1.2373442458, 14.4493753354)
NIS_95_DOF1 = (0.0009820691, 5.0238861873)
NIS_95_DOF2 = (0.0506356159, 7.3777589082)


@dataclass(frozen=True)
class ExactTransportScanSummary:
    node_count: int
    topology_type: str
    scenario: str
    mode: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_nis: float
    mean_nis_95_coverage: float
    mean_nis_by_modality: dict[str, float]
    mean_nis_95_coverage_by_modality: dict[str, float]
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int
    minimum_joint_eigenvalue: float
    transmitted_state_messages: int
    rejection_counts: dict[str, int]
    mean_run_seconds: float
    total_replay_seconds: float
    replay_count: int
    batch_count: int
    maximum_replay_seconds: float
    mean_replay_span: float
    maximum_replay_span: float
    maximum_batch_size: int
    replayed_remote_events: int
    replayed_observations: int
    fallback_count: int
    maximum_remote_event_count: int
    maximum_observation_count: int
    maximum_checkpoint_count: int
    maximum_posterior_state_count: int
    maximum_pinned_checkpoint_count: int
    maximum_resync_required_count: int
    maximum_retained_journal_count: int


@dataclass(frozen=True)
class ExactTransportScaleScanResult:
    summary_by_scenario_and_mode: dict[tuple[str, str], ExactTransportScanSummary]
    diagnostic_records: tuple[dict[str, object], ...] = ()
    visibility_summary: VisibilityOpportunitySummary | None = None


@dataclass(frozen=True)
class ExactTransportTopologyScanResult:
    result_by_topology: dict[str, ExactTransportScaleScanResult]


def run_v14_exact_transport_smoke_scan(
    *, seeds: int = 20, duration: float = 60.0, dt: float = 2.0,
    range_sigma: float = 2.0, range_rate_sigma: float = 0.02,
    radar_correlation: float = 0.0,
    az_el_sigma: float = np.deg2rad(0.05), optical_sigma: float = 1e-3,
    absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
    node_count: int = 3, topology_type: str = "chain",
    scenario_names: tuple[str, ...] | None = None,
    modes: tuple[str, ...] = ("propagate_only", "exact_transport_event_replay"),
    visibility_by_modality: dict[str, VisibilityConfig] | None = None,
    relative_modalities: tuple[str, ...] = ("RANGE",),
    communication_outage_windows_by_directed_link: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ] | None = None,
    measurement_period_by_modality: Mapping[str, float] | None = None,
    visibility_temporal_filter_by_modality: Mapping[
        str, VisibilityTemporalFilterConfig
    ] | None = None,
) -> ExactTransportScaleScanResult:
    """Run the production network API over five configurable fleet cases."""
    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    if node_count < 2:
        raise ValueError("node_count must be at least two.")
    supported_relative_modalities = {
        "RADAR", "INFRARED", "OPTICAL", "RANGE", "RANGE_RATE", "AZ_EL",
    }
    if (
        not relative_modalities
        or len(set(relative_modalities)) != len(relative_modalities)
        or set(relative_modalities) - supported_relative_modalities
    ):
        raise ValueError(
            "relative_modalities contains an unsupported or duplicate modality."
        )
    if "RADAR" in relative_modalities and set(relative_modalities) & {
        "RANGE", "RANGE_RATE",
    }:
        raise ValueError("RADAR cannot be combined with legacy RANGE components.")
    if "INFRARED" in relative_modalities and "AZ_EL" in relative_modalities:
        raise ValueError("INFRARED cannot be combined with legacy AZ_EL.")
    if not -1.0 < radar_correlation < 1.0:
        raise ValueError("radar_correlation must be strictly between -1 and 1.")
    all_scenarios = {
        "ideal": (0.0, 0.0, 10.0),
        "loss_20_percent": (0.2, 0.0, 10.0),
        "delay_one_epoch": (0.0, dt, 10.0),
        "delay_loss": (0.2, dt, 10.0),
        "insufficient_history": (0.0, 3.0 * dt, 2.0 * dt),
    }
    selected_scenarios = tuple(all_scenarios) if scenario_names is None else scenario_names
    if not selected_scenarios:
        raise ValueError("At least one scenario must be selected.")
    if len(set(selected_scenarios)) != len(selected_scenarios):
        raise ValueError("Scenario names must be unique.")
    unknown_scenarios = set(selected_scenarios) - set(all_scenarios)
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario names: {sorted(unknown_scenarios)}")
    supported_modes = {"propagate_only", "exact_transport_event_replay"}
    if len(set(modes)) != len(modes):
        raise ValueError("Modes must be unique.")
    unknown_modes = set(modes) - supported_modes
    if unknown_modes or not modes:
        raise ValueError(f"Unsupported or empty modes: {sorted(unknown_modes)}")
    scenarios = {name: all_scenarios[name] for name in selected_scenarios}
    collected = {(scenario, mode): [] for scenario in scenarios for mode in modes}
    diagnostic_records = []
    visibility_summary = None
    for seed in range(seeds):
        for scenario, (loss, delay, history_window) in scenarios.items():
            case = _build_case(
                seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
                range_rate_sigma=range_rate_sigma, az_el_sigma=az_el_sigma,
                radar_correlation=radar_correlation,
                optical_sigma=optical_sigma,
                absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=loss, delay=delay,
                acknowledge_messages=delay <= history_window,
                node_count=node_count, topology_type=topology_type,
                visibility_by_modality=visibility_by_modality,
                relative_modalities=relative_modalities,
                communication_outage_windows_by_directed_link=(
                    communication_outage_windows_by_directed_link
                ),
                measurement_period_by_modality=measurement_period_by_modality,
                visibility_temporal_filter_by_modality=(
                    visibility_temporal_filter_by_modality
                ),
            )
            if visibility_summary is None:
                visibility_summary = case["visibility_summary"]
            for mode in modes:
                started = perf_counter()
                history = run_network_schmidt_filter(
                    timestamps=case["timestamps"],
                    initial_state_by_node=case["initial_states"],
                    initial_covariance_by_node=case["initial_covariances"],
                    topology=case["topology"],
                    observation_messages=case["observations"],
                    absolute_position_observations=case["absolute_observations"],
                    observation_usage="observer_only",
                    process_noise_acceleration=process_noise_acceleration,
                    consider_refresh_mode=mode,
                    state_messages_by_receiver=(case["state_messages"] if mode == "exact_transport_event_replay" else None),
                    replay_history_window=(history_window if mode == "exact_transport_event_replay" else None),
                    expected_lineage_by_link=(case["lineages"] if mode == "exact_transport_event_replay" else None),
                )
                run_seconds = perf_counter() - started
                collected[(scenario, mode)].append(
                    _metrics(
                        history, case["truth"], len(case["transmitted_messages"]),
                        run_seconds,
                    )
                )
                for record in history.refresh_diagnostic_records:
                    diagnostic_records.append({
                        "seed": seed, "node_count": node_count,
                        "topology_type": topology_type,
                        "scenario": scenario, "mode": mode, **record
                    })
    summaries = {}
    for key, values in collected.items():
        scenario, mode = key
        transmitted = sum(value[8] for value in values)
        accepted = sum(value[7] for value in values)
        rejected = sum(value[9] for value in values)
        processed = accepted + rejected
        summaries[key] = ExactTransportScanSummary(
            node_count=node_count, topology_type=topology_type,
            scenario=scenario, mode=mode, run_count=len(values),
            mean_position_rmse=float(np.mean([value[0] for value in values])),
            mean_velocity_rmse=float(np.mean([value[1] for value in values])),
            mean_nees=float(np.mean([value[2] for value in values])),
            mean_nees_95_coverage=float(np.mean([value[3] for value in values])),
            mean_nis=float(np.mean([value[4] for value in values])),
            mean_nis_95_coverage=float(np.mean([value[5] for value in values])),
            mean_nis_by_modality=_mean_metric_dict([value[14] for value in values]),
            mean_nis_95_coverage_by_modality=_mean_metric_dict(
                [value[15] for value in values]
            ),
            message_acceptance_rate=(accepted / processed if processed else 0.0),
            message_rejection_count=rejected,
            psd_failure_count=sum(value[10] for value in values),
            minimum_joint_eigenvalue=min(value[6] for value in values),
            transmitted_state_messages=transmitted,
            rejection_counts=_sum_rejection_counts([value[11] for value in values]),
            mean_run_seconds=float(np.mean([value[12] for value in values])),
            total_replay_seconds=sum(value[13]["total_replay_seconds"] for value in values),
            replay_count=sum(value[13]["replay_count"] for value in values),
            batch_count=sum(value[13]["batch_count"] for value in values),
            maximum_replay_seconds=max(value[13]["maximum_replay_seconds"] for value in values),
            mean_replay_span=(
                sum(value[13]["total_replay_span"] for value in values)
                / sum(value[13]["replay_count"] for value in values)
                if sum(value[13]["replay_count"] for value in values) else 0.0
            ),
            maximum_replay_span=max(value[13]["maximum_replay_span"] for value in values),
            maximum_batch_size=max(value[13]["maximum_batch_size"] for value in values),
            replayed_remote_events=sum(value[13]["replayed_remote_events"] for value in values),
            replayed_observations=sum(value[13]["replayed_observations"] for value in values),
            fallback_count=sum(value[13]["fallback_count"] for value in values),
            maximum_remote_event_count=max(value[13]["maximum_remote_event_count"] for value in values),
            maximum_observation_count=max(value[13]["maximum_observation_count"] for value in values),
            maximum_checkpoint_count=max(value[13]["maximum_checkpoint_count"] for value in values),
            maximum_posterior_state_count=max(value[13]["maximum_posterior_state_count"] for value in values),
            maximum_pinned_checkpoint_count=max(value[13]["maximum_pinned_checkpoint_count"] for value in values),
            maximum_resync_required_count=max(value[13]["maximum_resync_required_count"] for value in values),
            maximum_retained_journal_count=max(value[13]["maximum_retained_journal_count"] for value in values),
        )
    return ExactTransportScaleScanResult(
        summaries, tuple(diagnostic_records), visibility_summary
    )


def run_v14_exact_transport_topology_scan(
    *, node_count: int = 5, topology_types: tuple[str, ...] = ("chain", "ring", "star"),
    seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
    range_sigma: float = 2.0, range_rate_sigma: float = 0.02,
    radar_correlation: float = 0.0,
    az_el_sigma: float = np.deg2rad(0.05), optical_sigma: float = 1e-3,
    absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
    scenario_names: tuple[str, ...] | None = None,
    modes: tuple[str, ...] = ("propagate_only", "exact_transport_event_replay"),
    visibility_by_modality: dict[str, VisibilityConfig] | None = None,
    relative_modalities: tuple[str, ...] = ("RANGE",),
    communication_outage_windows_by_directed_link: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ] | None = None,
    measurement_period_by_modality: Mapping[str, float] | None = None,
    visibility_temporal_filter_by_modality: Mapping[
        str, VisibilityTemporalFilterConfig
    ] | None = None,
) -> ExactTransportTopologyScanResult:
    results = {}
    for topology_type in topology_types:
        results[topology_type] = run_v14_exact_transport_smoke_scan(
            seeds=seeds, duration=duration, dt=dt,
            range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
            radar_correlation=radar_correlation,
            az_el_sigma=az_el_sigma,
            optical_sigma=optical_sigma,
            absolute_sigma=absolute_sigma,
            process_noise_acceleration=process_noise_acceleration,
            node_count=node_count, topology_type=topology_type,
            scenario_names=scenario_names, modes=modes,
            visibility_by_modality=visibility_by_modality,
            relative_modalities=relative_modalities,
            communication_outage_windows_by_directed_link=(
                communication_outage_windows_by_directed_link
            ),
            measurement_period_by_modality=measurement_period_by_modality,
            visibility_temporal_filter_by_modality=(
                visibility_temporal_filter_by_modality
            ),
        )
    return ExactTransportTopologyScanResult(results)


def _build_case(*, seed, duration, dt, range_sigma, absolute_sigma,
                process_noise_acceleration, packet_loss, delay, acknowledge_messages,
                node_count, topology_type, visibility_by_modality=None,
                truth_initial_state_by_node=None, range_rate_sigma=0.02,
                radar_correlation=0.0,
                az_el_sigma=np.deg2rad(0.05),
                optical_sigma=1e-3,
                az_el_frame="ECI", attitude_history_by_node=None,
                estimated_attitude_history_by_node=None,
                attitude_covariance=None,
                visibility_temporal_filter_by_modality=None,
                communication_outage_windows_by_directed_link=None,
                absolute_navigation_dropout_windows=(),
                measurement_period_by_modality=None,
                relative_modalities=("RANGE",)):
    rng = np.random.default_rng(20260830 + seed)
    range_rate_rng = np.random.default_rng(20260930 + seed)
    az_el_rng = np.random.default_rng(20261030 + seed)
    optical_rng = np.random.default_rng(20261130 + seed)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0)
    center = 0.5 * (node_count - 1)
    truth_initials = (
        {
            f"sat_{index + 1:02d}": base + np.array([
                1200.0 * (index - center),
                100.0 * np.sin(2.0 * np.pi * index / node_count),
                30.0 * np.cos(2.0 * np.pi * index / node_count),
                0.0, 0.02 * (index - center), 0.0,
            ])
            for index in range(node_count)
        }
        if truth_initial_state_by_node is None
        else {
            str(node_id): np.asarray(state, dtype=float).reshape(6).copy()
            for node_id, state in truth_initial_state_by_node.items()
        }
    )
    if len(truth_initials) != node_count:
        raise ValueError("Truth initial-state count must match node_count.")
    scenario = generate_fleet_scenario(timestamps=timestamps, initial_state_by_node=truth_initials)
    truth = scenario.truth_state_history_by_node
    az_el_frame = str(az_el_frame).upper()
    if az_el_frame not in {"ECI", "BODY"}:
        raise ValueError("az_el_frame must be 'ECI' or 'BODY'.")
    if az_el_frame == "BODY":
        if attitude_history_by_node is None:
            raise ValueError("BODY AZ_EL requires attitude_history_by_node.")
        if set(attitude_history_by_node) != set(scenario.node_ids):
            raise ValueError("Attitude-history keys must match scenario nodes.")
        attitude_history_by_node = {
            node: np.asarray(values, dtype=float).reshape(len(timestamps), 4)
            for node, values in attitude_history_by_node.items()
        }
        if estimated_attitude_history_by_node is None:
            estimated_attitude_history_by_node = attitude_history_by_node
        if set(estimated_attitude_history_by_node) != set(scenario.node_ids):
            raise ValueError("Estimated-attitude keys must match scenario nodes.")
        estimated_attitude_history_by_node = {
            node: np.asarray(values, dtype=float).reshape(len(timestamps), 4)
            for node, values in estimated_attitude_history_by_node.items()
        }
        if attitude_covariance is not None:
            attitude_covariance = np.asarray(attitude_covariance, dtype=float).reshape(3, 3)
    covariance = np.diag([10, 10, 10, 0.02, 0.02, 0.02]) ** 2
    initial_states = {node: truth_initials[node] + rng.multivariate_normal(np.zeros(6), covariance) for node in scenario.node_ids}
    initial_covariances = {node: covariance.copy() for node in scenario.node_ids}
    topology_builders = {
        "chain": chain_topology, "ring": ring_topology, "star": star_topology,
    }
    if topology_type not in topology_builders:
        raise ValueError("topology_type must be 'chain', 'ring', or 'star'.")
    topology = topology_builders[topology_type](list(scenario.node_ids))
    visibility_summary = None
    visible_range_opportunities = None
    if visibility_by_modality is not None:
        if set(visibility_by_modality) != set(relative_modalities):
            raise ValueError(
                "Visibility configurations must match enabled relative modalities."
            )
        opportunities = generate_inter_satellite_observation_opportunities(
            timestamps=timestamps,
            truth_state_history_by_node=truth,
            candidate_topology=topology,
            visibility_by_modality=visibility_by_modality,
            attitude_history_by_node=(
                attitude_history_by_node if az_el_frame == "BODY" else None
            ),
        )
        if visibility_temporal_filter_by_modality is not None:
            opportunities = stabilize_observation_opportunities(
                opportunities,
                visibility_by_modality=visibility_by_modality,
                temporal_filter_by_modality=(
                    visibility_temporal_filter_by_modality
                ),
            )
        visibility_summary = summarize_observation_opportunities(opportunities)
        visible_range_opportunities = {
            (item.timestamp, item.observer_id, item.target_id, item.modality)
            for item in opportunities if item.visibility.visible
        }
    edges = tuple((receiver, source) for receiver in topology.node_ids for source in topology.neighbors(receiver))
    communication_outages = _validated_communication_outages(
        communication_outage_windows_by_directed_link, edges=edges,
    )
    absolute_navigation_dropout_windows = tuple(
        (float(start), float(end))
        for start, end in absolute_navigation_dropout_windows
    )
    if any(
        not np.isfinite(start) or not np.isfinite(end) or end < start
        for start, end in absolute_navigation_dropout_windows
    ):
        raise ValueError(
            "Absolute-navigation dropout windows require finite start <= end."
        )
    measurement_periods = _validated_measurement_periods(
        measurement_period_by_modality, modalities=relative_modalities,
    )
    sender_state = {node: value.copy() for node, value in initial_states.items()}
    sender_covariance = {node: covariance.copy() for node in scenario.node_ids}
    accumulators = {
        edge: ExactTransportAccumulator(
            source_node_id=edge[1], lineage_id=f"{edge[1]}->{edge[0]}:0",
            reference_timestamp=0.0, reference_state=sender_state[edge[1]],
            reference_covariance=sender_covariance[edge[1]],
        ) for edge in edges
    }
    channels = {
        edge: MessageChannel(
            packet_loss_rate={edge[1]: packet_loss}, delay_by_source={edge[1]: delay},
            random_seed=seed * 101 + index,
        ) for index, edge in enumerate(edges)
    }
    state_messages = {node: [] for node in topology.node_ids}
    transmitted_messages: list[StateMessage] = []
    pending_acks = []
    consecutive_losses = {edge: 0 for edge in edges}
    link_sequence = {edge: 0 for edge in edges}
    observations = []
    absolute_observations = []
    h = np.zeros((3, 6)); h[:, :3] = np.eye(3)
    absolute_covariance = np.eye(3) * absolute_sigma**2
    for index, timestamp in enumerate(timestamps):
        for arrival, edge, message in sorted(pending_acks, key=lambda item: item[0]):
            if arrival <= timestamp and acknowledge_messages:
                accumulators[edge].acknowledge(message)
        pending_acks = [item for item in pending_acks if item[0] > timestamp or not acknowledge_messages]
        prediction_transition = {node: np.eye(6) for node in topology.node_ids}
        prediction_noise = {node: np.zeros((6, 6)) for node in topology.node_ids}
        if index > 0:
            for node in topology.node_ids:
                transition = numerical_jacobian_discrete(lambda value: rk4_step_absolute(value, dt), sender_state[node])
                sender_state[node] = rk4_step_absolute(sender_state[node], dt)
                noise = make_process_noise(dt, process_noise_acceleration)
                sender_covariance[node] = transition @ sender_covariance[node] @ transition.T + noise
                prediction_transition[node] = transition; prediction_noise[node] = noise
        for node in topology.node_ids:
            navigation_available = not any(
                start <= float(timestamp) <= end
                for start, end in absolute_navigation_dropout_windows
            )
            update_transition = np.eye(6)
            update_noise = np.zeros((6, 6))
            information_ids = ()
            if navigation_available:
                innovation_covariance = h @ sender_covariance[node] @ h.T + absolute_covariance
                gain = sender_covariance[node] @ h.T @ np.linalg.inv(innovation_covariance)
                update_transition = np.eye(6) - gain @ h
                update_noise = gain @ absolute_covariance @ gain.T
                measurement = truth[node][index, :3] + rng.normal(0.0, absolute_sigma, 3)
                information_id = f"{node}:absolute:{index}"
                information_ids = (information_id,)
                absolute_observations.append(AbsolutePositionObservation(
                    timestamp=float(timestamp), satellite_id=node,
                    measurement_eci=measurement.copy(),
                    covariance=absolute_covariance.copy(), confidence=1.0,
                    valid_flag=True, observation_id=information_id,
                ))
                sender_state[node] += gain @ (measurement - h @ sender_state[node])
                sender_covariance[node] = update_transition @ sender_covariance[node] @ update_transition.T + update_noise
            for receiver, source in edges:
                if source != node:
                    continue
                combined_transition = update_transition @ prediction_transition[node]
                combined_noise = update_transition @ prediction_noise[node] @ update_transition.T + update_noise
                accumulator = accumulators[(receiver, source)]
                accumulator.append(
                    timestamp=float(timestamp), updated_state=sender_state[node],
                    error_transition=combined_transition,
                    independent_process_noise=combined_noise,
                    information_ids=information_ids,
                    event_error_transition=update_transition,
                    event_process_noise=update_noise,
                )
                message = accumulator.build_message()
                link_sequence[(receiver, source)] += 1
                transmitted = (
                    None
                    if _link_is_in_outage(
                        communication_outages,
                        receiver=receiver,
                        source=source,
                        timestamp=float(timestamp),
                    )
                    else channels[(receiver, source)].transmit(message)
                )
                if transmitted is not None:
                    ack_eligible = bool(acknowledge_messages)
                    transmitted = replace(
                        transmitted,
                        metadata={
                            "link_sequence": link_sequence[(receiver, source)],
                            "consecutive_losses_before_delivery": consecutive_losses[(receiver, source)],
                            "reference_event_count": len(message.transport_events),
                            "ack_eligible": ack_eligible,
                        },
                    )
                    consecutive_losses[(receiver, source)] = 0
                    state_messages[receiver].append(transmitted)
                    transmitted_messages.append(transmitted)
                    if ack_eligible:
                        pending_acks.append((float(transmitted.arrival_timestamp), (receiver, source), message))
                else:
                    consecutive_losses[(receiver, source)] += 1
        for observer in topology.node_ids:
            for target in topology.neighbors(observer):
                range_noise = rng.normal(0.0, range_sigma)
                if "RADAR" in relative_modalities and _measurement_is_due(
                    "RADAR", float(timestamp), measurement_periods
                ) and (
                    visible_range_opportunities is None
                    or (float(timestamp), observer, target, "RADAR")
                    in visible_range_opportunities
                ):
                    radar_covariance = np.array([
                        [
                            range_sigma**2,
                            radar_correlation * range_sigma * range_rate_sigma,
                        ],
                        [
                            radar_correlation * range_sigma * range_rate_sigma,
                            range_rate_sigma**2,
                        ],
                    ])
                    radar_noise = rng.multivariate_normal(
                        np.zeros(2), radar_covariance
                    )
                    information_id = f"{observer}->{target}:radar:{index}"
                    observations.append(ObservationMessage(
                        message_id=information_id,
                        physical_observation_id=information_id,
                        observer_id=observer,
                        target_id=target,
                        timestamp=float(timestamp),
                        modality="RADAR",
                        measurement=np.array([
                            measure_relative_range(
                                truth[observer][index], truth[target][index]
                            ),
                            measure_relative_range_rate(
                                truth[observer][index], truth[target][index]
                            ),
                        ]) + radar_noise,
                        covariance=radar_covariance,
                        metadata=inter_satellite_semantic_metadata("RADAR"),
                    ))
                if "RANGE" in relative_modalities and _measurement_is_due(
                    "RANGE", float(timestamp), measurement_periods
                ) and (
                    visible_range_opportunities is None
                    or (float(timestamp), observer, target, "RANGE")
                    in visible_range_opportunities
                ):
                    information_id = f"{observer}->{target}:range:{index}"
                    observations.append(ObservationMessage(
                        message_id=information_id,
                        physical_observation_id=information_id,
                        observer_id=observer, target_id=target,
                        timestamp=float(timestamp), modality="RANGE",
                        measurement=np.array([measure_relative_range(
                            truth[observer][index], truth[target][index]
                        ) + range_noise]),
                        covariance=np.array([[range_sigma**2]]),
                        metadata=inter_satellite_semantic_metadata("RANGE"),
                    ))
                if "RANGE_RATE" in relative_modalities and _measurement_is_due(
                    "RANGE_RATE", float(timestamp), measurement_periods
                ):
                    rate_noise = range_rate_rng.normal(0.0, range_rate_sigma)
                    if (
                        visible_range_opportunities is None
                        or (float(timestamp), observer, target, "RANGE_RATE")
                        in visible_range_opportunities
                    ):
                        information_id = f"{observer}->{target}:range_rate:{index}"
                        observations.append(ObservationMessage(
                            message_id=information_id,
                            physical_observation_id=information_id,
                            observer_id=observer,
                            target_id=target, timestamp=float(timestamp),
                            modality="RANGE_RATE",
                            measurement=np.array([measure_relative_range_rate(
                                truth[observer][index], truth[target][index]
                            ) + rate_noise]),
                            covariance=np.array([[range_rate_sigma**2]]),
                            metadata=inter_satellite_semantic_metadata("RANGE_RATE"),
                        ))
                angular_modality = (
                    "INFRARED" if "INFRARED" in relative_modalities
                    else "AZ_EL" if "AZ_EL" in relative_modalities
                    else None
                )
                if angular_modality is not None and _measurement_is_due(
                    angular_modality, float(timestamp), measurement_periods
                ):
                    angle_noise = az_el_rng.normal(0.0, az_el_sigma, 2)
                    if (
                        visible_range_opportunities is None
                        or (float(timestamp), observer, target, angular_modality)
                        in visible_range_opportunities
                    ):
                        information_id = (
                            f"{observer}->{target}:{angular_modality.lower()}:{index}"
                        )
                        truth_quaternion = (
                            attitude_history_by_node[observer][index]
                            if az_el_frame == "BODY" else None
                        )
                        estimate_quaternion = (
                            estimated_attitude_history_by_node[observer][index]
                            if az_el_frame == "BODY" else None
                        )
                        sensor_covariance = np.eye(2) * az_el_sigma**2
                        measurement_covariance = (
                            body_angle_effective_covariance(
                                truth[observer][index], truth[target][index],
                                quaternion_i2b_wxyz=estimate_quaternion,
                                sensor_covariance=sensor_covariance,
                                attitude_covariance=attitude_covariance,
                            )
                            if attitude_covariance is not None
                            else sensor_covariance
                        )
                        observations.append(ObservationMessage(
                            message_id=information_id,
                            physical_observation_id=information_id,
                            observer_id=observer,
                            target_id=target, timestamp=float(timestamp),
                            modality=angular_modality, frame=az_el_frame,
                            measurement=measure_relative_az_el(
                                truth[observer][index], truth[target][index],
                                frame=az_el_frame,
                                quaternion_i2b_wxyz=truth_quaternion,
                            ) + angle_noise,
                            covariance=measurement_covariance,
                            metadata={
                                **inter_satellite_semantic_metadata(angular_modality),
                                **(
                                    {"quaternion_i2b_wxyz": estimate_quaternion.copy()}
                                    if estimate_quaternion is not None else {}
                                ),
                            },
                        ))
                if "OPTICAL" in relative_modalities and _measurement_is_due(
                    "OPTICAL", float(timestamp), measurement_periods
                ):
                    optical_noise = optical_rng.normal(0.0, optical_sigma, 2)
                    if (
                        visible_range_opportunities is None
                        or (float(timestamp), observer, target, "OPTICAL")
                        in visible_range_opportunities
                    ):
                        information_id = f"{observer}->{target}:optical:{index}"
                        truth_optical_quaternion = (
                            attitude_history_by_node[observer][index]
                            if attitude_history_by_node is not None
                            else _target_pointing_quaternion(
                                truth[observer][index], truth[target][index]
                            )
                        )
                        estimate_optical_quaternion = (
                            estimated_attitude_history_by_node[observer][index]
                            if estimated_attitude_history_by_node is not None
                            else truth_optical_quaternion
                        )
                        observations.append(ObservationMessage(
                            message_id=information_id,
                            physical_observation_id=information_id,
                            observer_id=observer,
                            target_id=target,
                            timestamp=float(timestamp),
                            modality="OPTICAL",
                            frame="BODY",
                            measurement=measure_relative_optical_uv(
                                truth[observer][index], truth[target][index],
                                frame="BODY",
                                quaternion_i2b_wxyz=truth_optical_quaternion,
                            ) + optical_noise,
                            covariance=np.eye(2) * optical_sigma**2,
                            metadata={
                                **inter_satellite_semantic_metadata("OPTICAL"),
                                "quaternion_i2b_wxyz": (
                                    estimate_optical_quaternion.copy()
                                ),
                            },
                        ))
    return {
        "timestamps": timestamps, "truth": truth, "initial_states": initial_states,
        "initial_covariances": initial_covariances, "topology": topology,
        "observations": observations, "state_messages": state_messages,
        "absolute_observations": absolute_observations,
        "transmitted_messages": transmitted_messages,
        "lineages": {(receiver, source): f"{source}->{receiver}:0" for receiver, source in edges},
        "visibility_summary": visibility_summary,
    }


def _metrics(history, truth, transmitted_count, run_seconds):
    position = []; velocity = []; nees = []; nis = []; minimum = float("inf"); failures = 0
    nis_by_modality: dict[str, list[float]] = {}
    for node in history.node_ids:
        error = history.active_state_history_by_node[node] - truth[node]
        position.append(error[:, :3]); velocity.append(error[:, 3:])
        nees.extend(compute_nees_history(history.active_state_history_by_node[node], truth[node], history.active_covariance_history_by_node[node]))
        for epoch in history.nis_history_by_node[node]:
            for information_id, value in epoch.items():
                if ":absolute:" in information_id:
                    continue
                nis.append(value)
                nis_by_modality.setdefault(
                    _modality_from_information_id(information_id), []
                ).append(value)
        for covariance in history.joint_covariance_history_by_node[node]:
            value = float(np.linalg.eigvalsh(covariance).min()); minimum = min(minimum, value)
            failures += int(value < -1e-8)
    nees = np.asarray(nees); nis = np.asarray(nis)
    accepted = int(history.refresh_diagnostics.get("accepted", 0))
    rejected = sum(value for key, value in history.refresh_diagnostics.items() if key != "accepted")
    rejection_counts = {
        key: int(value) for key, value in history.refresh_diagnostics.items()
        if key != "accepted" and value
    }
    replay_stats = list(history.replay_performance_by_node.values())
    performance = {
        "total_replay_seconds": sum(value.total_replay_seconds for value in replay_stats),
        "replay_count": sum(value.replay_count for value in replay_stats),
        "batch_count": sum(value.batch_count for value in replay_stats),
        "maximum_replay_seconds": max(
            (value.maximum_replay_seconds for value in replay_stats), default=0.0
        ),
        "maximum_replay_span": max(
            (value.maximum_replay_span for value in replay_stats), default=0.0
        ),
        "total_replay_span": sum(value.total_replay_span for value in replay_stats),
        "maximum_batch_size": max(
            (value.maximum_batch_size for value in replay_stats), default=0
        ),
        "replayed_remote_events": sum(value.replayed_remote_events for value in replay_stats),
        "replayed_observations": sum(value.replayed_observations for value in replay_stats),
        "fallback_count": sum(value.fallback_count for value in replay_stats),
        "maximum_remote_event_count": max(
            (value.maximum_remote_event_count for value in replay_stats), default=0
        ),
        "maximum_observation_count": max(
            (value.maximum_observation_count for value in replay_stats), default=0
        ),
        "maximum_checkpoint_count": max(
            (value.maximum_checkpoint_count for value in replay_stats), default=0
        ),
        "maximum_posterior_state_count": max(
            (value.maximum_posterior_state_count for value in replay_stats), default=0
        ),
        "maximum_pinned_checkpoint_count": max(
            (value.maximum_pinned_checkpoint_count for value in replay_stats), default=0
        ),
        "maximum_resync_required_count": max(
            (value.maximum_resync_required_count for value in replay_stats), default=0
        ),
        "maximum_retained_journal_count": max(
            (value.maximum_retained_journal_count for value in replay_stats), default=0
        ),
    }
    return (
        compute_rmse(np.vstack(position)), compute_rmse(np.vstack(velocity)),
        float(np.mean(nees)), _coverage(nees, NEES_95_DOF6),
        float(np.mean(nis)), _modality_aware_nis_coverage(nis_by_modality), minimum,
        accepted, transmitted_count, rejected, failures, rejection_counts,
        float(run_seconds), performance,
        {key: float(np.mean(value)) for key, value in nis_by_modality.items()},
        {
            key: _coverage(np.asarray(value), _nis_interval(key))
            for key, value in nis_by_modality.items()
        },
    )


def _coverage(values, interval):
    lower, upper = interval
    return float(np.mean((values >= lower) & (values <= upper)))


def _modality_from_information_id(information_id):
    if ":radar:" in information_id:
        return "RADAR"
    if ":range_rate:" in information_id:
        return "RANGE_RATE"
    if ":az_el:" in information_id:
        return "AZ_EL"
    if ":infrared:" in information_id:
        return "INFRARED"
    if ":optical:" in information_id:
        return "OPTICAL"
    return "RANGE"


def _nis_interval(modality):
    return (
        NIS_95_DOF2
        if modality in {"RADAR", "AZ_EL", "INFRARED", "OPTICAL"}
        else NIS_95_DOF1
    )


def _target_pointing_quaternion(observer_state, target_state):
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    x_body_eci = target[:3] - observer[:3]
    x_body_eci /= np.linalg.norm(x_body_eci)
    reference = observer[:3] / np.linalg.norm(observer[:3])
    z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
    if np.linalg.norm(z_body_eci) < 1e-10:
        reference = np.array([0.0, 0.0, 1.0])
        z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
    z_body_eci /= np.linalg.norm(z_body_eci)
    y_body_eci = np.cross(z_body_eci, x_body_eci)
    return dcm_to_quat_wxyz(np.vstack((x_body_eci, y_body_eci, z_body_eci)))


def _validated_communication_outages(outages, *, edges):
    if outages is None:
        return {}
    valid_edges = set(edges)
    normalized = {}
    for raw_edge, raw_windows in outages.items():
        edge = (str(raw_edge[0]), str(raw_edge[1]))
        if edge not in valid_edges:
            raise ValueError(f"Communication outage references unknown link: {edge}")
        windows = []
        for start, end in raw_windows:
            start = float(start)
            end = float(end)
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                raise ValueError(
                    "Communication outage windows require finite start <= end."
                )
            windows.append((start, end))
        normalized[edge] = tuple(windows)
    return normalized


def _link_is_in_outage(outages, *, receiver, source, timestamp):
    return any(
        start <= timestamp <= end
        for start, end in outages.get((str(receiver), str(source)), ())
    )


def _validated_measurement_periods(periods, *, modalities):
    if periods is None:
        return {}
    unknown = set(periods) - set(modalities)
    if unknown:
        raise ValueError(
            f"Measurement periods reference disabled modalities: {sorted(unknown)}"
        )
    normalized = {str(key): float(value) for key, value in periods.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("Measurement periods must be finite and positive.")
    return normalized


def _measurement_is_due(modality, timestamp, periods):
    period = periods.get(str(modality))
    if period is None:
        return True
    quotient = float(timestamp) / period
    return bool(np.isclose(quotient, round(quotient), rtol=0.0, atol=1e-9))


def _modality_aware_nis_coverage(values_by_modality):
    covered = 0
    count = 0
    for modality, values in values_by_modality.items():
        array = np.asarray(values)
        lower, upper = _nis_interval(modality)
        covered += int(np.count_nonzero((array >= lower) & (array <= upper)))
        count += int(array.size)
    return covered / count if count else float("nan")


def _mean_metric_dict(values):
    keys = sorted({key for value in values for key in value})
    return {
        key: float(np.mean([value[key] for value in values if key in value]))
        for key in keys
    }


def _sum_rejection_counts(values):
    result = {}
    for counts in values:
        for key, count in counts.items():
            result[key] = result.get(key, 0) + int(count)
    return result


def export_exact_transport_diagnostics(
    result: ExactTransportScaleScanResult, output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(result.diagnostic_records)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path
