from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import (
    NetworkTopology,
)
from experiments.fleet_case_setup import (
    build_fleet_geometry,
    build_visibility_selection,
    normalize_attitude_inputs,
)
from experiments.exact_transport_state_simulator import (
    ExactTransportStateSimulator,
)
from experiments.inter_satellite_observation_factory import (
    generate_inter_satellite_observations_for_epoch,
    target_pointing_quaternion as _target_pointing_quaternion,
)
from experiments.summary_statistics import mean_metric_dict
from experiments.network_filter_metrics import (
    modality_aware_nis_coverage as _modality_aware_nis_coverage,
    modality_from_information_id as _modality_from_information_id,
    network_history_metrics as _metrics,
    nis_interval as _nis_interval,
)
from experiments.scenario_controls import (
    link_is_in_outage as _link_is_in_outage,
    measurement_is_due as _measurement_is_due,
    topology_edge_is_inactive as _topology_edge_is_inactive,
    topology_runtime_schedule as _topology_runtime_schedule,
    validate_absolute_navigation_dropouts,
    validate_communication_outages as _validated_communication_outages,
    validate_measurement_periods as _validated_measurement_periods,
    validate_topology_inactive_windows as _validated_topology_inactive_windows,
)
from orbital_core.measurement_semantics import (
    PHYSICAL_SENSOR_MODALITIES,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    VisibilityTemporalFilterConfig,
)


# RANGE-only remains available, but callers must select it explicitly when
# running the legacy covariance-consistency or ablation baseline.
RANGE_ONLY_BASELINE_MODALITIES = ("RANGE",)

Array = np.ndarray


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
    relative_modalities: tuple[str, ...] = PHYSICAL_SENSOR_MODALITIES,
    communication_outage_windows_by_directed_link: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ] | None = None,
    measurement_period_by_modality: Mapping[str, float] | None = None,
    visibility_temporal_filter_by_modality: Mapping[
        str, VisibilityTemporalFilterConfig
    ] | None = None,
) -> ExactTransportScaleScanResult:
    """Run the production network API over five configurable fleet cases.

    The public default exercises the physical RADAR/INFRARED/OPTICAL suite.
    Pass ``RANGE_ONLY_BASELINE_MODALITIES`` explicitly for the legacy
    covariance-consistency baseline.
    """
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
            case = build_exact_transport_case(
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
            mean_nis_by_modality=mean_metric_dict([value[14] for value in values]),
            mean_nis_95_coverage_by_modality=mean_metric_dict(
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
    relative_modalities: tuple[str, ...] = PHYSICAL_SENSOR_MODALITIES,
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


def build_exact_transport_case(
                *, seed, duration, dt, range_sigma, absolute_sigma,
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
                absolute_navigation_dropout_windows_by_node=None,
                topology_inactive_windows_by_undirected_edge=None,
                measurement_period_by_modality=None,
                topology_override: NetworkTopology | None = None,
                relative_modalities=("RANGE",),
                future_noise_seed=None, future_noise_start_index=None):
    if (future_noise_seed is None) != (future_noise_start_index is None):
        raise ValueError(
            "future_noise_seed and future_noise_start_index must be set together."
        )
    if (
        future_noise_start_index is not None
        and int(future_noise_start_index) < 0
    ):
        raise ValueError("future_noise_start_index cannot be negative.")
    rng = np.random.default_rng(20260830 + seed)
    range_rate_rng = np.random.default_rng(20260930 + seed)
    az_el_rng = np.random.default_rng(20261030 + seed)
    optical_rng = np.random.default_rng(20261130 + seed)
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    geometry = build_fleet_geometry(
        timestamps=timestamps,
        node_count=node_count,
        truth_initial_state_by_node=truth_initial_state_by_node,
        topology_type=topology_type,
        topology_override=topology_override,
    )
    scenario = geometry.scenario
    truth_initials = geometry.truth_initial_state_by_node
    truth = geometry.truth_state_history_by_node
    topology = geometry.topology
    (
        az_el_frame,
        attitude_history_by_node,
        estimated_attitude_history_by_node,
        attitude_covariance,
    ) = normalize_attitude_inputs(
        frame=az_el_frame,
        attitude_history_by_node=attitude_history_by_node,
        estimated_attitude_history_by_node=estimated_attitude_history_by_node,
        attitude_covariance=attitude_covariance,
        node_ids=scenario.node_ids,
        sample_count=len(timestamps),
    )
    covariance = np.diag([10, 10, 10, 0.02, 0.02, 0.02]) ** 2
    initial_states = {node: truth_initials[node] + rng.multivariate_normal(np.zeros(6), covariance) for node in scenario.node_ids}
    initial_covariances = {node: covariance.copy() for node in scenario.node_ids}
    visibility_summary, visible_range_opportunities = build_visibility_selection(
        timestamps=timestamps,
        truth_state_history_by_node=truth,
        topology=topology,
        relative_modalities=relative_modalities,
        visibility_by_modality=visibility_by_modality,
        visibility_temporal_filter_by_modality=(
            visibility_temporal_filter_by_modality
        ),
        attitude_history_by_node=attitude_history_by_node,
        frame=az_el_frame,
    )
    edges = tuple((receiver, source) for receiver in topology.node_ids for source in topology.neighbors(receiver))
    communication_outages = _validated_communication_outages(
        communication_outage_windows_by_directed_link, edges=edges,
    )
    topology_inactive_windows = _validated_topology_inactive_windows(
        topology_inactive_windows_by_undirected_edge, topology=topology,
    )
    topology_versions, active_neighbors_by_timestamp = (
        _topology_runtime_schedule(
            timestamps, topology=topology,
            inactive_windows=topology_inactive_windows,
        )
    )
    (
        absolute_navigation_dropout_windows,
        node_dropout_windows,
    ) = validate_absolute_navigation_dropouts(
        absolute_navigation_dropout_windows,
        absolute_navigation_dropout_windows_by_node,
        node_ids=scenario.node_ids,
    )
    measurement_periods = _validated_measurement_periods(
        measurement_period_by_modality, modalities=relative_modalities,
    )
    state_simulator = ExactTransportStateSimulator(
        initial_state_by_node=initial_states,
        initial_covariance_by_node=initial_covariances,
        topology=topology,
        edges=edges,
        packet_loss=packet_loss,
        delay=delay,
        random_seed=seed,
        acknowledge_messages=acknowledge_messages,
        dt=dt,
        process_noise_acceleration=process_noise_acceleration,
        absolute_sigma=absolute_sigma,
        absolute_navigation_dropout_windows=(
            absolute_navigation_dropout_windows
        ),
        node_dropout_windows=node_dropout_windows,
        communication_outages=communication_outages,
        topology_inactive_windows=topology_inactive_windows,
        topology_versions=topology_versions,
    )
    observations = []
    absolute_observations = []
    for index, timestamp in enumerate(timestamps):
        if (
            future_noise_start_index is not None
            and index == int(future_noise_start_index)
        ):
            future_seed = int(future_noise_seed)
            rng = np.random.default_rng(20260830 + future_seed)
            range_rate_rng = np.random.default_rng(20260930 + future_seed)
            az_el_rng = np.random.default_rng(20261030 + future_seed)
            optical_rng = np.random.default_rng(20261130 + future_seed)
        absolute_observations.extend(state_simulator.advance_epoch(
            index=index,
            timestamp=timestamp,
            truth=truth,
            rng=rng,
        ))
        observations.extend(generate_inter_satellite_observations_for_epoch(
            index=index,
            timestamp=timestamp,
            truth=truth,
            topology=topology,
            topology_inactive_windows=topology_inactive_windows,
            relative_modalities=relative_modalities,
            measurement_periods=measurement_periods,
            visible_opportunities=visible_range_opportunities,
            range_rng=rng,
            range_rate_rng=range_rate_rng,
            az_el_rng=az_el_rng,
            optical_rng=optical_rng,
            range_sigma=range_sigma,
            range_rate_sigma=range_rate_sigma,
            radar_correlation=radar_correlation,
            az_el_sigma=az_el_sigma,
            optical_sigma=optical_sigma,
            az_el_frame=az_el_frame,
            attitude_history_by_node=attitude_history_by_node,
            estimated_attitude_history_by_node=(
                estimated_attitude_history_by_node
            ),
            attitude_covariance=attitude_covariance,
        ))
    return {
        "timestamps": timestamps, "truth": truth, "initial_states": initial_states,
        "initial_covariances": initial_covariances, "topology": topology,
        "observations": observations,
        "state_messages": state_simulator.state_messages,
        "absolute_observations": absolute_observations,
        "transmitted_messages": state_simulator.transmitted_messages,
        "lineages": state_simulator.lineages,
        "visibility_summary": visibility_summary,
        "topology_version_by_timestamp": topology_versions,
        "active_neighbors_by_timestamp": active_neighbors_by_timestamp,
    }


# Compatibility for existing research scripts and tests that used the old
# private name before case construction became shared experiment infrastructure.
_build_case = build_exact_transport_case


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
