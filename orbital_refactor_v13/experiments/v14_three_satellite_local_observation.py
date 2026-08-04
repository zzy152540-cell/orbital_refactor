from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_exact_transport_scale_scan import _build_case, _metrics
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.metrics import compute_nees_history, compute_rmse
from scenarios.fleet_scenario import (
    DifferentialOrbitOffset,
    generate_differential_orbit_fleet_scenario,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    evaluate_inter_satellite_visibility,
)


@dataclass(frozen=True)
class ThreeSatelliteLocalObservationSummary:
    visibility_case: str
    mode: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_position_rmse_by_node: dict[str, float]
    mean_nees_by_node: dict[str, float]
    mean_nis_by_modality: dict[str, float]
    mean_nis_95_coverage_by_modality: dict[str, float]
    observation_count_by_directed_edge: dict[tuple[str, str], int]
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int
    configured_sensor_modalities: tuple[str, ...] = ("RADAR", "INFRARED")
    transported_measurement_components: tuple[str, ...] = (
        "RANGE", "RANGE_RATE", "AZ_EL",
    )
    full_three_sensor_suite: bool = False


@dataclass(frozen=True)
class ThreeSatelliteLocalObservationResult:
    summary_by_case_and_mode: dict[
        tuple[str, str], ThreeSatelliteLocalObservationSummary
    ]
    visibility_summary: VisibilityOpportunitySummary


@dataclass(frozen=True)
class OpticalSchedulingSummary:
    selected_count_by_directed_edge: dict[tuple[str, str], int]
    switch_count_by_observer: dict[str, int]
    maximum_unobserved_visible_epochs_by_directed_edge: dict[tuple[str, str], int]


@dataclass(frozen=True)
class ThreeSatelliteBodySchedulingResult:
    eci_upper_bound: ThreeSatelliteLocalObservationSummary
    body_scheduled: ThreeSatelliteLocalObservationSummary
    scheduling: OpticalSchedulingSummary


def run_v14_three_satellite_local_observation_experiment(
    *, seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
    maximum_range: float = 5000.0,
    range_sigma: float = 2.0, range_rate_sigma: float = 0.05,
    radar_correlation: float = 0.0,
    az_el_sigma: float = np.deg2rad(0.05),
    absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
) -> ThreeSatelliteLocalObservationResult:
    """Run three physical satellites using only each observer's own measurements."""

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_differential_orbit_fleet_scenario(
        timestamps=timestamps,
        base_semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.001, inclination=np.deg2rad(23.0),
        raan=0.0, argument_of_perigee=0.0, base_true_anomaly=0.0,
        offset_by_node={
            "sat_a": DifferentialOrbitOffset(),
            "sat_b": DifferentialOrbitOffset(
                semi_major_axis=1500.0, true_anomaly=0.0005,
            ),
            "sat_c": DifferentialOrbitOffset(
                semi_major_axis=-1200.0, true_anomaly=-0.0007,
            ),
        },
    )
    initial_truth = {
        node: history[0]
        for node, history in scenario.truth_state_history_by_node.items()
    }
    modalities = ("RADAR", "INFRARED", "OPTICAL")
    visibility = {
        modality: VisibilityConfig(maximum_range=maximum_range)
        for modality in modalities
    }
    cases = ("continuous", "visibility_limited")
    modes = ("propagate_only", "exact_transport_event_replay")
    collected = {(case, mode): [] for case in cases for mode in modes}
    collected_by_node = {(case, mode): [] for case in cases for mode in modes}
    edge_counts = {}
    visibility_summary = None
    for seed in range(seeds):
        built = {
            "continuous": _build_case(
                seed=seed, duration=duration, dt=dt,
                range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
                radar_correlation=radar_correlation,
                az_el_sigma=az_el_sigma, absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=3, topology_type="ring",
                truth_initial_state_by_node=initial_truth,
                relative_modalities=modalities,
            ),
            "visibility_limited": _build_case(
                seed=seed, duration=duration, dt=dt,
                range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
                radar_correlation=radar_correlation,
                az_el_sigma=az_el_sigma, absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=3, topology_type="ring",
                visibility_by_modality=visibility,
                truth_initial_state_by_node=initial_truth,
                relative_modalities=modalities,
            ),
        }
        if visibility_summary is None:
            visibility_summary = built["visibility_limited"]["visibility_summary"]
        for case_name, case in built.items():
            counts: dict[tuple[str, str], int] = {}
            for observation in case["observations"]:
                edge = (str(observation.observer_id), str(observation.target_id))
                counts[edge] = counts.get(edge, 0) + 1
            edge_counts.setdefault(case_name, counts)
            for mode in modes:
                history = run_network_schmidt_filter(
                    timestamps=case["timestamps"],
                    initial_state_by_node=case["initial_states"],
                    initial_covariance_by_node=case["initial_covariances"],
                    topology=case["topology"],
                    observation_messages=case["observations"],
                    observation_usage="observer_only",
                    process_noise_acceleration=process_noise_acceleration,
                    consider_refresh_mode=mode,
                    state_messages_by_receiver=(
                        case["state_messages"]
                        if mode == "exact_transport_event_replay" else None
                    ),
                    replay_history_window=(
                        10.0 if mode == "exact_transport_event_replay" else None
                    ),
                    expected_lineage_by_link=(
                        case["lineages"]
                        if mode == "exact_transport_event_replay" else None
                    ),
                )
                collected[(case_name, mode)].append(
                    _metrics(history, case["truth"], 0, 0.0)
                )
                collected_by_node[(case_name, mode)].append(
                    _metrics_by_node(history, case["truth"])
                )
    summaries = {}
    for (case_name, mode), values in collected.items():
        accepted = sum(value[7] for value in values)
        rejected = sum(value[9] for value in values)
        modalities_present = sorted({key for value in values for key in value[14]})
        summaries[(case_name, mode)] = ThreeSatelliteLocalObservationSummary(
            visibility_case=case_name, mode=mode, run_count=len(values),
            mean_position_rmse=float(np.mean([value[0] for value in values])),
            mean_velocity_rmse=float(np.mean([value[1] for value in values])),
            mean_nees=float(np.mean([value[2] for value in values])),
            mean_position_rmse_by_node={
                node: float(np.mean([
                    value[node][0]
                    for value in collected_by_node[(case_name, mode)]
                ]))
                for node in sorted(initial_truth)
            },
            mean_nees_by_node={
                node: float(np.mean([
                    value[node][1]
                    for value in collected_by_node[(case_name, mode)]
                ]))
                for node in sorted(initial_truth)
            },
            mean_nis_by_modality={
                key: float(np.mean([value[14][key] for value in values]))
                for key in modalities_present
            },
            mean_nis_95_coverage_by_modality={
                key: float(np.mean([value[15][key] for value in values]))
                for key in modalities_present
            },
            observation_count_by_directed_edge=edge_counts[case_name],
            message_acceptance_rate=(
                accepted / (accepted + rejected) if accepted + rejected else 0.0
            ),
            message_rejection_count=rejected,
            psd_failure_count=sum(value[10] for value in values),
            configured_sensor_modalities=("OPTICAL", "INFRARED", "RADAR"),
            transported_measurement_components=(
                "RADAR", "INFRARED", "OPTICAL",
            ),
            full_three_sensor_suite=True,
        )
    return ThreeSatelliteLocalObservationResult(summaries, visibility_summary)


def _metrics_by_node(history, truth):
    result = {}
    for node in history.node_ids:
        error = history.active_state_history_by_node[node] - truth[node]
        nees = compute_nees_history(
            history.active_state_history_by_node[node], truth[node],
            history.active_covariance_history_by_node[node],
        )
        result[node] = (compute_rmse(error[:, :3]), float(np.mean(nees)))
    return result


def run_v14_three_satellite_body_scheduling_experiment(
    *, seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
    maximum_range: float = 5000.0, fov_half_angle: float = np.deg2rad(2.0),
    range_sigma: float = 2.0, range_rate_sigma: float = 0.05,
    az_el_sigma: float = np.deg2rad(0.05), absolute_sigma: float = 3.0,
    process_noise_acceleration: float = 1e-8,
) -> ThreeSatelliteBodySchedulingResult:
    """Compare all-visible ECI angles with one scheduled BODY optical target."""

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = _three_satellite_scenario(timestamps)
    truth = scenario.truth_state_history_by_node
    initial_truth = {node: values[0] for node, values in truth.items()}
    attitude_history, scheduling = _longest_unobserved_first_attitudes(
        timestamps=timestamps, truth=truth, maximum_range=maximum_range,
    )
    modalities = ("RANGE", "RANGE_RATE", "AZ_EL")
    eci_visibility = {
        modality: VisibilityConfig(maximum_range=maximum_range)
        for modality in modalities
    }
    body_visibility = {
        "RANGE": VisibilityConfig(maximum_range=maximum_range),
        "RANGE_RATE": VisibilityConfig(maximum_range=maximum_range),
        "AZ_EL": VisibilityConfig(
            maximum_range=maximum_range,
            field_of_view_half_angle=fov_half_angle,
        ),
    }
    collected = {"eci_upper_bound": [], "body_scheduled": []}
    collected_by_node = {"eci_upper_bound": [], "body_scheduled": []}
    edge_counts = {}
    for seed in range(seeds):
        cases = {
            "eci_upper_bound": _build_case(
                seed=seed, duration=duration, dt=dt,
                range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
                az_el_sigma=az_el_sigma, absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=3, topology_type="ring",
                visibility_by_modality=eci_visibility,
                truth_initial_state_by_node=initial_truth,
                relative_modalities=modalities,
            ),
            "body_scheduled": _build_case(
                seed=seed, duration=duration, dt=dt,
                range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
                az_el_sigma=az_el_sigma, absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=3, topology_type="ring",
                visibility_by_modality=body_visibility,
                truth_initial_state_by_node=initial_truth,
                relative_modalities=modalities,
                az_el_frame="BODY",
                attitude_history_by_node=attitude_history,
            ),
        }
        for case_name, case in cases.items():
            if case_name not in edge_counts:
                counts = {}
                for observation in case["observations"]:
                    edge = (
                        str(observation.observer_id),
                        str(observation.target_id),
                    )
                    counts[edge] = counts.get(edge, 0) + 1
                edge_counts[case_name] = counts
            history = run_network_schmidt_filter(
                timestamps=case["timestamps"],
                initial_state_by_node=case["initial_states"],
                initial_covariance_by_node=case["initial_covariances"],
                topology=case["topology"],
                observation_messages=case["observations"],
                observation_usage="observer_only",
                process_noise_acceleration=process_noise_acceleration,
                consider_refresh_mode="exact_transport_event_replay",
                state_messages_by_receiver=case["state_messages"],
                replay_history_window=10.0,
                expected_lineage_by_link=case["lineages"],
            )
            collected[case_name].append(_metrics(history, case["truth"], 0, 0.0))
            collected_by_node[case_name].append(
                _metrics_by_node(history, case["truth"])
            )
    summaries = {
        case_name: _aggregate_scheduled_case(
            case_name, values, collected_by_node[case_name],
            edge_counts[case_name], tuple(initial_truth),
        )
        for case_name, values in collected.items()
    }
    return ThreeSatelliteBodySchedulingResult(
        summaries["eci_upper_bound"], summaries["body_scheduled"], scheduling
    )


def _three_satellite_scenario(timestamps):
    return generate_differential_orbit_fleet_scenario(
        timestamps=timestamps,
        base_semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.001, inclination=np.deg2rad(23.0),
        raan=0.0, argument_of_perigee=0.0, base_true_anomaly=0.0,
        offset_by_node={
            "sat_a": DifferentialOrbitOffset(),
            "sat_b": DifferentialOrbitOffset(
                semi_major_axis=1500.0, true_anomaly=0.0005,
            ),
            "sat_c": DifferentialOrbitOffset(
                semi_major_axis=-1200.0, true_anomaly=-0.0007,
            ),
        },
    )


def _longest_unobserved_first_attitudes(*, timestamps, truth, maximum_range):
    node_ids = tuple(truth)
    range_limits = VisibilityConfig(maximum_range=maximum_range)
    selected_history = {node: [] for node in node_ids}
    last_selected = {
        (observer, target): -1
        for observer in node_ids for target in node_ids if target != observer
    }
    attitude = {node: [] for node in node_ids}
    visible_history = {
        edge: [] for edge in last_selected
    }
    for index, _ in enumerate(timestamps):
        for observer in node_ids:
            candidates = []
            for target in node_ids:
                if target == observer:
                    continue
                visible = evaluate_inter_satellite_visibility(
                    truth[observer][index], truth[target][index], range_limits,
                ).visible
                visible_history[(observer, target)].append(visible)
                if visible:
                    candidates.append(target)
            selected = (
                min(candidates, key=lambda target: (
                    last_selected[(observer, target)], target,
                ))
                if candidates else None
            )
            selected_history[observer].append(selected)
            if selected is not None:
                last_selected[(observer, selected)] = index
                line_of_sight = (
                    truth[selected][index, :3] - truth[observer][index, :3]
                )
            else:
                line_of_sight = truth[observer][index, 3:]
            attitude[observer].append(
                _quaternion_with_x_boresight(line_of_sight, truth[observer][index, :3])
            )
    counts = {}
    switches = {}
    maximum_gaps = {}
    for observer, selected in selected_history.items():
        switches[observer] = sum(
            left is not None and right is not None and left != right
            for left, right in zip(selected, selected[1:])
        )
        for target in node_ids:
            if target == observer:
                continue
            edge = (observer, target)
            counts[edge] = sum(value == target for value in selected)
            gap = best = 0
            for is_visible, chosen in zip(visible_history[edge], selected):
                gap = gap + 1 if is_visible and chosen != target else 0
                best = max(best, gap)
            maximum_gaps[edge] = best
    return (
        {node: np.vstack(values) for node, values in attitude.items()},
        OpticalSchedulingSummary(counts, switches, maximum_gaps),
    )


def _quaternion_with_x_boresight(line_of_sight, observer_position):
    x_axis = np.asarray(line_of_sight, dtype=float).copy()
    x_axis /= np.linalg.norm(x_axis)
    reference = np.asarray(observer_position, dtype=float).copy()
    reference /= np.linalg.norm(reference)
    z_axis = reference - np.dot(reference, x_axis) * x_axis
    if np.linalg.norm(z_axis) < 1e-10:
        reference = np.array([0.0, 0.0, 1.0])
        z_axis = reference - np.dot(reference, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return dcm_to_quat_wxyz(np.vstack((x_axis, y_axis, z_axis)))


def _aggregate_scheduled_case(case_name, values, by_node, edge_counts, node_ids):
    accepted = sum(value[7] for value in values)
    rejected = sum(value[9] for value in values)
    modalities = sorted({key for value in values for key in value[14]})
    return ThreeSatelliteLocalObservationSummary(
        visibility_case=case_name, mode="exact_transport_event_replay",
        run_count=len(values),
        mean_position_rmse=float(np.mean([value[0] for value in values])),
        mean_velocity_rmse=float(np.mean([value[1] for value in values])),
        mean_nees=float(np.mean([value[2] for value in values])),
        mean_position_rmse_by_node={
            node: float(np.mean([value[node][0] for value in by_node]))
            for node in node_ids
        },
        mean_nees_by_node={
            node: float(np.mean([value[node][1] for value in by_node]))
            for node in node_ids
        },
        mean_nis_by_modality={
            key: float(np.mean([value[14][key] for value in values]))
            for key in modalities
        },
        mean_nis_95_coverage_by_modality={
            key: float(np.mean([value[15][key] for value in values]))
            for key in modalities
        },
        observation_count_by_directed_edge=edge_counts,
        message_acceptance_rate=(
            accepted / (accepted + rejected) if accepted + rejected else 0.0
        ),
        message_rejection_count=rejected,
        psd_failure_count=sum(value[10] for value in values),
    )
