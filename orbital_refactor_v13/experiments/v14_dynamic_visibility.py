from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_exact_transport_scale_scan import _build_case
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.attitude import (
    quat_multiply_wxyz,
    quat_normalize_wxyz,
    small_angle_quaternion_wxyz,
)
from orbital_core.metrics import compute_nees_history, compute_rmse
from scenarios.fleet_scenario import (
    DifferentialOrbitOffset,
    generate_differential_orbit_fleet_scenario,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    VisibilityOpportunitySummary,
    VisibilityTemporalFilterConfig,
)


@dataclass(frozen=True)
class DynamicVisibilityRunSummary:
    transition_type: str
    relative_modalities: tuple[str, ...]
    visibility_case: str
    mode: str
    run_count: int
    transition_timestamp: float
    mean_pre_transition_position_rmse: float
    mean_post_transition_position_rmse: float
    mean_final_position_error: float
    mean_pre_transition_velocity_rmse: float
    mean_post_transition_velocity_rmse: float
    mean_final_velocity_error: float
    mean_nees: float
    mean_nis: float
    mean_nis_by_modality: dict[str, float]
    mean_transition_nis: float | None
    mean_transition_nis_by_modality: dict[str, float]
    message_acceptance_rate: float
    message_rejection_count: int
    psd_failure_count: int


@dataclass(frozen=True)
class DynamicVisibilityExperimentResult:
    summary_by_case_and_mode: dict[tuple[str, str], DynamicVisibilityRunSummary]
    visibility_summary: VisibilityOpportunitySummary


@dataclass(frozen=True)
class RangeRateSensitivityResult:
    range_only_summary: DynamicVisibilityRunSummary
    summary_by_range_rate_sigma: dict[float, DynamicVisibilityRunSummary]


@dataclass(frozen=True)
class AzElSensitivityResult:
    range_and_rate_summary: DynamicVisibilityRunSummary
    summary_by_az_el_sigma_degrees: dict[float, DynamicVisibilityRunSummary]


@dataclass(frozen=True)
class AttitudeErrorConsistencyResult:
    ideal_attitude: DynamicVisibilityRunSummary
    ignored_attitude_uncertainty: DynamicVisibilityRunSummary
    propagated_attitude_uncertainty: DynamicVisibilityRunSummary


def run_v14_dynamic_visibility_experiment(
    *, seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
    maximum_range: float | None = None, transition_type: str = "loss",
    visibility_driver: str = "range",
    range_sigma: float = 2.0,
    range_rate_sigma: float = 0.02,
    az_el_sigma: float = np.deg2rad(0.05),
    az_el_frame: str = "ECI",
    az_el_field_of_view_half_angle: float | None = None,
    attitude_error_sigma: float = 0.0,
    inflate_attitude_covariance: bool = False,
    fov_jitter_amplitude: float = 0.0,
    fov_hysteresis: float = 0.0,
    fov_acquisition_epochs: int = 1,
    fov_loss_epochs: int = 1,
    relative_modalities: tuple[str, ...] = ("RANGE",),
    absolute_sigma: float = 3.0, process_noise_acceleration: float = 1e-8,
) -> DynamicVisibilityExperimentResult:
    """Compare continuous and visibility-limited RANGE on a drifting orbit pair."""

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    if transition_type not in {"loss", "recovery"}:
        raise ValueError("transition_type must be 'loss' or 'recovery'.")
    if visibility_driver not in {"range", "fov"}:
        raise ValueError("visibility_driver must be 'range' or 'fov'.")
    if (
        not relative_modalities
        or len(set(relative_modalities)) != len(relative_modalities)
        or set(relative_modalities) - {"RANGE", "RANGE_RATE", "AZ_EL"}
    ):
        raise ValueError("relative_modalities must uniquely select RANGE, RANGE_RATE, and/or AZ_EL.")
    if maximum_range is None:
        maximum_range = (
            1.0e9 if visibility_driver == "fov"
            else (5000.0 if transition_type == "loss" else 5800.0)
        )
    anomaly_offset = -0.0006 if transition_type == "loss" else 0.0008
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = generate_differential_orbit_fleet_scenario(
        timestamps=timestamps,
        base_semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.001,
        inclination=np.deg2rad(23.0),
        raan=0.0,
        argument_of_perigee=0.0,
        base_true_anomaly=0.0,
        offset_by_node={
            "sat_a": DifferentialOrbitOffset(),
            "sat_b": DifferentialOrbitOffset(
                semi_major_axis=2000.0, true_anomaly=anomaly_offset,
            ),
        },
    )
    initial_truth = {
        node_id: values[0]
        for node_id, values in scenario.truth_state_history_by_node.items()
    }
    normalized_az_el_frame = str(az_el_frame).upper()
    if normalized_az_el_frame not in {"ECI", "BODY"}:
        raise ValueError("az_el_frame must be 'ECI' or 'BODY'.")
    if normalized_az_el_frame == "BODY" and "AZ_EL" not in relative_modalities:
        raise ValueError("BODY az_el_frame requires the AZ_EL modality.")
    if (
        az_el_field_of_view_half_angle is not None
        and normalized_az_el_frame != "BODY"
    ):
        raise ValueError("AZ_EL FOV requires az_el_frame='BODY'.")
    if visibility_driver == "fov":
        if normalized_az_el_frame != "BODY" or "AZ_EL" not in relative_modalities:
            raise ValueError("FOV visibility transitions require BODY AZ_EL.")
        if az_el_field_of_view_half_angle is None:
            az_el_field_of_view_half_angle = float(np.deg2rad(5.0))
    if attitude_error_sigma < 0.0:
        raise ValueError("attitude_error_sigma cannot be negative.")
    if attitude_error_sigma > 0.0 and normalized_az_el_frame != "BODY":
        raise ValueError("Attitude error requires az_el_frame='BODY'.")
    if inflate_attitude_covariance and attitude_error_sigma <= 0.0:
        raise ValueError(
            "Attitude covariance inflation requires positive attitude_error_sigma."
        )
    if fov_jitter_amplitude < 0.0:
        raise ValueError("fov_jitter_amplitude cannot be negative.")
    if (fov_jitter_amplitude > 0.0 or fov_hysteresis > 0.0) and (
        visibility_driver != "fov"
    ):
        raise ValueError("FOV jitter and hysteresis require visibility_driver='fov'.")
    attitude_history = (
        _two_node_target_pointing_attitude_history(
            scenario.truth_state_history_by_node
        )
        if normalized_az_el_frame == "BODY" else None
    )
    if visibility_driver == "fov":
        attitude_history = _slew_target_pointing_attitude_history(
            attitude_history,
            maximum_offset=2.0 * az_el_field_of_view_half_angle,
            transition_type=transition_type,
            jitter_amplitude=fov_jitter_amplitude,
        )
    visibility_config = {
        modality: VisibilityConfig(
            maximum_range=maximum_range,
            field_of_view_half_angle=(
                az_el_field_of_view_half_angle
                if modality == "AZ_EL" else None
            ),
        )
        for modality in relative_modalities
    }
    temporal_filter = (
        {
            "AZ_EL": VisibilityTemporalFilterConfig(
                acquisition_epochs=fov_acquisition_epochs,
                loss_epochs=fov_loss_epochs,
                fov_hysteresis=fov_hysteresis,
            ),
        }
        if visibility_driver == "fov" and (
            fov_hysteresis > 0.0
            or fov_acquisition_epochs > 1
            or fov_loss_epochs > 1
        ) else None
    )
    modes = ("propagate_only", "exact_transport_event_replay")
    collected = {
        (case, mode): []
        for case in ("continuous_range", "visibility_limited")
        for mode in modes
    }
    visibility_summary = None
    for seed in range(seeds):
        estimated_attitude_history = (
            _perturb_attitude_history(
                attitude_history, sigma=attitude_error_sigma, seed=seed,
            )
            if attitude_error_sigma > 0.0 else attitude_history
        )
        attitude_covariance = (
            np.eye(3) * attitude_error_sigma**2
            if inflate_attitude_covariance else None
        )
        cases = {
            "continuous_range": _build_case(
                seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
                range_rate_sigma=range_rate_sigma, az_el_sigma=az_el_sigma,
                az_el_frame=normalized_az_el_frame,
                attitude_history_by_node=attitude_history,
                estimated_attitude_history_by_node=estimated_attitude_history,
                attitude_covariance=attitude_covariance,
                visibility_temporal_filter_by_modality=temporal_filter,
                absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=2, topology_type="chain",
                truth_initial_state_by_node=initial_truth,
                relative_modalities=relative_modalities,
            ),
            "visibility_limited": _build_case(
                seed=seed, duration=duration, dt=dt, range_sigma=range_sigma,
                range_rate_sigma=range_rate_sigma, az_el_sigma=az_el_sigma,
                az_el_frame=normalized_az_el_frame,
                attitude_history_by_node=attitude_history,
                estimated_attitude_history_by_node=estimated_attitude_history,
                attitude_covariance=attitude_covariance,
                visibility_temporal_filter_by_modality=temporal_filter,
                absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                packet_loss=0.0, delay=0.0, acknowledge_messages=True,
                node_count=2, topology_type="chain",
                visibility_by_modality=visibility_config,
                truth_initial_state_by_node=initial_truth,
                relative_modalities=relative_modalities,
            ),
        }
        if visibility_summary is None:
            visibility_summary = cases["visibility_limited"]["visibility_summary"]
            if visibility_driver == "fov":
                count_by_timestamp = {
                    timestamp: count
                    for (timestamp, modality), count in
                    visibility_summary.visible_directed_edge_count_by_timestamp_and_modality.items()
                    if modality == "AZ_EL"
                }
            else:
                count_by_timestamp = (
                    visibility_summary.visible_directed_edge_count_by_timestamp
                )
            epoch_counts = tuple(count_by_timestamp.values())
            if not any(epoch_counts) or not any(count == 0 for count in epoch_counts):
                raise ValueError(
                    "The selected duration and range must contain a visibility transition."
                )
            epoch_visibility = tuple(
                (timestamp, count > 0)
                for timestamp, count in
                count_by_timestamp.items()
            )
            initially_visible = epoch_visibility[0][1]
            expected_initial = transition_type == "loss"
            if initially_visible != expected_initial:
                raise ValueError("Visibility transition direction does not match its type.")
            transition_timestamp = next(
                timestamp for timestamp, visible in epoch_visibility
                if visible != initially_visible
            )
        for visibility_case, case in cases.items():
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
                collected[(visibility_case, mode)].append(
                    _run_metrics(history, case["truth"], transition_timestamp)
                )
    summaries = {}
    for key, values in collected.items():
        visibility_case, mode = key
        accepted = sum(value[8] for value in values)
        rejected = sum(value[9] for value in values)
        transition_nis = [value[6] for value in values if value[6] is not None]
        nis_modalities = sorted({
            modality for value in values for modality in value[5]
        })
        transition_modalities = sorted({
            modality for value in values for modality in value[7]
        })
        summaries[key] = DynamicVisibilityRunSummary(
            transition_type=transition_type,
            relative_modalities=relative_modalities,
            visibility_case=visibility_case,
            mode=mode,
            run_count=len(values),
            transition_timestamp=transition_timestamp,
            mean_pre_transition_position_rmse=float(np.mean([value[0] for value in values])),
            mean_post_transition_position_rmse=float(np.mean([value[1] for value in values])),
            mean_final_position_error=float(np.mean([value[2] for value in values])),
            mean_pre_transition_velocity_rmse=float(
                np.mean([value[11] for value in values])
            ),
            mean_post_transition_velocity_rmse=float(
                np.mean([value[12] for value in values])
            ),
            mean_final_velocity_error=float(np.mean([value[13] for value in values])),
            mean_nees=float(np.mean([value[3] for value in values])),
            mean_nis=float(np.mean([value[4] for value in values])),
            mean_nis_by_modality={
                modality: float(np.mean([
                    value[5][modality] for value in values
                    if modality in value[5]
                ]))
                for modality in nis_modalities
            },
            mean_transition_nis=(
                float(np.mean(transition_nis)) if transition_nis else None
            ),
            mean_transition_nis_by_modality={
                modality: float(np.mean([
                    value[7][modality] for value in values
                    if modality in value[7]
                ]))
                for modality in transition_modalities
            },
            message_acceptance_rate=(
                accepted / (accepted + rejected) if accepted + rejected else 0.0
            ),
            message_rejection_count=rejected,
            psd_failure_count=sum(value[10] for value in values),
        )
    return DynamicVisibilityExperimentResult(summaries, visibility_summary)


def run_v14_range_rate_sensitivity(
    *, range_rate_sigmas: tuple[float, ...] = (0.01, 0.02, 0.05),
    seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
) -> RangeRateSensitivityResult:
    """Compare RANGE-only and dual-modality exact recovery using fixed seeds."""

    if (
        not range_rate_sigmas
        or len(set(range_rate_sigmas)) != len(range_rate_sigmas)
        or any(value <= 0.0 for value in range_rate_sigmas)
    ):
        raise ValueError("range_rate_sigmas must contain unique positive values.")
    key = ("visibility_limited", "exact_transport_event_replay")
    range_only = run_v14_dynamic_visibility_experiment(
        seeds=seeds, duration=duration, dt=dt, transition_type="recovery",
    ).summary_by_case_and_mode[key]
    summaries = {}
    for sigma in range_rate_sigmas:
        summaries[float(sigma)] = run_v14_dynamic_visibility_experiment(
            seeds=seeds, duration=duration, dt=dt, transition_type="recovery",
            relative_modalities=("RANGE", "RANGE_RATE"),
            range_rate_sigma=float(sigma),
        ).summary_by_case_and_mode[key]
    return RangeRateSensitivityResult(range_only, summaries)


def run_v14_az_el_sensitivity(
    *, az_el_sigmas_degrees: tuple[float, ...] = (0.01, 0.05, 0.1),
    range_rate_sigma: float = 0.05,
    seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
) -> AzElSensitivityResult:
    """Compare ECI AZ_EL noise levels against a fixed RANGE+RANGE_RATE baseline."""

    if (
        not az_el_sigmas_degrees
        or len(set(az_el_sigmas_degrees)) != len(az_el_sigmas_degrees)
        or any(value <= 0.0 for value in az_el_sigmas_degrees)
    ):
        raise ValueError("az_el_sigmas_degrees must contain unique positive values.")
    if range_rate_sigma <= 0.0:
        raise ValueError("range_rate_sigma must be positive.")
    key = ("visibility_limited", "exact_transport_event_replay")
    baseline = run_v14_dynamic_visibility_experiment(
        seeds=seeds, duration=duration, dt=dt, transition_type="recovery",
        relative_modalities=("RANGE", "RANGE_RATE"),
        range_rate_sigma=range_rate_sigma,
    ).summary_by_case_and_mode[key]
    summaries = {}
    for sigma_degrees in az_el_sigmas_degrees:
        summaries[float(sigma_degrees)] = run_v14_dynamic_visibility_experiment(
            seeds=seeds, duration=duration, dt=dt, transition_type="recovery",
            relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
            range_rate_sigma=range_rate_sigma,
            az_el_sigma=float(np.deg2rad(sigma_degrees)),
        ).summary_by_case_and_mode[key]
    return AzElSensitivityResult(baseline, summaries)


def run_v14_attitude_error_consistency(
    *, attitude_error_sigma_degrees: float = 0.05,
    az_el_sigma_degrees: float = 0.05,
    range_rate_sigma: float = 0.05,
    field_of_view_half_angle_degrees: float = 5.0,
    seeds: int = 10, duration: float = 120.0, dt: float = 2.0,
) -> AttitudeErrorConsistencyResult:
    """Compare ignored and propagated BODY attitude uncertainty."""

    if attitude_error_sigma_degrees <= 0.0:
        raise ValueError("attitude_error_sigma_degrees must be positive.")
    common = dict(
        seeds=seeds, duration=duration, dt=dt, transition_type="recovery",
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        range_rate_sigma=range_rate_sigma,
        az_el_sigma=float(np.deg2rad(az_el_sigma_degrees)),
        az_el_frame="BODY",
        az_el_field_of_view_half_angle=float(
            np.deg2rad(field_of_view_half_angle_degrees)
        ),
    )
    key = ("visibility_limited", "exact_transport_event_replay")
    ideal = run_v14_dynamic_visibility_experiment(
        **common,
    ).summary_by_case_and_mode[key]
    error_sigma = float(np.deg2rad(attitude_error_sigma_degrees))
    ignored = run_v14_dynamic_visibility_experiment(
        **common, attitude_error_sigma=error_sigma,
    ).summary_by_case_and_mode[key]
    propagated = run_v14_dynamic_visibility_experiment(
        **common, attitude_error_sigma=error_sigma,
        inflate_attitude_covariance=True,
    ).summary_by_case_and_mode[key]
    return AttitudeErrorConsistencyResult(ideal, ignored, propagated)


def _run_metrics(history, truth, transition_timestamp):
    pre_transition_errors = []
    post_transition_errors = []
    final_errors = []
    pre_transition_velocity_errors = []
    post_transition_velocity_errors = []
    final_velocity_errors = []
    nees = []
    nis = []
    nis_by_modality: dict[str, list[float]] = {}
    psd_failures = 0
    pre_transition_mask = history.timestamps < transition_timestamp
    post_transition_mask = ~pre_transition_mask
    transition_index = int(np.flatnonzero(
        np.isclose(history.timestamps, transition_timestamp)
    )[0])
    for node_id in history.node_ids:
        error = history.active_state_history_by_node[node_id] - truth[node_id]
        pre_transition_errors.append(error[pre_transition_mask, :3])
        post_transition_errors.append(error[post_transition_mask, :3])
        final_errors.append(float(np.linalg.norm(error[-1, :3])))
        pre_transition_velocity_errors.append(error[pre_transition_mask, 3:])
        post_transition_velocity_errors.append(error[post_transition_mask, 3:])
        final_velocity_errors.append(float(np.linalg.norm(error[-1, 3:])))
        nees.extend(compute_nees_history(
            history.active_state_history_by_node[node_id], truth[node_id],
            history.active_covariance_history_by_node[node_id],
        ))
        for epoch in history.nis_history_by_node[node_id]:
            for information_id, value in epoch.items():
                nis.append(value)
                modality = _modality_from_information_id(information_id)
                nis_by_modality.setdefault(modality, []).append(value)
        for covariance in history.joint_covariance_history_by_node[node_id]:
            psd_failures += int(np.linalg.eigvalsh(covariance).min() < -1e-8)
    accepted = int(history.refresh_diagnostics.get("accepted", 0))
    rejected = sum(
        value for key, value in history.refresh_diagnostics.items()
        if key != "accepted"
    )
    transition_values = []
    transition_by_modality: dict[str, list[float]] = {}
    for epochs in history.nis_history_by_node.values():
        for information_id, value in epochs[transition_index].items():
            transition_values.append(value)
            modality = _modality_from_information_id(information_id)
            transition_by_modality.setdefault(modality, []).append(value)
    return (
        compute_rmse(np.vstack(pre_transition_errors)),
        compute_rmse(np.vstack(post_transition_errors)),
        float(np.mean(final_errors)), float(np.mean(nees)), float(np.mean(nis)),
        {
            modality: float(np.mean(values))
            for modality, values in nis_by_modality.items()
        },
        (float(np.mean(transition_values)) if transition_values else None),
        {
            modality: float(np.mean(values))
            for modality, values in transition_by_modality.items()
        },
        accepted, rejected, psd_failures,
        compute_rmse(np.vstack(pre_transition_velocity_errors)),
        compute_rmse(np.vstack(post_transition_velocity_errors)),
        float(np.mean(final_velocity_errors)),
    )


def _modality_from_information_id(information_id):
    if ":range_rate:" in information_id:
        return "RANGE_RATE"
    if ":az_el:" in information_id:
        return "AZ_EL"
    return "RANGE"


def _two_node_target_pointing_attitude_history(truth_by_node):
    node_ids = tuple(truth_by_node)
    if len(node_ids) != 2:
        raise ValueError("Target-pointing attitude helper requires exactly two nodes.")
    result = {}
    for observer, target in (node_ids, tuple(reversed(node_ids))):
        quaternions = []
        for observer_state, target_state in zip(
            truth_by_node[observer], truth_by_node[target]
        ):
            x_body_eci = target_state[:3] - observer_state[:3]
            x_body_eci /= np.linalg.norm(x_body_eci)
            reference = observer_state[:3] / np.linalg.norm(observer_state[:3])
            z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
            if np.linalg.norm(z_body_eci) < 1e-10:
                reference = np.array([0.0, 0.0, 1.0])
                z_body_eci = reference - np.dot(reference, x_body_eci) * x_body_eci
            z_body_eci /= np.linalg.norm(z_body_eci)
            y_body_eci = np.cross(z_body_eci, x_body_eci)
            dcm_i2b = np.vstack((x_body_eci, y_body_eci, z_body_eci))
            quaternions.append(dcm_to_quat_wxyz(dcm_i2b))
        result[observer] = np.vstack(quaternions)
    return result


def _perturb_attitude_history(attitude_by_node, *, sigma, seed):
    rng = np.random.default_rng(20261130 + seed)
    result = {}
    for node_id, history in attitude_by_node.items():
        result[node_id] = np.vstack([
            quat_normalize_wxyz(quat_multiply_wxyz(
                small_angle_quaternion_wxyz(rng.normal(0.0, sigma, 3)),
                quaternion,
            ))
            for quaternion in history
        ])
    return result


def _slew_target_pointing_attitude_history(
    attitude_by_node, *, maximum_offset, transition_type, jitter_amplitude=0.0,
):
    result = {}
    for node_id, history in attitude_by_node.items():
        fractions = np.linspace(0.0, 1.0, len(history))
        if transition_type == "recovery":
            fractions = fractions[::-1]
        offsets = (
            maximum_offset * fractions
            + jitter_amplitude * np.sin(2.0 * np.pi * 12.0 * fractions)
        )
        result[node_id] = np.vstack([
            quat_normalize_wxyz(quat_multiply_wxyz(
                small_angle_quaternion_wxyz(np.array([0.0, 0.0, offset])),
                quaternion,
            ))
            for quaternion, offset in zip(history, offsets)
        ])
    return result
