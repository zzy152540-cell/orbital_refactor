from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.network_schmidt_runner import (
    NetworkModuleOutput,
    NetworkRuntimeDiagnostics,
)
from experiments.v14_exact_transport_scale_scan import _build_case
from experiments.v14_three_satellite_local_observation import _three_satellite_scenario
from orbital_core.ci_fusion import ci_fuse_posteriors
from orbital_core.dynamics import (
    accel_two_body_j2,
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from orbital_core.quality import quality_score_from_covariance
from orbital_core.measurement_integrity import MeasurementIntegrityPolicy
from interfaces.data_objects import FusionStatus, ModuleOutput, RuntimeStatus, StateOutput
from orbital_core.metrics import compute_nees_history, compute_rmse
from orbital_core.inter_satellite_model import RelativeMeasurementModel
from scenarios.measurement_visibility import VisibilityConfig


NEES_95_DOF6 = (1.2373442458, 14.4493753354)
SENSOR_MODALITIES = ("RADAR", "INFRARED", "OPTICAL")


@dataclass(frozen=True)
class SchmidtArchitectureSummary:
    architecture: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_position_covariance_trace: float
    mean_position_rmse_by_node: dict[str, float]
    mean_runtime_seconds: float


@dataclass(frozen=True)
class FederatedSchmidtCIResult:
    sequential_schmidt: SchmidtArchitectureSummary
    federated_ci: SchmidtArchitectureSummary
    local_by_modality: dict[str, SchmidtArchitectureSummary]
    mean_ci_weight_by_modality: dict[str, float]
    mean_ci_weight_by_node_and_modality: dict[str, dict[str, float]]
    ci_objective: str
    ci_grid_points: int
    gated_observation_count_by_modality: dict[str, int]
    ci_exclusion_count_by_modality: dict[str, int]
    ci_prediction_only_exclusion_count_by_modality: dict[str, int]
    ci_all_modalities_unavailable_count: int
    representative_module_output_by_node: dict[str, NetworkModuleOutput]
    phase_summary_by_architecture: dict[
        str, dict[str, SchmidtArchitectureSummary]
    ]


def run_v14_three_satellite_federated_schmidt_ci_experiment(
    *,
    seeds: int = 10,
    duration: float = 120.0,
    dt: float = 2.0,
    maximum_range: float = 5000.0,
    range_sigma: float = 2.0,
    range_rate_sigma: float = 0.05,
    az_el_sigma: float = np.deg2rad(0.05),
    optical_sigma: float = 1e-3,
    absolute_sigma: float = 3.0,
    absolute_navigation_dropout_windows: tuple[
        tuple[float, float], ...
    ] = (),
    process_noise_acceleration: float = 1e-8,
    ci_objective: str = "trace",
    ci_grid_points: int = 31,
    radar_actual_noise_scale: float = 1.0,
    optical_outlier_bias: tuple[float, float] | None = None,
    optical_outlier_window: tuple[float, float] | None = None,
    infrared_outlier_bias: tuple[float, float] | None = None,
    infrared_outlier_window: tuple[float, float] | None = None,
    dropout_windows_by_modality: dict[
        str, tuple[tuple[float, float], ...]
    ] | None = None,
    nis_gate_threshold_by_modality: dict[str, float] | None = None,
    nis_inflation_threshold_by_modality: dict[str, float] | None = None,
    maximum_measurement_covariance_scale_by_modality: dict[str, float] | None = None,
    integrity_policy_by_modality: dict[
        str, MeasurementIntegrityPolicy
    ] | None = None,
) -> FederatedSchmidtCIResult:
    """Compare sequential Schmidt updates with output-only local federated CI.

    Each modality owns an independent Schmidt/exact-replay filter. CI fuses
    only the three six-state active posteriors for the same physical satellite.
    The fused result is an output and is never fed back into the local tracks.
    """

    if seeds < 1:
        raise ValueError("seeds must be at least one.")
    if ci_grid_points < 2:
        raise ValueError("ci_grid_points must be at least two.")
    if radar_actual_noise_scale <= 0.0:
        raise ValueError("radar_actual_noise_scale must be positive.")
    if optical_outlier_bias is not None and optical_outlier_window is None:
        raise ValueError("optical_outlier_bias requires optical_outlier_window.")
    if infrared_outlier_bias is not None and infrared_outlier_window is None:
        raise ValueError("infrared_outlier_bias requires infrared_outlier_window.")
    _validate_fault_windows(
        optical_outlier_window=optical_outlier_window,
        infrared_outlier_window=infrared_outlier_window,
        dropout_windows_by_modality=dropout_windows_by_modality,
    )
    for start, end in absolute_navigation_dropout_windows:
        if (
            not np.isfinite(start) or not np.isfinite(end) or end < start
        ):
            raise ValueError(
                "Absolute-navigation dropout windows require finite start <= end."
            )
    gate_thresholds = {
        str(modality): float(threshold)
        for modality, threshold in (nis_gate_threshold_by_modality or {}).items()
    }
    if set(gate_thresholds) - set(SENSOR_MODALITIES):
        raise ValueError("NIS gate thresholds reference unsupported modalities.")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in gate_thresholds.values()
    ):
        raise ValueError("NIS gate thresholds must be finite and positive.")
    inflation_thresholds = {
        str(modality): float(threshold)
        for modality, threshold in (
            nis_inflation_threshold_by_modality or {}
        ).items()
    }
    maximum_covariance_scales = {
        str(modality): float(scale)
        for modality, scale in (
            maximum_measurement_covariance_scale_by_modality or {}
        ).items()
    }
    if set(inflation_thresholds) - set(SENSOR_MODALITIES):
        raise ValueError("NIS inflation thresholds reference unsupported modalities.")
    if set(maximum_covariance_scales) - set(SENSOR_MODALITIES):
        raise ValueError("Maximum covariance scales reference unsupported modalities.")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in inflation_thresholds.values()
    ):
        raise ValueError("NIS inflation thresholds must be finite and positive.")
    if any(
        not np.isfinite(value) or value < 1.0
        for value in maximum_covariance_scales.values()
    ):
        raise ValueError("Maximum covariance scales must be finite and at least one.")
    integrity_policies = dict(integrity_policy_by_modality or {})
    if set(integrity_policies) - set(SENSOR_MODALITIES):
        raise ValueError("Integrity policies reference unsupported modalities.")
    if not all(
        isinstance(value, MeasurementIntegrityPolicy)
        for value in integrity_policies.values()
    ):
        raise TypeError("Integrity policies must be MeasurementIntegrityPolicy values.")
    overlapping = set(integrity_policies) & (
        set(gate_thresholds) | set(inflation_thresholds)
        | set(maximum_covariance_scales)
    )
    if overlapping:
        raise ValueError(
            "A modality cannot use both an integrity policy and legacy thresholds."
        )
    hard_thresholds = dict(gate_thresholds)
    hard_thresholds.update({
        modality: float(policy.hard_gate_threshold)
        for modality, policy in integrity_policies.items()
        if policy.hard_gate_threshold is not None
    })
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    scenario = _three_satellite_scenario(timestamps)
    truth = scenario.truth_state_history_by_node
    initial_truth = {node: values[0] for node, values in truth.items()}
    visibility = {
        modality: VisibilityConfig(maximum_range=maximum_range)
        for modality in SENSOR_MODALITIES
    }
    collected = {
        "sequential_schmidt": [],
        "federated_ci": [],
        **{f"local_{modality}": [] for modality in SENSOR_MODALITIES},
    }
    weights = {modality: [] for modality in SENSOR_MODALITIES}
    weights_by_node = {
        node: {modality: [] for modality in SENSOR_MODALITIES}
        for node in scenario.node_ids
    }
    gated_counts = {modality: 0 for modality in SENSOR_MODALITIES}
    exclusion_counts = {modality: 0 for modality in SENSOR_MODALITIES}
    prediction_only_exclusion_counts = {
        modality: 0 for modality in SENSOR_MODALITIES
    }
    all_modalities_unavailable_count = 0
    representative_outputs = {}
    phase_masks = _absolute_navigation_phase_masks(
        timestamps, absolute_navigation_dropout_windows
    )
    phase_collected = {
        architecture: {phase: [] for phase in phase_masks}
        for architecture in ("sequential_schmidt", "federated_ci")
    }

    for seed in range(seeds):
        case = _build_case(
            seed=seed, duration=duration, dt=dt,
            range_sigma=range_sigma, range_rate_sigma=range_rate_sigma,
            az_el_sigma=az_el_sigma, optical_sigma=optical_sigma,
            absolute_sigma=absolute_sigma,
            absolute_navigation_dropout_windows=(
                absolute_navigation_dropout_windows
            ),
            process_noise_acceleration=process_noise_acceleration,
            packet_loss=0.0, delay=0.0, acknowledge_messages=True,
            node_count=3, topology_type="ring",
            visibility_by_modality=visibility,
            truth_initial_state_by_node=initial_truth,
            relative_modalities=SENSOR_MODALITIES,
        )
        case["observations"] = _apply_observation_faults(
            case["observations"], truth=case["truth"], seed=seed,
            radar_actual_noise_scale=radar_actual_noise_scale,
            optical_outlier_bias=optical_outlier_bias,
            optical_outlier_window=optical_outlier_window,
            infrared_outlier_bias=infrared_outlier_bias,
            infrared_outlier_window=infrared_outlier_window,
            dropout_windows_by_modality=dropout_windows_by_modality,
        )
        sequential_started = perf_counter()
        sequential = _run_case(
            case, observations=case["observations"],
            process_noise_acceleration=process_noise_acceleration,
            nis_gate_threshold_by_modality=gate_thresholds,
            nis_inflation_threshold_by_modality=inflation_thresholds,
            maximum_measurement_covariance_scale_by_modality=(
                maximum_covariance_scales
            ),
            integrity_policy_by_modality=integrity_policies,
        )
        sequential_seconds = perf_counter() - sequential_started
        collected["sequential_schmidt"].append(
            _history_metrics(sequential, case["truth"], sequential_seconds)
        )
        for phase, mask in phase_masks.items():
            phase_collected["sequential_schmidt"][phase].append(
                _array_metrics(
                    sequential.active_state_history_by_node,
                    sequential.active_covariance_history_by_node,
                    case["truth"], runtime_seconds=0.0, sample_mask=mask,
                )
            )

        local_histories = {}
        local_seconds = 0.0
        for modality in SENSOR_MODALITIES:
            started = perf_counter()
            local_histories[modality] = _run_case(
                case,
                observations=[
                    item for item in case["observations"]
                    if item.modality == modality
                ],
                process_noise_acceleration=process_noise_acceleration,
                nis_gate_threshold_by_modality=gate_thresholds,
                nis_inflation_threshold_by_modality=inflation_thresholds,
                maximum_measurement_covariance_scale_by_modality=(
                    maximum_covariance_scales
                ),
                integrity_policy_by_modality=integrity_policies,
            )
            elapsed = perf_counter() - started
            local_seconds += elapsed
            collected[f"local_{modality}"].append(
                _history_metrics(
                    local_histories[modality], case["truth"], elapsed
                )
            )
            threshold = hard_thresholds.get(modality)
            if threshold is not None:
                gated_counts[modality] += sum(
                    value > threshold
                    for node_id, epochs in local_histories[
                        modality
                    ].nis_history_by_node.items()
                    for index, epoch in enumerate(epochs)
                    for information_id, value in epoch.items()
                    if local_histories[
                        modality
                    ].modality_history_by_node[node_id][index].get(
                        information_id
                    ) == modality
                )

        ci_started = perf_counter()
        fused_state = {
            node: np.zeros((len(timestamps), 6), dtype=float)
            for node in scenario.node_ids
        }
        fused_covariance = {
            node: np.zeros((len(timestamps), 6, 6), dtype=float)
            for node in scenario.node_ids
        }
        fusion_weights_by_node = {node: [] for node in scenario.node_ids}
        modality_validity_by_node = {node: [] for node in scenario.node_ids}
        for node in scenario.node_ids:
            for index in range(len(timestamps)):
                participating = []
                for modality in SENSOR_MODALITIES:
                    threshold = hard_thresholds.get(modality)
                    epoch_nis = {
                        information_id: value
                        for information_id, value in local_histories[
                            modality
                        ].nis_history_by_node[node][index].items()
                        if local_histories[
                            modality
                        ].modality_history_by_node[node][index].get(
                            information_id
                        ) == modality
                    }
                    if not epoch_nis:
                        prediction_only_exclusion_counts[modality] += 1
                        continue
                    accepted = any(
                        threshold is None or value <= threshold
                        for value in epoch_nis.values()
                    )
                    if not accepted:
                        exclusion_counts[modality] += 1
                        continue
                    participating.append((
                        modality,
                        local_histories[modality]
                        .active_state_history_by_node[node][index],
                        local_histories[modality]
                        .active_covariance_history_by_node[node][index],
                    ))
                if not participating:
                    all_modalities_unavailable_count += 1
                    navigation_track = local_histories[SENSOR_MODALITIES[0]]
                    navigation_ids = {
                        information_id
                        for information_id, tracked_modality in (
                            navigation_track.modality_history_by_node[
                                node
                            ][index].items()
                        )
                        if tracked_modality == "ABSOLUTE_POSITION"
                    }
                    if navigation_ids:
                        fused_state[node][index] = local_histories[
                            SENSOR_MODALITIES[0]
                        ].active_state_history_by_node[node][index]
                        fused_covariance[node][index] = local_histories[
                            SENSOR_MODALITIES[0]
                        ].active_covariance_history_by_node[node][index]
                    elif index > 0:
                        delta_time = float(
                            timestamps[index] - timestamps[index - 1]
                        )
                        previous_state = fused_state[node][index - 1]
                        transition = numerical_jacobian_discrete(
                            lambda value: rk4_step_absolute(value, delta_time),
                            previous_state,
                        )
                        fused_state[node][index] = rk4_step_absolute(
                            previous_state, delta_time
                        )
                        fused_covariance[node][index] = (
                            transition @ fused_covariance[node][index - 1]
                            @ transition.T
                            + make_process_noise(
                                delta_time, process_noise_acceleration
                            )
                        )
                    else:
                        fused_state[node][index] = navigation_track.active_state_history_by_node[
                            node
                        ][index]
                        fused_covariance[node][index] = navigation_track.active_covariance_history_by_node[
                            node
                        ][index]
                    fusion_weights_by_node[node].append({})
                    modality_validity_by_node[node].append({
                        modality: False for modality in SENSOR_MODALITIES
                    })
                    continue
                fusion = ci_fuse_posteriors(
                    participating,
                    objective=ci_objective,
                    grid_points=ci_grid_points,
                )
                fused_state[node][index] = fusion.state
                fused_covariance[node][index] = fusion.covariance
                fusion_weights_by_node[node].append(dict(fusion.weights))
                modality_validity_by_node[node].append({
                    modality: any(
                        item[0] == modality for item in participating
                    )
                    for modality in SENSOR_MODALITIES
                })
                for modality in SENSOR_MODALITIES:
                    weight = fusion.weights.get(modality, 0.0)
                    weights[modality].append(weight)
                    weights_by_node[node][modality].append(
                        weight
                    )
        ci_seconds = perf_counter() - ci_started
        collected["federated_ci"].append(
            _array_metrics(
                fused_state, fused_covariance, case["truth"],
                runtime_seconds=local_seconds + ci_seconds,
            )
        )
        for phase, mask in phase_masks.items():
            phase_collected["federated_ci"][phase].append(
                _array_metrics(
                    fused_state, fused_covariance, case["truth"],
                    runtime_seconds=0.0, sample_mask=mask,
                )
            )
        representative_outputs = _federated_module_outputs(
            timestamps=timestamps,
            fused_state_by_node=fused_state,
            fused_covariance_by_node=fused_covariance,
            fusion_weights_by_node=fusion_weights_by_node,
            modality_validity_by_node=modality_validity_by_node,
            local_histories=local_histories,
            processing_time=local_seconds + ci_seconds,
        )

    return FederatedSchmidtCIResult(
        sequential_schmidt=_aggregate(
            "sequential_schmidt", collected["sequential_schmidt"]
        ),
        federated_ci=_aggregate("federated_ci", collected["federated_ci"]),
        local_by_modality={
            modality: _aggregate(
                f"local_{modality.lower()}", collected[f"local_{modality}"]
            )
            for modality in SENSOR_MODALITIES
        },
        mean_ci_weight_by_modality={
            modality: float(np.mean(values)) if values else 0.0
            for modality, values in weights.items()
        },
        mean_ci_weight_by_node_and_modality={
            node: {
                modality: float(np.mean(values)) if values else 0.0
                for modality, values in by_modality.items()
            }
            for node, by_modality in weights_by_node.items()
        },
        ci_objective=ci_objective,
        ci_grid_points=ci_grid_points,
        gated_observation_count_by_modality=gated_counts,
        ci_exclusion_count_by_modality=exclusion_counts,
        ci_prediction_only_exclusion_count_by_modality=(
            prediction_only_exclusion_counts
        ),
        ci_all_modalities_unavailable_count=all_modalities_unavailable_count,
        representative_module_output_by_node=representative_outputs,
        phase_summary_by_architecture={
            architecture: {
                phase: _aggregate(
                    f"{architecture}_{phase}", values
                )
                for phase, values in by_phase.items()
            }
            for architecture, by_phase in phase_collected.items()
        },
    )


def _federated_module_outputs(
    *, timestamps, fused_state_by_node, fused_covariance_by_node,
    fusion_weights_by_node, modality_validity_by_node, local_histories,
    processing_time,
):
    outputs = {}
    node_ids = list(fused_state_by_node)
    local_outputs = {
        modality: history.to_module_outputs(processing_time=processing_time)
        for modality, history in local_histories.items()
    }
    for node_id in node_ids:
        state = fused_state_by_node[node_id][-1]
        covariance = fused_covariance_by_node[node_id][-1]
        weights = fusion_weights_by_node[node_id][-1]
        valid_flags = modality_validity_by_node[node_id][-1]
        active_modalities = [
            modality for modality, valid in valid_flags.items() if valid
        ]
        abnormal_events = [
            event
            for modality in SENSOR_MODALITIES
            for event in local_outputs[modality][
                node_id
            ].module_output.abnormal_events
        ]
        local_diagnostics = [
            local_outputs[modality][node_id].network_diagnostics
            for modality in SENSOR_MODALITIES
        ]
        observation_count = sum(
            local_outputs[modality][
                node_id
            ].module_output.runtime_status.observation_count
            for modality in SENSOR_MODALITIES
        )
        status = (
            "DEGRADED" if any(
                event.severity == "ERROR" for event in abnormal_events
            )
            else "OK" if active_modalities
            else "NAVIGATION_ONLY" if any(
                tracked_modality == "ABSOLUTE_POSITION"
                for tracked_modality in local_histories[
                    SENSOR_MODALITIES[0]
                ].modality_history_by_node[node_id][-1].values()
            )
            else "PREDICTION_ONLY"
        )
        module_output = ModuleOutput(
            state_output=StateOutput(
                timestamp=float(timestamps[-1]), target_id=node_id,
                position_estimate=state[:3].copy(),
                velocity_estimate=state[3:].copy(),
                acceleration_estimate=accel_two_body_j2(state[:3]),
                covariance=covariance.copy(), valid_flag=True,
                confidence_level=quality_score_from_covariance(covariance),
            ),
            fusion_status=FusionStatus(
                modality_weights=dict(weights),
                modality_valid_flags=dict(valid_flags),
                active_nodes=node_ids,
            ),
            abnormal_events=abnormal_events,
            runtime_status=RuntimeStatus(
                processing_time=float(processing_time),
                observation_count=observation_count,
                active_modality_count=len(active_modalities),
                active_node_count=len(node_ids), status=status,
            ),
        )
        diagnostics = NetworkRuntimeDiagnostics(
            node_id=node_id,
            neighbor_count=local_diagnostics[0].neighbor_count,
            replay_count=sum(item.replay_count for item in local_diagnostics),
            replay_batch_count=sum(
                item.replay_batch_count for item in local_diagnostics
            ),
            replay_fallback_count=sum(
                item.replay_fallback_count for item in local_diagnostics
            ),
            maximum_replay_seconds=max(
                item.maximum_replay_seconds for item in local_diagnostics
            ),
            maximum_retained_journal_count=max(
                item.maximum_retained_journal_count
                for item in local_diagnostics
            ),
            configured_neighbors=local_diagnostics[0].configured_neighbors,
            link_health_by_neighbor=dict(
                local_diagnostics[0].link_health_by_neighbor
            ),
            last_receive_timestamp_by_neighbor=dict(
                local_diagnostics[0].last_receive_timestamp_by_neighbor
            ),
            losses_before_last_delivery_by_neighbor=dict(
                local_diagnostics[0].losses_before_last_delivery_by_neighbor
            ),
            resynchronization_required_by_neighbor=dict(
                local_diagnostics[0].resynchronization_required_by_neighbor
            ),
            message_rejection_counts_by_reason={
                reason: sum(
                    item.message_rejection_counts_by_reason.get(reason, 0)
                    for item in local_diagnostics
                )
                for reason in {
                    key for item in local_diagnostics
                    for key in item.message_rejection_counts_by_reason
                }
            },
            maximum_checkpoint_count=max(
                item.maximum_checkpoint_count for item in local_diagnostics
            ),
            maximum_pinned_checkpoint_count=max(
                item.maximum_pinned_checkpoint_count
                for item in local_diagnostics
            ),
            maximum_resync_required_count=max(
                item.maximum_resync_required_count
                for item in local_diagnostics
            ),
        )
        outputs[node_id] = NetworkModuleOutput(module_output, diagnostics)
    return outputs


def _run_case(
    case, *, observations, process_noise_acceleration,
    nis_gate_threshold_by_modality,
    nis_inflation_threshold_by_modality,
    maximum_measurement_covariance_scale_by_modality,
    integrity_policy_by_modality,
):
    return run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        observation_messages=observations,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=process_noise_acceleration,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
        nis_gate_threshold_by_modality=nis_gate_threshold_by_modality,
        nis_inflation_threshold_by_modality=nis_inflation_threshold_by_modality,
        maximum_measurement_covariance_scale_by_modality=(
            maximum_measurement_covariance_scale_by_modality
        ),
        integrity_policy_by_modality=integrity_policy_by_modality,
    )


def _history_metrics(history, truth, runtime_seconds):
    return _array_metrics(
        history.active_state_history_by_node,
        history.active_covariance_history_by_node,
        truth,
        runtime_seconds=runtime_seconds,
    )


def _array_metrics(
    states, covariances, truth, *, runtime_seconds, sample_mask=None,
):
    position_errors = []
    velocity_errors = []
    nees = []
    position_traces = []
    position_by_node = {}
    if sample_mask is None:
        sample_mask = slice(None)
    for node in truth:
        error = (states[node] - truth[node])[sample_mask]
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        position_by_node[node] = compute_rmse(error[:, :3])
        nees.extend(compute_nees_history(
            states[node][sample_mask], truth[node][sample_mask],
            covariances[node][sample_mask],
        ))
        position_traces.extend(
            np.trace(
                covariances[node][sample_mask, :3, :3], axis1=1, axis2=2
            )
        )
    nees_array = np.asarray(nees, dtype=float)
    return {
        "position_rmse": compute_rmse(np.vstack(position_errors)),
        "velocity_rmse": compute_rmse(np.vstack(velocity_errors)),
        "nees": float(np.mean(nees_array)),
        "nees_coverage": float(np.mean(
            (nees_array >= NEES_95_DOF6[0])
            & (nees_array <= NEES_95_DOF6[1])
        )),
        "position_trace": float(np.mean(position_traces)),
        "position_by_node": position_by_node,
        "runtime_seconds": float(runtime_seconds),
    }


def _absolute_navigation_phase_masks(timestamps, dropout_windows):
    if not dropout_windows:
        return {}
    timestamps = np.asarray(timestamps, dtype=float)
    first_start = min(float(start) for start, _ in dropout_windows)
    last_end = max(float(end) for _, end in dropout_windows)
    masks = {
        "pre_dropout": timestamps < first_start,
        "dropout": np.asarray([
            any(start <= timestamp <= end for start, end in dropout_windows)
            for timestamp in timestamps
        ], dtype=bool),
        "post_recovery": timestamps > last_end,
    }
    return {name: mask for name, mask in masks.items() if np.any(mask)}


def _aggregate(architecture, values):
    nodes = tuple(values[0]["position_by_node"])
    return SchmidtArchitectureSummary(
        architecture=architecture,
        run_count=len(values),
        mean_position_rmse=float(np.mean([v["position_rmse"] for v in values])),
        mean_velocity_rmse=float(np.mean([v["velocity_rmse"] for v in values])),
        mean_nees=float(np.mean([v["nees"] for v in values])),
        mean_nees_95_coverage=float(np.mean([v["nees_coverage"] for v in values])),
        mean_position_covariance_trace=float(
            np.mean([v["position_trace"] for v in values])
        ),
        mean_position_rmse_by_node={
            node: float(np.mean([v["position_by_node"][node] for v in values]))
            for node in nodes
        },
        mean_runtime_seconds=float(np.mean([v["runtime_seconds"] for v in values])),
    )


def _validate_fault_windows(
    *, optical_outlier_window, infrared_outlier_window,
    dropout_windows_by_modality,
):
    windows = []
    if optical_outlier_window is not None:
        windows.append(optical_outlier_window)
    if infrared_outlier_window is not None:
        windows.append(infrared_outlier_window)
    for modality, values in (dropout_windows_by_modality or {}).items():
        if modality not in SENSOR_MODALITIES:
            raise ValueError(f"Unsupported dropout modality: {modality}")
        windows.extend(values)
    for start, end in windows:
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("Fault windows require finite start <= end.")


def _apply_observation_faults(
    observations, *, truth, seed, radar_actual_noise_scale,
    optical_outlier_bias, optical_outlier_window, dropout_windows_by_modality,
    infrared_outlier_bias, infrared_outlier_window,
):
    rng = np.random.default_rng(20270104 + seed)
    # Truth histories do not carry timestamps, while experiment epochs are
    # uniformly indexed by the ordered observation timestamps.
    ordered_timestamps = sorted({float(item.timestamp) for item in observations})
    timestamp_to_index = {
        timestamp: index for index, timestamp in enumerate(ordered_timestamps)
    }
    result = []
    for observation in observations:
        timestamp = float(observation.timestamp)
        if any(
            start <= timestamp <= end
            for start, end in (dropout_windows_by_modality or {}).get(
                observation.modality, ()
            )
        ):
            continue
        modified = observation
        if observation.modality == "RADAR" and radar_actual_noise_scale != 1.0:
            index = timestamp_to_index[timestamp]
            model = RelativeMeasurementModel("RADAR", observation.frame)
            ideal = model.predict(
                truth[observation.observer_id][index],
                truth[observation.target_id][index],
            )
            actual_noise = rng.multivariate_normal(
                np.zeros(2),
                observation.covariance * radar_actual_noise_scale**2,
            )
            modified = replace(observation, measurement=ideal + actual_noise)
        if (
            observation.modality == "OPTICAL"
            and optical_outlier_bias is not None
            and optical_outlier_window[0] <= timestamp <= optical_outlier_window[1]
        ):
            modified = replace(
                modified,
                measurement=(
                    np.asarray(modified.measurement, dtype=float)
                    + np.asarray(optical_outlier_bias, dtype=float)
                ),
            )
        if (
            observation.modality == "INFRARED"
            and infrared_outlier_bias is not None
            and infrared_outlier_window[0] <= timestamp <= infrared_outlier_window[1]
        ):
            modified = replace(
                modified,
                measurement=(
                    np.asarray(modified.measurement, dtype=float)
                    + np.asarray(infrared_outlier_bias, dtype=float)
                ),
            )
        result.append(modified)
    return result
