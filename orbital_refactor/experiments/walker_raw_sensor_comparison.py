from __future__ import annotations

from dataclasses import replace

import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.dynamics import rk4_step_absolute


def run_walker_infrared_image_comparison(
    *, duration=20.0, dt=2.0, seed=0, camera_config=None,
):
    """Compare BODY analytic angles with focal-plane centroid angles."""
    config = camera_config or InfraredCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=300.0, focal_length_y_pixels=300.0,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.26,
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    analytic, image = replace_infrared_messages_with_body_pair(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"], config=config,
        analytic_rng=np.random.default_rng(20261330 + seed),
        image_rng=np.random.default_rng(20261430 + seed),
    )
    analytic_history = _run(case, analytic)
    image_history = _run(case, image)
    return _comparison_summary(
        case, analytic, image, analytic_history, image_history,
        modality="INFRARED", raw_label="image",
    )


def run_walker_radar_map_comparison(
    *, duration=20.0, dt=2.0, seed=0, map_config=None,
):
    """Compare analytic radar with a truth-centered calibration power map."""
    config = map_config or RadarRangeDopplerConfig(
        width=129, height=129, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    mapped = replace_radar_messages_with_power_maps(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"], config=config,
        rng=np.random.default_rng(20261530 + seed),
    )
    analytic_history = _run(case, case["observations"])
    mapped_history = _run(case, mapped)
    result = _comparison_summary(
        case, case["observations"], mapped,
        analytic_history, mapped_history,
        modality="RADAR", raw_label="range_doppler",
    )
    return {**result, "acquisition_center_mode": "truth_calibration"}


def run_walker_radar_predicted_center_comparison(
    *, duration=20.0, dt=2.0, seed=0, map_config=None,
    initial_position_sigma=10.0, initial_velocity_sigma=0.02,
    absolute_navigation_dropout_duration=0.0,
):
    """Use only the preceding filter posterior to center each radar map.

    The analytic run supplies a tracker history for this staged front-end
    experiment.  At epoch k, acquisition uses the posterior at k-1 propagated
    to k; no state from the current epoch is used.
    """
    config = map_config or RadarRangeDopplerConfig(
        width=129, height=129, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )
    case = _walker_case(
        seed=seed, duration=duration, dt=dt,
        initial_position_sigma=initial_position_sigma,
        initial_velocity_sigma=initial_velocity_sigma,
        absolute_navigation_dropout_duration=(
            absolute_navigation_dropout_duration
        ),
    )
    analytic_history = _run(case, case["observations"])
    centers = build_lagged_radar_acquisition_centers(
        case, analytic_history,
    )
    mapped = replace_radar_messages_with_power_maps(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"], config=config,
        rng=np.random.default_rng(20261630 + seed),
        acquisition_center_by_message_id=centers,
        acquisition_center_mode="lagged_filter_prediction",
    )
    mapped_history = _run(case, mapped)
    result = _comparison_summary(
        case, case["observations"], mapped,
        analytic_history, mapped_history,
        modality="RADAR", raw_label="range_doppler",
    )
    return {
        **result,
        "acquisition_center_mode": "lagged_filter_prediction",
        "acquisition_center_component_rmse": (
            _radar_acquisition_center_component_rmse(mapped, case)
        ),
    }


def run_walker_radar_acquisition_boundary_sweep(
    *, duration=20.0, dt=2.0, seed=0, offsets=None, map_config=None,
):
    """Scan controlled prediction offsets against a narrowed radar window."""
    config = map_config or RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )
    selected_offsets = offsets or (
        (0.0, 0.0),
        (128.0, 0.0),
        (200.0, 0.0),
        (216.0, 0.0),
        (224.0, 0.0),
        (0.0, 3.2),
        (0.0, 5.2),
        (0.0, 5.4),
        (0.0, 5.6),
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    analytic_history = _run(case, case["observations"])
    nominal_centers = build_lagged_radar_acquisition_centers(
        case, analytic_history,
    )
    records = []
    for scan_index, offset in enumerate(selected_offsets):
        offset_vector = np.asarray(offset, dtype=float).reshape(2)
        centers = {
            message_id: center + offset_vector
            for message_id, center in nominal_centers.items()
        }
        mapped = replace_radar_messages_with_power_maps(
            case["observations"], timestamps=case["timestamps"],
            truth_state_history_by_node=case["truth"], config=config,
            rng=np.random.default_rng(20261730 + seed * 100 + scan_index),
            acquisition_center_by_message_id=centers,
            acquisition_center_mode="lagged_prediction_with_offset",
        )
        mapped_history = _run(case, mapped)
        result = _comparison_summary(
            case, case["observations"], mapped,
            analytic_history, mapped_history,
            modality="RADAR", raw_label="range_doppler",
        )
        count = result["message_count"]
        records.append({
            **result,
            "range_offset_m": float(offset_vector[0]),
            "range_rate_offset_mps": float(offset_vector[1]),
            "capture_rate": result["detection_count"] / count,
            "window_half_span_m": (
                config.principal_x * config.range_bin_size_m
            ),
            "window_half_span_mps": (
                config.principal_y * config.range_rate_bin_size_mps
            ),
            "usable_half_span_m": (
                config.principal_x
                - config.centroid_edge_margin_sigma * config.psf_sigma_pixels
            ) * config.range_bin_size_m,
            "usable_half_span_mps": (
                config.principal_y
                - config.centroid_edge_margin_sigma * config.psf_sigma_pixels
            ) * config.range_rate_bin_size_mps,
        })
    return tuple(records)


def run_walker_radar_reacquisition_comparison(
    *, duration=20.0, dt=2.0, seed=0,
    initial_offset=(224.0, 0.0), tracking_config=None,
    reacquisition_config=None, maximum_reacquisition_scale=4.0,
    initial_position_sigma=10.0, initial_velocity_sigma=0.02,
    absolute_navigation_dropout_duration=0.0,
):
    """Exercise fail, expanded-window reacquisition, and tracking recovery."""
    tracking = tracking_config or RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )
    reacquisition = reacquisition_config or replace(
        tracking,
        range_bin_size_m=2.0 * tracking.range_bin_size_m,
        range_rate_bin_size_mps=2.0 * tracking.range_rate_bin_size_mps,
    )
    case = _walker_case(
        seed=seed, duration=duration, dt=dt,
        initial_position_sigma=initial_position_sigma,
        initial_velocity_sigma=initial_velocity_sigma,
        absolute_navigation_dropout_duration=(
            absolute_navigation_dropout_duration
        ),
    )
    analytic_history = _run(case, case["observations"])
    nominal_centers = build_lagged_radar_acquisition_centers(
        case, analytic_history,
    )
    mapped = replace_radar_messages_with_reacquisition(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"],
        acquisition_center_by_message_id=nominal_centers,
        initial_offset=np.asarray(initial_offset, dtype=float),
        tracking_config=tracking, reacquisition_config=reacquisition,
        maximum_reacquisition_scale=maximum_reacquisition_scale,
        rng=np.random.default_rng(20261830 + seed),
    )
    mapped_history = _run(case, mapped)
    result = _comparison_summary(
        case, case["observations"], mapped,
        analytic_history, mapped_history,
        modality="RADAR", raw_label="range_doppler",
    )
    radar_messages = [
        item for item in mapped if item.modality.upper() == "RADAR"
    ]
    first_timestamp = float(case["timestamps"][0])
    first_attempts = [
        item for item in radar_messages
        if np.isclose(item.timestamp, first_timestamp)
    ]
    recovery_messages = [
        item for item in radar_messages
        if item.metadata["acquisition_search_mode"] == "reacquisition"
        and item.valid_flag
    ]
    initial_missed_links = {
        (item.observer_id, item.target_id)
        for item in first_attempts if not item.valid_flag
    }
    first_recovery_by_link = {}
    for item in recovery_messages:
        link = (item.observer_id, item.target_id)
        first_recovery_by_link.setdefault(link, float(item.timestamp))
    final_timestamp = float(case["timestamps"][-1])
    final_attempts = [
        item for item in radar_messages
        if np.isclose(item.timestamp, final_timestamp)
    ]
    initial_position_rmse, initial_velocity_rmse = _initial_state_rmse(case)
    return {
        **result,
        "acquisition_center_mode": "fail_expand_reacquire",
        "initial_offset": tuple(float(value) for value in initial_offset),
        "initial_position_rmse_m": initial_position_rmse,
        "initial_velocity_rmse_mps": initial_velocity_rmse,
        "initial_capture_rate": (
            sum(item.valid_flag for item in first_attempts)
            / len(first_attempts)
        ),
        "final_capture_rate": (
            sum(item.valid_flag for item in final_attempts)
            / len(final_attempts)
        ),
        "tracking_attempt_count": sum(
            item.metadata["acquisition_search_mode"] == "tracking"
            for item in radar_messages
        ),
        "reacquisition_attempt_count": sum(
            item.metadata["acquisition_search_mode"] == "reacquisition"
            for item in radar_messages
        ),
        "reacquisition_success_count": sum(
            item.metadata["acquisition_search_mode"] == "reacquisition"
            and item.valid_flag for item in radar_messages
        ),
        "maximum_consecutive_acquisition_failures": max(
            item.metadata["consecutive_acquisition_failures"]
            for item in radar_messages
        ),
        "recovered_link_count": len(
            initial_missed_links & set(first_recovery_by_link)
        ),
        "unrecovered_initial_link_count": len(
            initial_missed_links - set(first_recovery_by_link)
        ),
        "maximum_first_recovery_latency_s": (
            max(
                first_recovery_by_link[link] - first_timestamp
                for link in initial_missed_links
                if link in first_recovery_by_link
            )
            if initial_missed_links & set(first_recovery_by_link) else 0.0
        ),
        "refresh_diagnostics": dict(mapped_history.refresh_diagnostics),
    }


def replace_radar_messages_with_reacquisition(
    messages, *, timestamps, truth_state_history_by_node,
    acquisition_center_by_message_id, initial_offset,
    tracking_config, reacquisition_config, rng,
    maximum_reacquisition_scale=4.0,
):
    """Apply a per-directed-link two-state radar acquisition controller."""
    index_by_time = _index_by_time(timestamps)
    offset = np.asarray(initial_offset, dtype=float).reshape(2)
    link_state = {}
    result = []
    for message in messages:
        if message.modality.upper() != "RADAR":
            result.append(message)
            continue
        link = (str(message.observer_id), str(message.target_id))
        state = link_state.setdefault(
            link, {"acquired": False, "consecutive_failures": 0},
        )
        search_mode = (
            "reacquisition"
            if state["consecutive_failures"] > 0 else "tracking"
        )
        if search_mode == "tracking":
            config = tracking_config
        elif state["consecutive_failures"] == 1:
            config = reacquisition_config
        else:
            scale = min(
                float(maximum_reacquisition_scale),
                2.0 ** state["consecutive_failures"],
            )
            config = replace(
                tracking_config,
                range_bin_size_m=(
                    scale * tracking_config.range_bin_size_m
                ),
                range_rate_bin_size_mps=(
                    scale * tracking_config.range_rate_bin_size_mps
                ),
            )
        center = np.asarray(
            acquisition_center_by_message_id[message.message_id], dtype=float,
        )
        if not state["acquired"]:
            center = center + offset
        index = index_by_time[float(message.timestamp)]
        observer = truth_state_history_by_node[message.observer_id][index]
        target = truth_state_history_by_node[message.target_id][index]
        frame = render_radar_range_doppler_frame(
            timestamp=message.timestamp,
            observer_id=message.observer_id, target_id=message.target_id,
            observer_state=observer, target_state=target,
            acquisition_range_m=center[0],
            acquisition_range_rate_mps=center[1],
            config=config, rng=rng,
        )
        extracted = radar_frame_to_observation_message(frame, config=config)
        if extracted.valid_flag:
            state["acquired"] = True
            state["consecutive_failures"] = 0
        else:
            state["consecutive_failures"] += 1
        extracted = replace(extracted, metadata={
            **extracted.metadata,
            "acquisition_center_mode": "fail_expand_reacquire",
            "acquisition_search_mode": search_mode,
            "acquisition_window_scale": (
                config.range_bin_size_m / tracking_config.range_bin_size_m
            ),
            "consecutive_acquisition_failures": (
                state["consecutive_failures"]
            ),
        })
        result.append(_preserve_message_identity(message, extracted))
    return result


def replace_infrared_messages_with_body_pair(
    messages, *, timestamps, truth_state_history_by_node, config,
    analytic_rng, image_rng,
):
    index_by_time = _index_by_time(timestamps)
    analytic_result = []
    image_result = []
    for message in messages:
        if message.modality.upper() != "INFRARED":
            analytic_result.append(message)
            image_result.append(message)
            continue
        index = index_by_time[float(message.timestamp)]
        observer = truth_state_history_by_node[message.observer_id][index]
        target = truth_state_history_by_node[message.target_id][index]
        quaternion = target_pointing_quaternion(observer, target)
        ideal = measure_relative_az_el(
            observer, target, frame="BODY",
            quaternion_i2b_wxyz=quaternion,
        )
        analytic_result.append(replace(
            message, frame="BODY",
            measurement=ideal + analytic_rng.multivariate_normal(
                np.zeros(2), message.covariance,
            ),
            metadata={**message.metadata, "quaternion_i2b_wxyz": quaternion},
        ))
        frame = render_infrared_point_source_frame(
            timestamp=message.timestamp,
            observer_id=message.observer_id, target_id=message.target_id,
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=quaternion, config=config, rng=image_rng,
        )
        extracted = infrared_frame_to_observation_message(frame, config=config)
        image_result.append(_preserve_message_identity(message, extracted))
    return analytic_result, image_result


def replace_radar_messages_with_power_maps(
    messages, *, timestamps, truth_state_history_by_node, config, rng,
    acquisition_center_by_message_id=None,
    acquisition_center_mode="truth_calibration",
):
    index_by_time = _index_by_time(timestamps)
    result = []
    for message in messages:
        if message.modality.upper() != "RADAR":
            result.append(message)
            continue
        index = index_by_time[float(message.timestamp)]
        observer = truth_state_history_by_node[message.observer_id][index]
        target = truth_state_history_by_node[message.target_id][index]
        truth_center = np.array([
            measure_relative_range(observer, target),
            measure_relative_range_rate(observer, target),
        ])
        center = (acquisition_center_by_message_id or {}).get(
            message.message_id, truth_center,
        )
        frame = render_radar_range_doppler_frame(
            timestamp=message.timestamp,
            observer_id=message.observer_id, target_id=message.target_id,
            observer_state=observer, target_state=target,
            acquisition_range_m=center[0],
            acquisition_range_rate_mps=center[1],
            config=config, rng=rng,
        )
        extracted = radar_frame_to_observation_message(frame, config=config)
        result.append(_preserve_message_identity(
            message,
            replace(extracted, metadata={
                **extracted.metadata,
                "acquisition_center_mode": acquisition_center_mode,
            }),
        ))
    return result


def build_lagged_radar_acquisition_centers(case, history):
    """Build prior-only range/range-rate centers keyed by message identity."""
    index_by_time = _index_by_time(case["timestamps"])
    centers = {}
    for message in case["observations"]:
        if message.modality.upper() != "RADAR":
            continue
        index = index_by_time[float(message.timestamp)]
        observer_id = message.observer_id
        target_id = message.target_id
        if index == 0:
            observer = case["initial_states"][observer_id]
            target = case["initial_states"][target_id]
        else:
            delta_time = float(
                case["timestamps"][index] - case["timestamps"][index - 1]
            )
            observer = rk4_step_absolute(
                history.active_state_history_by_node[observer_id][index - 1],
                delta_time,
            )
            cached_targets = history.neighbor_state_history_by_node.get(
                observer_id, {},
            )
            target_previous = (
                cached_targets[target_id][index - 1]
                if target_id in cached_targets
                else history.active_state_history_by_node[target_id][index - 1]
            )
            target = rk4_step_absolute(target_previous, delta_time)
        centers[message.message_id] = np.array([
            measure_relative_range(observer, target),
            measure_relative_range_rate(observer, target),
        ])
    return centers


def _preserve_message_identity(original, extracted):
    return replace(
        extracted,
        message_id=original.message_id,
        physical_observation_id=original.physical_observation_id,
        source_timestamp=original.source_timestamp,
        arrival_timestamp=original.arrival_timestamp,
        metadata={**original.metadata, **extracted.metadata},
    )


def _walker_case(
    *, seed, duration, dt,
    initial_position_sigma=10.0, initial_velocity_sigma=0.02,
    absolute_navigation_dropout_duration=0.0,
):
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=6000e3,
    )
    dropout_duration = float(absolute_navigation_dropout_duration)
    if not np.isfinite(dropout_duration) or dropout_duration < 0.0:
        raise ValueError(
            "absolute_navigation_dropout_duration must be nonnegative."
        )
    node_dropouts = (
        {
            node: ((0.0, dropout_duration),)
            for node in audit.scenario.node_ids
        }
        if dropout_duration > 0.0 else None
    )
    return build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=6000e3,
        topology=audit.persistent_topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
        initial_position_sigma=initial_position_sigma,
        initial_velocity_sigma=initial_velocity_sigma,
        absolute_navigation_dropout_windows_by_node=node_dropouts,
    )


def _run(case, messages):
    return run_network_schmidt_filter(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"], observation_messages=messages,
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only", process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
    )


def _comparison_summary(
    case, analytic_messages, raw_messages, analytic_history, raw_history,
    *, modality, raw_label,
):
    analytic_errors = _measurement_errors(analytic_messages, case, modality)
    raw_errors = _measurement_errors(raw_messages, case, modality)
    return {
        "modality": modality,
        "raw_label": raw_label,
        "message_count": sum(
            item.modality.upper() == modality for item in raw_messages
        ),
        "detection_count": sum(
            item.modality.upper() == modality and item.valid_flag
            for item in raw_messages
        ),
        "analytic_measurement_rmse": _vector_rmse(analytic_errors),
        "raw_measurement_rmse": _vector_rmse(raw_errors),
        "analytic_measurement_component_rmse": _component_rmse(
            analytic_errors,
        ),
        "raw_measurement_component_rmse": _component_rmse(raw_errors),
        "analytic_position_rmse_m": _position_rmse(
            analytic_history, case["truth"],
        ),
        "raw_position_rmse_m": _position_rmse(raw_history, case["truth"]),
        "analytic_relative_position_rmse_m": _relative_position_rmse(
            analytic_history, case["truth"], case["topology"],
        ),
        "raw_relative_position_rmse_m": _relative_position_rmse(
            raw_history, case["truth"], case["topology"],
        ),
        "analytic_mean_nis": _mean_modality_nis(
            analytic_history, analytic_messages, modality,
        ),
        "raw_mean_nis": _mean_modality_nis(
            raw_history, raw_messages, modality,
        ),
    }


def _measurement_rmse(messages, case, modality):
    return _vector_rmse(_measurement_errors(messages, case, modality))


def _measurement_errors(messages, case, modality):
    index_by_time = _index_by_time(case["timestamps"])
    errors = []
    for message in messages:
        if message.modality.upper() != modality or not message.valid_flag:
            continue
        index = index_by_time[float(message.timestamp)]
        observer = case["truth"][message.observer_id][index]
        target = case["truth"][message.target_id][index]
        if modality == "INFRARED":
            ideal = measure_relative_az_el(
                observer, target, frame=message.frame,
                quaternion_i2b_wxyz=message.metadata.get(
                    "quaternion_i2b_wxyz",
                ),
            )
        else:
            ideal = np.array([
                measure_relative_range(observer, target),
                measure_relative_range_rate(observer, target),
            ])
        errors.append(np.asarray(message.measurement) - ideal)
    return np.vstack(errors) if errors else np.empty((0, 2), dtype=float)


def _vector_rmse(errors):
    if errors.shape[0] == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def _component_rmse(errors):
    if errors.shape[0] == 0:
        return tuple(float("nan") for _ in range(errors.shape[1]))
    values = np.sqrt(np.mean(errors * errors, axis=0))
    return tuple(float(value) for value in values)


def _radar_acquisition_center_component_rmse(messages, case):
    index_by_time = _index_by_time(case["timestamps"])
    errors = []
    for message in messages:
        if message.modality.upper() != "RADAR":
            continue
        index = index_by_time[float(message.timestamp)]
        observer = case["truth"][message.observer_id][index]
        target = case["truth"][message.target_id][index]
        ideal = np.array([
            measure_relative_range(observer, target),
            measure_relative_range_rate(observer, target),
        ])
        errors.append(
            np.asarray(message.metadata["acquisition_center"], dtype=float)
            - ideal
        )
    return _component_rmse(np.vstack(errors))


def _position_rmse(history, truth):
    errors = [
        history.active_state_history_by_node[node][:, :3] - truth[node][:, :3]
        for node in history.node_ids
    ]
    return float(np.sqrt(np.mean([
        np.sum(error * error, axis=1) for error in errors
    ])))


def _relative_position_rmse(history, truth, topology):
    edge_errors = []
    for left in topology.node_ids:
        for right in topology.neighbors(left):
            if str(left) >= str(right):
                continue
            estimated_relative = (
                history.active_state_history_by_node[right][:, :3]
                - history.active_state_history_by_node[left][:, :3]
            )
            truth_relative = truth[right][:, :3] - truth[left][:, :3]
            edge_errors.append(estimated_relative - truth_relative)
    values = np.vstack(edge_errors)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _initial_state_rmse(case):
    errors = np.vstack([
        case["initial_states"][node] - case["truth"][node][0]
        for node in case["initial_states"]
    ])
    return (
        float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
    )


def _mean_modality_nis(history, messages, modality):
    ids = {
        item.information_id for item in messages
        if item.modality.upper() == modality
    }
    values = [
        value
        for node in history.node_ids
        for epoch in history.nis_history_by_node[node]
        for information_id, value in epoch.items()
        if information_id in ids
    ]
    return float(np.mean(values)) if values else float("nan")


def _index_by_time(timestamps):
    return {
        float(value): index for index, value in enumerate(timestamps)
    }
