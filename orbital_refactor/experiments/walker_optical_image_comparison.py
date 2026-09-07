from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from experiments.v14_walker_geometry_audit import run_v14_walker_geometry_audit
from experiments.walker_filter_setup import build_walker_filter_case
from orbital_core.measurements import measure_relative_optical_uv


def run_walker_optical_image_comparison(
    *, duration=20.0, dt=2.0, seed=0, camera_config=None,
    retain_raw_frames=False,
):
    """Compare analytic UV generation with image-centroid UV on Walker 20/10/1."""
    config = camera_config or OpticalCameraConfig(
        width=128, height=128,
        focal_length_x_pixels=100.0, focal_length_y_pixels=100.0,
        source_peak=100.0, background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.25,
    )
    audit = run_v14_walker_geometry_audit(
        total_satellites=20, plane_count=10, phasing=1,
        duration=1800.0, dt=30.0, maximum_range=6000e3,
    )
    case = build_walker_filter_case(
        seed=seed, duration=duration, dt=dt, maximum_range=6000e3,
        topology=audit.persistent_topology,
        truth_history_by_node=audit.scenario.truth_state_history_by_node,
        topology_type="walker_persistent",
    )
    started = perf_counter()
    image_messages, raw_frames = replace_optical_messages_with_images(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"], config=config,
        rng=np.random.default_rng(20261230 + seed),
        retain_raw_frames=retain_raw_frames,
    )
    image_generation_seconds = perf_counter() - started
    run_arguments = dict(
        timestamps=case["timestamps"],
        initial_state_by_node=case["initial_states"],
        initial_covariance_by_node=case["initial_covariances"],
        topology=case["topology"],
        absolute_position_observations=case["absolute_observations"],
        observation_usage="observer_only",
        process_noise_acceleration=1e-8,
        consider_refresh_mode="exact_transport_event_replay",
        state_messages_by_receiver=case["state_messages"],
        replay_history_window=10.0,
        expected_lineage_by_link=case["lineages"],
    )
    started = perf_counter()
    analytic = run_network_schmidt_filter(
        observation_messages=case["observations"], **run_arguments,
    )
    analytic_seconds = perf_counter() - started
    started = perf_counter()
    image = run_network_schmidt_filter(
        observation_messages=image_messages, **run_arguments,
    )
    image_seconds = perf_counter() - started
    return {
        "seed": int(seed),
        "camera_config": config,
        "raw_frames": raw_frames,
        "optical_message_count": sum(
            item.modality == "OPTICAL" for item in image_messages
        ),
        "image_detection_count": sum(
            item.modality == "OPTICAL" and item.valid_flag
            for item in image_messages
        ),
        "analytic_uv_rmse": _uv_rmse(
            case["observations"], case["truth"], case["timestamps"],
        ),
        "image_uv_rmse": _uv_rmse(
            image_messages, case["truth"], case["timestamps"],
        ),
        "analytic_position_rmse_m": _position_rmse(
            analytic, case["truth"],
        ),
        "image_position_rmse_m": _position_rmse(image, case["truth"]),
        "analytic_optical_mean_nis": _mean_optical_nis(
            analytic, case["observations"],
        ),
        "image_optical_mean_nis": _mean_optical_nis(
            image, image_messages,
        ),
        "analytic_filter_seconds": float(analytic_seconds),
        "image_generation_seconds": float(image_generation_seconds),
        "image_filter_seconds": float(image_seconds),
    }


def replace_optical_messages_with_images(
    messages, *, timestamps, truth_state_history_by_node, config, rng,
    retain_raw_frames=False,
):
    """Replace only OPTICAL measurements while preserving message lineage."""
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    index_by_time = {float(value): index for index, value in enumerate(times)}
    result = []
    frames = {}
    for message in messages:
        if message.modality.upper() != "OPTICAL":
            result.append(message)
            continue
        try:
            index = index_by_time[float(message.timestamp)]
        except KeyError as exc:
            raise ValueError("Optical message timestamp is outside the truth timeline.") from exc
        quaternion = np.asarray(
            message.metadata["quaternion_i2b_wxyz"], dtype=float,
        )
        frame = render_optical_point_source_frame(
            timestamp=message.timestamp,
            observer_id=message.observer_id, target_id=message.target_id,
            observer_state=truth_state_history_by_node[message.observer_id][index],
            target_state=truth_state_history_by_node[message.target_id][index],
            quaternion_i2b_wxyz=quaternion,
            config=config, rng=rng,
        )
        extracted = optical_frame_to_observation_message(frame, config=config)
        result.append(replace(
            extracted,
            message_id=message.message_id,
            physical_observation_id=message.physical_observation_id,
            source_timestamp=message.source_timestamp,
            arrival_timestamp=message.arrival_timestamp,
            metadata={**message.metadata, **extracted.metadata},
        ))
        if retain_raw_frames:
            frames[message.information_id] = frame
    return result, frames


def _uv_rmse(messages, truth, timestamps):
    index_by_time = {
        float(value): index for index, value in enumerate(timestamps)
    }
    errors = []
    for message in messages:
        if message.modality.upper() != "OPTICAL" or not message.valid_flag:
            continue
        index = index_by_time[float(message.timestamp)]
        ideal = measure_relative_optical_uv(
            truth[message.observer_id][index], truth[message.target_id][index],
            frame="BODY",
            quaternion_i2b_wxyz=message.metadata["quaternion_i2b_wxyz"],
        )
        errors.append(np.asarray(message.measurement) - ideal)
    if not errors:
        return float("nan")
    values = np.vstack(errors)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _position_rmse(history, truth):
    errors = [
        history.active_state_history_by_node[node][:, :3]
        - truth[node][:, :3]
        for node in history.node_ids
    ]
    return float(np.sqrt(np.mean([
        np.sum(error * error, axis=1) for error in errors
    ])))


def _mean_optical_nis(history, messages):
    optical_ids = {
        item.information_id for item in messages
        if item.modality.upper() == "OPTICAL"
    }
    values = [
        value
        for node in history.node_ids
        for epoch in history.nis_history_by_node[node]
        for information_id, value in epoch.items()
        if information_id in optical_ids
    ]
    return float(np.mean(values)) if values else float("nan")
