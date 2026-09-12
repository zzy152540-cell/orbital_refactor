"""Selected-link raw sensor front end for visualization replay experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from adapters.raw_sensor_frame_adapter import (
    raw_sensor_frame_from_infrared,
    raw_sensor_frame_from_optical,
    raw_sensor_frame_from_radar,
)
from experiments.inter_satellite_observation_factory import (
    target_pointing_quaternion,
)


def replace_selected_link_with_raw_frontends(
    messages, *, timestamps, truth_state_history_by_node,
    capture_link: tuple[str, str], seed: int = 0,
):
    """Use and retain raw-front-end measurements for one directed link.

    Other links remain on the existing numerical measurement path. Radar uses
    the existing numerical measurement as a bounded acquisition center; the
    retained power map is therefore a simulation/replay front end rather than
    a claim of an independently implemented flight acquisition controller.
    """
    observer_id, target_id = map(str, capture_link)
    if observer_id == target_id:
        raise ValueError("A raw capture link requires two different nodes.")
    index_by_time = {
        float(value): index
        for index, value in enumerate(np.asarray(timestamps, dtype=float))
    }
    radar_config = RadarRangeDopplerConfig(
        width=65, height=65, range_bin_size_m=8.0,
        range_rate_bin_size_mps=0.2, source_peak=100.0,
        background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_bins=0.19,
    )
    infrared_config = InfraredCameraConfig(
        width=64, height=64, focal_length_x_pixels=120.0,
        focal_length_y_pixels=120.0, source_peak=100.0,
        background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.26,
    )
    optical_config = OpticalCameraConfig(
        width=64, height=64, focal_length_x_pixels=100.0,
        focal_length_y_pixels=100.0, source_peak=100.0,
        background=10.0, read_noise_sigma=10.0,
        reported_centroid_sigma_pixels=0.25,
    )
    generators = {
        "RADAR": np.random.default_rng(20262000 + seed),
        "INFRARED": np.random.default_rng(20262100 + seed),
        "OPTICAL": np.random.default_rng(20262200 + seed),
    }
    replaced, raw_by_information_id = [], {}
    for message in messages:
        if (message.observer_id, message.target_id) != (observer_id, target_id):
            replaced.append(message)
            continue
        index = index_by_time[float(message.timestamp)]
        observer = truth_state_history_by_node[observer_id][index]
        target = truth_state_history_by_node[target_id][index]
        modality = message.modality.upper()
        if modality == "RADAR":
            center = np.asarray(message.measurement, dtype=float).reshape(2)
            frame = render_radar_range_doppler_frame(
                timestamp=message.timestamp, observer_id=observer_id,
                target_id=target_id, observer_state=observer,
                target_state=target, acquisition_range_m=center[0],
                acquisition_range_rate_mps=center[1], config=radar_config,
                rng=generators[modality],
            )
            extracted = radar_frame_to_observation_message(
                frame, config=radar_config,
            )
            raw = raw_sensor_frame_from_radar(frame, config=radar_config)
        elif modality == "INFRARED":
            quaternion = target_pointing_quaternion(observer, target)
            frame = render_infrared_point_source_frame(
                timestamp=message.timestamp, observer_id=observer_id,
                target_id=target_id, observer_state=observer,
                target_state=target, quaternion_i2b_wxyz=quaternion,
                config=infrared_config, rng=generators[modality],
            )
            extracted = infrared_frame_to_observation_message(
                frame, config=infrared_config,
            )
            raw = raw_sensor_frame_from_infrared(frame, config=infrared_config)
        elif modality == "OPTICAL":
            quaternion = np.asarray(
                message.metadata.get(
                    "quaternion_i2b_wxyz",
                    target_pointing_quaternion(observer, target),
                ), dtype=float,
            )
            frame = render_optical_point_source_frame(
                timestamp=message.timestamp, observer_id=observer_id,
                target_id=target_id, observer_state=observer,
                target_state=target, quaternion_i2b_wxyz=quaternion,
                config=optical_config, rng=generators[modality],
            )
            extracted = optical_frame_to_observation_message(
                frame, config=optical_config,
            )
            raw = raw_sensor_frame_from_optical(frame, config=optical_config)
        else:
            replaced.append(message)
            continue
        preserved = replace(
            extracted, message_id=message.message_id,
            physical_observation_id=message.physical_observation_id,
            source_timestamp=message.source_timestamp,
            arrival_timestamp=message.arrival_timestamp,
            metadata={
                **message.metadata, **extracted.metadata,
                "visualization_raw_frontend": True,
                "raw_capture_scope": "SELECTED_DIRECTED_LINK",
            },
        )
        replaced.append(preserved)
        raw_by_information_id[preserved.information_id] = raw
    if not raw_by_information_id:
        raise ValueError("The selected raw capture link has no observations.")
    return tuple(replaced), raw_by_information_id
