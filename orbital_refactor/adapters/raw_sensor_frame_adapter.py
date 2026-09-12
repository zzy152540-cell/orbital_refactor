"""Lossless adapters between concrete simulated frames and RawSensorFrame."""

from __future__ import annotations

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig, InfraredImageFrame
from adapters.optical_image_adapter import OpticalCameraConfig, OpticalImageFrame
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig, RadarRangeDopplerFrame,
)
from interfaces.data_objects import RawSensorFrame
from interfaces.interface_contracts import validate_raw_sensor_frame


def raw_sensor_frame_from_radar(frame, *, config=None) -> RawSensorFrame:
    selected = config or RadarRangeDopplerConfig(
        width=frame.power.shape[1], height=frame.power.shape[0],
    )
    columns = np.arange(selected.width, dtype=float)
    rows = np.arange(selected.height, dtype=float)
    result = RawSensorFrame(
        timestamp=frame.timestamp, observer_id=frame.observer_id,
        target_id=frame.target_id, modality="RADAR", data=frame.power,
        data_kind="RANGE_DOPPLER_POWER_MAP",
        axes={
            "range_m": frame.acquisition_center[0]
            + (columns - selected.principal_x) * selected.range_bin_size_m,
            "range_rate_mps": frame.acquisition_center[1]
            + (rows - selected.principal_y) * selected.range_rate_bin_size_mps,
        },
        calibration={
            "principal_xy": [selected.principal_x, selected.principal_y],
            "range_bin_size_m": selected.range_bin_size_m,
            "range_rate_bin_size_mps": selected.range_rate_bin_size_mps,
        },
        valid_flag=bool(frame.target_in_window),
        metadata={
            "acquisition_center": np.asarray(frame.acquisition_center).copy(),
            "simulation": {
                "ideal_range_range_rate": np.asarray(
                    frame.ideal_range_range_rate
                ).copy(),
                "ideal_bin_xy": np.asarray(frame.ideal_bin_xy).copy(),
            },
        },
    )
    validate_raw_sensor_frame(result)
    return result


def raw_sensor_frame_from_infrared(frame, *, config=None) -> RawSensorFrame:
    selected = config or InfraredCameraConfig(
        width=frame.image.shape[1], height=frame.image.shape[0],
    )
    return _raw_camera_frame(
        frame, selected, modality="INFRARED",
        data_kind="INFRARED_POINT_SOURCE_IMAGE",
        ideal_name="ideal_az_el", ideal_value=frame.ideal_az_el,
    )


def raw_sensor_frame_from_optical(frame, *, config=None) -> RawSensorFrame:
    selected = config or OpticalCameraConfig(
        width=frame.image.shape[1], height=frame.image.shape[0],
    )
    return _raw_camera_frame(
        frame, selected, modality="OPTICAL",
        data_kind="OPTICAL_POINT_SOURCE_IMAGE",
        ideal_name="ideal_normalized_uv", ideal_value=frame.ideal_normalized_uv,
    )


def radar_frame_from_raw(frame: RawSensorFrame) -> RadarRangeDopplerFrame:
    validate_raw_sensor_frame(frame)
    _require_modality(frame, "RADAR")
    simulation = _simulation_metadata(frame)
    return RadarRangeDopplerFrame(
        timestamp=frame.timestamp, observer_id=frame.observer_id,
        target_id=frame.target_id, power=np.array(frame.data, copy=True),
        acquisition_center=np.asarray(frame.metadata["acquisition_center"], dtype=float),
        ideal_range_range_rate=np.asarray(
            simulation["ideal_range_range_rate"], dtype=float,
        ),
        ideal_bin_xy=np.asarray(simulation["ideal_bin_xy"], dtype=float),
        target_in_window=bool(frame.valid_flag),
    )


def infrared_frame_from_raw(frame: RawSensorFrame) -> InfraredImageFrame:
    validate_raw_sensor_frame(frame)
    _require_modality(frame, "INFRARED")
    simulation = _simulation_metadata(frame)
    return InfraredImageFrame(
        timestamp=frame.timestamp, observer_id=frame.observer_id,
        target_id=frame.target_id, image=np.array(frame.data, copy=True),
        quaternion_i2b_wxyz=np.asarray(frame.metadata["quaternion_i2b_wxyz"]),
        ideal_az_el=np.asarray(simulation["ideal_az_el"]),
        ideal_pixel_xy=np.asarray(simulation["ideal_pixel_xy"]),
        target_in_frame=bool(frame.valid_flag),
    )


def optical_frame_from_raw(frame: RawSensorFrame) -> OpticalImageFrame:
    validate_raw_sensor_frame(frame)
    _require_modality(frame, "OPTICAL")
    simulation = _simulation_metadata(frame)
    return OpticalImageFrame(
        timestamp=frame.timestamp, observer_id=frame.observer_id,
        target_id=frame.target_id, image=np.array(frame.data, copy=True),
        quaternion_i2b_wxyz=np.asarray(frame.metadata["quaternion_i2b_wxyz"]),
        ideal_normalized_uv=np.asarray(simulation["ideal_normalized_uv"]),
        ideal_pixel_xy=np.asarray(simulation["ideal_pixel_xy"]),
        target_in_frame=bool(frame.valid_flag),
    )


def _raw_camera_frame(frame, config, *, modality, data_kind, ideal_name, ideal_value):
    result = RawSensorFrame(
        timestamp=frame.timestamp, observer_id=frame.observer_id,
        target_id=frame.target_id, modality=modality, data=frame.image,
        data_kind=data_kind,
        axes={
            "pixel_x": np.arange(config.width, dtype=float),
            "pixel_y": np.arange(config.height, dtype=float),
        },
        calibration={
            "principal_xy": [config.principal_x, config.principal_y],
            "focal_length_xy_pixels": [
                config.focal_length_x_pixels, config.focal_length_y_pixels,
            ],
        },
        valid_flag=bool(frame.target_in_frame),
        metadata={
            "quaternion_i2b_wxyz": np.asarray(frame.quaternion_i2b_wxyz).copy(),
            "simulation": {
                ideal_name: np.asarray(ideal_value).copy(),
                "ideal_pixel_xy": np.asarray(frame.ideal_pixel_xy).copy(),
            },
        },
    )
    validate_raw_sensor_frame(result)
    return result


def _require_modality(frame, modality):
    if str(frame.modality).upper() != modality:
        raise ValueError(f"Expected a {modality} RawSensorFrame.")


def _simulation_metadata(frame):
    try:
        return frame.metadata["simulation"]
    except KeyError as exc:
        raise ValueError(
            "Reverse conversion requires simulation metadata from a concrete frame."
        ) from exc
