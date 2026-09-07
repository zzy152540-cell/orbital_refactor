from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters.point_source_image import (
    PointSourceImageConfig,
    extract_point_source_centroid,
    point_source_quality,
    render_gaussian_point_source,
)
from interfaces.data_objects import ObservationMessage
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


@dataclass(frozen=True)
class RadarRangeDopplerConfig(PointSourceImageConfig):
    """Local range-Doppler power-map settings, with columns as range bins."""

    width: int = 129
    height: int = 129
    range_bin_size_m: float = 10.0
    range_rate_bin_size_mps: float = 0.05
    reported_centroid_sigma_bins: float = 0.25

    def __post_init__(self):
        self.validate_image_settings()
        positive = (
            self.range_bin_size_m, self.range_rate_bin_size_mps,
            self.reported_centroid_sigma_bins,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Range-Doppler scales must be finite and positive.")


@dataclass(frozen=True)
class RadarRangeDopplerFrame:
    timestamp: float
    observer_id: str
    target_id: str
    power: np.ndarray
    acquisition_center: np.ndarray
    ideal_range_range_rate: np.ndarray
    ideal_bin_xy: np.ndarray
    target_in_window: bool


def render_radar_range_doppler_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    acquisition_range_m, acquisition_range_rate_mps,
    config=None, rng=None,
):
    """Render a processed local power map around an acquisition center."""
    selected = config or RadarRangeDopplerConfig()
    ideal = np.array([
        measure_relative_range(observer_state, target_state),
        measure_relative_range_rate(observer_state, target_state),
    ])
    center = np.array([
        float(acquisition_range_m), float(acquisition_range_rate_mps),
    ])
    bin_xy = np.array([
        selected.principal_x + (ideal[0] - center[0]) / selected.range_bin_size_m,
        selected.principal_y
        + (ideal[1] - center[1]) / selected.range_rate_bin_size_mps,
    ])
    power, in_window = render_gaussian_point_source(
        bin_xy, config=selected, rng=rng,
    )
    return RadarRangeDopplerFrame(
        timestamp=float(timestamp), observer_id=str(observer_id),
        target_id=str(target_id), power=power, acquisition_center=center,
        ideal_range_range_rate=ideal, ideal_bin_xy=bin_xy,
        target_in_window=in_window,
    )


def radar_bin_to_measurement(bin_xy, frame, config=None):
    selected = config or RadarRangeDopplerConfig()
    pixel = np.asarray(bin_xy, dtype=float).reshape(2)
    return frame.acquisition_center + np.array([
        (pixel[0] - selected.principal_x) * selected.range_bin_size_m,
        (pixel[1] - selected.principal_y) * selected.range_rate_bin_size_mps,
    ])


def radar_frame_to_observation_message(frame, *, config=None):
    selected = config or RadarRangeDopplerConfig()
    bin_xy, detected = extract_point_source_centroid(
        frame.power, config=selected, target_in_frame=frame.target_in_window,
    )
    frontend_quality = point_source_quality(
        frame.power, bin_xy, config=selected, detected=detected,
    )
    measurement = (
        radar_bin_to_measurement(bin_xy, frame, selected)
        if detected else np.zeros(2, dtype=float)
    )
    covariance = np.diag([
        (
            selected.reported_centroid_sigma_bins
            * selected.range_bin_size_m
        ) ** 2,
        (
            selected.reported_centroid_sigma_bins
            * selected.range_rate_bin_size_mps
        ) ** 2,
    ])
    information_id = (
        f"{frame.observer_id}->{frame.target_id}:range_doppler:"
        f"{frame.timestamp:g}"
    )
    return ObservationMessage(
        message_id=information_id, physical_observation_id=information_id,
        observer_id=frame.observer_id, target_id=frame.target_id,
        timestamp=frame.timestamp, modality="RADAR", frame="ECI",
        measurement=measurement, covariance=covariance,
        valid_flag=bool(detected), source_timestamp=frame.timestamp,
        arrival_timestamp=frame.timestamp,
        metadata={
            "sensor_modality": "RADAR",
            "measurement_type": "RANGE_RANGE_RATE",
            "measurement_component": "RADAR",
            "raw_source_type": "RANGE_DOPPLER_POWER_MAP",
            "raw_power_shape": frame.power.shape,
            "centroid_bin_xy": bin_xy,
            "acquisition_center": frame.acquisition_center.copy(),
            "range_bin_size_m": selected.range_bin_size_m,
            "range_rate_bin_size_mps": selected.range_rate_bin_size_mps,
            **frontend_quality,
        },
    )
