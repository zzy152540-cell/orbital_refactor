from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    background_drift_sigma: float = 0.0
    echo_amplitude_sigma_fraction: float = 0.0
    range_bias_m: float = 0.0
    range_rate_bias_mps: float = 0.0
    range_jitter_sigma_m: float = 0.0
    range_rate_jitter_sigma_mps: float = 0.0

    def __post_init__(self):
        self.validate_image_settings()
        positive = (
            self.range_bin_size_m, self.range_rate_bin_size_mps,
            self.reported_centroid_sigma_bins,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Range-Doppler scales must be finite and positive.")
        random_errors = (
            self.background_drift_sigma,
            self.echo_amplitude_sigma_fraction,
            self.range_jitter_sigma_m,
            self.range_rate_jitter_sigma_mps,
        )
        if np.any(~np.isfinite(random_errors)) or min(random_errors) < 0.0:
            raise ValueError("Radar random error scales must be nonnegative.")
        if not np.isfinite(self.range_bias_m) or not np.isfinite(
            self.range_rate_bias_mps
        ):
            raise ValueError("Radar systematic biases must be finite.")


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
    error_diagnostics: dict[str, float] = field(default_factory=dict)


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
    power, in_window, error_diagnostics = _render_with_radar_errors(
        bin_xy, config=selected, rng=rng,
    )
    return RadarRangeDopplerFrame(
        timestamp=float(timestamp), observer_id=str(observer_id),
        target_id=str(target_id), power=power, acquisition_center=center,
        ideal_range_range_rate=ideal, ideal_bin_xy=bin_xy,
        target_in_window=in_window, error_diagnostics=error_diagnostics,
    )


def radar_bin_to_measurement(bin_xy, frame, config=None):
    selected = config or RadarRangeDopplerConfig()
    pixel = np.asarray(bin_xy, dtype=float).reshape(2)
    return frame.acquisition_center + np.array([
        (pixel[0] - selected.principal_x) * selected.range_bin_size_m,
        (pixel[1] - selected.principal_y) * selected.range_rate_bin_size_mps,
    ])


def radar_frame_to_observation_message(
    frame, *, config=None, covariance_calibration=None,
):
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
    covariance_metadata = {"covariance_source": "reported_centroid_sigma"}
    if detected and covariance_calibration is not None:
        applicable, reason = covariance_calibration.applicability(
            config=selected, peak_snr=frontend_quality["peak_snr"],
        )
        if applicable:
            covariance, calibrated_bias, match = covariance_calibration.lookup(
                coordinate_xy=bin_xy,
                principal_xy=np.array([selected.principal_x, selected.principal_y]),
                peak_snr=frontend_quality["peak_snr"],
            )
            if covariance_calibration.apply_bias_correction:
                measurement = measurement - calibrated_bias
            covariance_metadata = {
                "covariance_source": "radar_empirical_table",
                "radar_covariance_calibration": match,
                "radar_bias_correction_applied": bool(
                    covariance_calibration.apply_bias_correction
                ),
                "radar_calibrated_bias": calibrated_bias,
                "radar_calibration_applicability": reason,
            }
        else:
            covariance_metadata["radar_calibration_applicability"] = reason
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
            "radar_error_diagnostics": dict(frame.error_diagnostics),
            **covariance_metadata,
            **frontend_quality,
        },
    )


def _render_with_radar_errors(bin_xy, *, config, rng=None):
    enabled = bool(
        config.background_drift_sigma > 0.0
        or config.echo_amplitude_sigma_fraction > 0.0
        or config.range_bias_m != 0.0
        or config.range_rate_bias_mps != 0.0
        or config.range_jitter_sigma_m > 0.0
        or config.range_rate_jitter_sigma_mps > 0.0
    )
    if not enabled:
        power, in_window = render_gaussian_point_source(
            bin_xy, config=config, rng=rng,
        )
        return power, in_window, {}

    generator = np.random.default_rng() if rng is None else rng
    jitter = np.array([
        generator.normal(0.0, config.range_jitter_sigma_m),
        generator.normal(0.0, config.range_rate_jitter_sigma_mps),
    ])
    physical_shift = np.array([
        config.range_bias_m, config.range_rate_bias_mps,
    ]) + jitter
    bin_shift = physical_shift / np.array([
        config.range_bin_size_m, config.range_rate_bin_size_mps,
    ])
    applied = np.asarray(bin_xy, dtype=float).reshape(2) + bin_shift
    amplitude_scale = max(
        0.0,
        1.0 + generator.normal(0.0, config.echo_amplitude_sigma_fraction),
    )
    noiseless = replace(
        config,
        source_peak=config.source_peak * amplitude_scale,
        read_noise_sigma=0.0,
    )
    power, in_window = render_gaussian_point_source(
        applied, config=noiseless, rng=generator,
    )
    signal = power - config.background
    background_drift = float(generator.normal(0.0, config.background_drift_sigma))
    power = signal + max(config.background + background_drift, 0.0)
    if config.read_noise_sigma > 0.0:
        power += generator.normal(0.0, config.read_noise_sigma, power.shape)
    return power, in_window, {
        "applied_range_bias_m": float(physical_shift[0]),
        "applied_range_rate_bias_mps": float(physical_shift[1]),
        "range_jitter_m": float(jitter[0]),
        "range_rate_jitter_mps": float(jitter[1]),
        "background_drift": background_drift,
        "echo_amplitude_scale": float(amplitude_scale),
    }
