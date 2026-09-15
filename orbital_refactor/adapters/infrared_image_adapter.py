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
from orbital_core.attitude import quat_to_dcm_i2b


@dataclass(frozen=True)
class InfraredCameraConfig(PointSourceImageConfig):
    """BODY-frame infrared focal-plane simulation settings."""

    focal_length_x_pixels: float = 300.0
    focal_length_y_pixels: float = 300.0
    reported_centroid_sigma_pixels: float = 0.25
    photon_noise_enabled: bool = False
    photon_gain_counts_per_intensity: float = 1.0
    thermal_background_drift_sigma: float = 0.0
    pointing_jitter_sigma_pixels: float = 0.0
    radial_distortion_k1_per_pixel2: float = 0.0
    fixed_pixel_bias_x: float = 0.0
    fixed_pixel_bias_y: float = 0.0

    def __post_init__(self):
        self.validate_image_settings()
        positive = (
            self.focal_length_x_pixels, self.focal_length_y_pixels,
            self.reported_centroid_sigma_pixels,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Infrared camera scales must be finite and positive.")
        errors = (
            self.thermal_background_drift_sigma,
            self.pointing_jitter_sigma_pixels,
        )
        if np.any(~np.isfinite(errors)) or min(errors) < 0.0:
            raise ValueError("Infrared random error scales must be nonnegative.")
        finite = (
            self.photon_gain_counts_per_intensity,
            self.radial_distortion_k1_per_pixel2,
            self.fixed_pixel_bias_x, self.fixed_pixel_bias_y,
        )
        if np.any(~np.isfinite(finite)) or self.photon_gain_counts_per_intensity <= 0.0:
            raise ValueError("Infrared systematic error settings must be finite.")


@dataclass(frozen=True)
class InfraredImageFrame:
    timestamp: float
    observer_id: str
    target_id: str
    image: np.ndarray
    quaternion_i2b_wxyz: np.ndarray
    ideal_az_el: np.ndarray
    ideal_pixel_xy: np.ndarray
    target_in_frame: bool
    error_diagnostics: dict[str, float] = field(default_factory=dict)


def render_infrared_point_source_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    quaternion_i2b_wxyz, config=None, rng=None,
):
    selected = config or InfraredCameraConfig()
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    quaternion = np.asarray(quaternion_i2b_wxyz, dtype=float).reshape(4)
    relative = quat_to_dcm_i2b(quaternion) @ (target[:3] - observer[:3])
    depth = float(relative[0])
    if depth <= 0.0:
        ideal_az_el = np.array([np.nan, np.nan])
        pixel_xy = np.array([np.nan, np.nan])
    else:
        normalized = relative[1:] / depth
        ideal_az_el = normalized_to_az_el(normalized)
        pixel_xy = normalized * np.array([
            selected.focal_length_x_pixels,
            selected.focal_length_y_pixels,
        ]) + np.array([selected.principal_x, selected.principal_y])
    image, in_frame, error_diagnostics = _render_with_infrared_errors(
        pixel_xy, config=selected, rng=rng,
    )
    return InfraredImageFrame(
        timestamp=float(timestamp), observer_id=str(observer_id),
        target_id=str(target_id), image=image,
        quaternion_i2b_wxyz=quaternion.copy(), ideal_az_el=ideal_az_el,
        ideal_pixel_xy=pixel_xy, target_in_frame=bool(depth > 0.0 and in_frame),
        error_diagnostics=error_diagnostics,
    )


def normalized_to_az_el(normalized):
    """Convert BODY pinhole coordinates [y/x, z/x] to azimuth/elevation."""
    u, v = np.asarray(normalized, dtype=float).reshape(2)
    return np.array([
        np.arctan(u),
        np.arctan2(v, np.sqrt(1.0 + u * u)),
    ])


def infrared_pixel_to_az_el(pixel_xy, config=None):
    selected = config or InfraredCameraConfig()
    normalized = (
        np.asarray(pixel_xy, dtype=float).reshape(2)
        - np.array([selected.principal_x, selected.principal_y])
    ) / np.array([
        selected.focal_length_x_pixels,
        selected.focal_length_y_pixels,
    ])
    return normalized_to_az_el(normalized)


def infrared_frame_to_observation_message(
    frame, *, config=None, covariance_calibration=None,
):
    selected = config or InfraredCameraConfig()
    pixel_xy, detected = extract_point_source_centroid(
        frame.image, config=selected, target_in_frame=frame.target_in_frame,
    )
    frontend_quality = point_source_quality(
        frame.image, pixel_xy, config=selected, detected=detected,
    )
    measurement = (
        infrared_pixel_to_az_el(pixel_xy, selected)
        if detected else np.zeros(2, dtype=float)
    )
    covariance = _angle_covariance(pixel_xy, selected) if detected else np.eye(2)
    covariance_metadata = {"covariance_source": "reported_centroid_sigma"}
    if detected and covariance_calibration is not None:
        covariance, calibrated_bias, match = covariance_calibration.lookup(
            config=selected, pixel_xy=pixel_xy,
            peak_snr=frontend_quality["peak_snr"],
        )
        if covariance_calibration.apply_bias_correction:
            measurement = measurement - calibrated_bias
        covariance_metadata = {
            "covariance_source": "infrared_empirical_table",
            "infrared_covariance_calibration": match,
            "infrared_bias_correction_applied": bool(
                covariance_calibration.apply_bias_correction
            ),
            "infrared_calibrated_bias_rad": calibrated_bias,
        }
    information_id = (
        f"{frame.observer_id}->{frame.target_id}:infrared_image:"
        f"{frame.timestamp:g}"
    )
    return ObservationMessage(
        message_id=information_id, physical_observation_id=information_id,
        observer_id=frame.observer_id, target_id=frame.target_id,
        timestamp=frame.timestamp, modality="INFRARED", frame="BODY",
        measurement=measurement, covariance=covariance,
        valid_flag=bool(detected), source_timestamp=frame.timestamp,
        arrival_timestamp=frame.timestamp,
        metadata={
            "sensor_modality": "INFRARED",
            "measurement_type": "AZIMUTH_ELEVATION",
            "measurement_component": "INFRARED",
            "raw_source_type": "INFRARED_POINT_SOURCE_IMAGE",
            "raw_image_shape": frame.image.shape,
            "centroid_pixel_xy": pixel_xy,
            "quaternion_i2b_wxyz": frame.quaternion_i2b_wxyz.copy(),
            "infrared_error_diagnostics": dict(frame.error_diagnostics),
            **covariance_metadata,
            **frontend_quality,
        },
    )


def _angle_covariance(pixel_xy, config):
    pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
    epsilon = 1.0e-4
    jacobian = np.empty((2, 2), dtype=float)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = epsilon
        jacobian[:, axis] = (
            infrared_pixel_to_az_el(pixel + offset, config)
            - infrared_pixel_to_az_el(pixel - offset, config)
        ) / (2.0 * epsilon)
    pixel_covariance = np.eye(2) * config.reported_centroid_sigma_pixels**2
    return jacobian @ pixel_covariance @ jacobian.T


def _render_with_infrared_errors(pixel_xy, *, config, rng=None):
    enabled = bool(
        config.photon_noise_enabled
        or config.thermal_background_drift_sigma > 0.0
        or config.pointing_jitter_sigma_pixels > 0.0
        or config.radial_distortion_k1_per_pixel2 != 0.0
        or config.fixed_pixel_bias_x != 0.0
        or config.fixed_pixel_bias_y != 0.0
    )
    if not enabled:
        image, in_frame = render_gaussian_point_source(
            pixel_xy, config=config, rng=rng,
        )
        return image, in_frame, {}

    generator = np.random.default_rng() if rng is None else rng
    ideal = np.asarray(pixel_xy, dtype=float).reshape(2)
    principal = np.array([config.principal_x, config.principal_y])
    offset = ideal - principal
    radius_squared = float(offset @ offset)
    distorted = principal + offset * (
        1.0 + config.radial_distortion_k1_per_pixel2 * radius_squared
    )
    jitter = generator.normal(
        0.0, config.pointing_jitter_sigma_pixels, size=2
    )
    applied = distorted + np.array([
        config.fixed_pixel_bias_x, config.fixed_pixel_bias_y,
    ]) + jitter
    noiseless_config = replace(config, read_noise_sigma=0.0)
    image, in_frame = render_gaussian_point_source(
        applied, config=noiseless_config, rng=generator,
    )
    signal = image - config.background
    background_drift = float(generator.normal(
        0.0, config.thermal_background_drift_sigma
    ))
    actual_background = max(config.background + background_drift, 0.0)
    image = signal + actual_background
    if config.photon_noise_enabled:
        gain = config.photon_gain_counts_per_intensity
        image = generator.poisson(np.maximum(image * gain, 0.0)) / gain
    if config.read_noise_sigma > 0.0:
        image += generator.normal(0.0, config.read_noise_sigma, image.shape)
    return image, in_frame, {
        "applied_pixel_x": float(applied[0]),
        "applied_pixel_y": float(applied[1]),
        "jitter_x_pixels": float(jitter[0]),
        "jitter_y_pixels": float(jitter[1]),
        "background_drift": background_drift,
        "radial_shift_pixels": float(np.linalg.norm(distorted - ideal)),
        "fixed_bias_norm_pixels": float(np.hypot(
            config.fixed_pixel_bias_x, config.fixed_pixel_bias_y
        )),
    }
