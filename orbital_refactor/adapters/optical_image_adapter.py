from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from interfaces.data_objects import ObservationMessage
from orbital_core.attitude import quat_to_dcm_i2b
from adapters.point_source_image import (
    PointSourceImageConfig,
    extract_point_source_centroid,
    point_source_quality,
    render_gaussian_point_source,
)


@dataclass(frozen=True)
class OpticalCameraConfig(PointSourceImageConfig):
    """Minimal pinhole camera and point-source image simulation settings."""

    focal_length_x_pixels: float = 400.0
    focal_length_y_pixels: float = 400.0
    reported_centroid_sigma_pixels: float = 0.2
    photon_noise_enabled: bool = False
    photon_gain_counts_per_intensity: float = 1.0
    stray_light_drift_sigma: float = 0.0
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
            raise ValueError("Optical camera scales must be finite and positive.")
        random_errors = (
            self.stray_light_drift_sigma,
            self.pointing_jitter_sigma_pixels,
        )
        if np.any(~np.isfinite(random_errors)) or min(random_errors) < 0.0:
            raise ValueError("Optical random error scales must be nonnegative.")
        systematic = (
            self.photon_gain_counts_per_intensity,
            self.radial_distortion_k1_per_pixel2,
            self.fixed_pixel_bias_x,
            self.fixed_pixel_bias_y,
        )
        if (
            np.any(~np.isfinite(systematic))
            or self.photon_gain_counts_per_intensity <= 0.0
        ):
            raise ValueError("Optical systematic error settings must be finite.")


@dataclass(frozen=True)
class OpticalImageFrame:
    timestamp: float
    observer_id: str
    target_id: str
    image: np.ndarray
    quaternion_i2b_wxyz: np.ndarray
    ideal_normalized_uv: np.ndarray
    ideal_pixel_xy: np.ndarray
    target_in_frame: bool
    error_diagnostics: dict[str, float] = field(default_factory=dict)


def render_optical_point_source_frame(
    *, timestamp, observer_id, target_id, observer_state, target_state,
    quaternion_i2b_wxyz, config=None, rng=None,
) -> OpticalImageFrame:
    """Render one BODY +X-boresight optical point source as a grayscale image."""
    selected = config or OpticalCameraConfig()
    observer = np.asarray(observer_state, dtype=float).reshape(6)
    target = np.asarray(target_state, dtype=float).reshape(6)
    quaternion = np.asarray(quaternion_i2b_wxyz, dtype=float).reshape(4)
    relative_body = quat_to_dcm_i2b(quaternion) @ (
        target[:3] - observer[:3]
    )
    depth = float(relative_body[0])
    if depth <= 0.0:
        normalized_uv = np.array([np.nan, np.nan])
        pixel_xy = np.array([np.nan, np.nan])
        target_in_frame = False
    else:
        normalized_uv = relative_body[1:] / depth
        pixel_xy = normalized_uv * np.array([
            selected.focal_length_x_pixels,
            selected.focal_length_y_pixels,
        ]) + np.array([selected.principal_x, selected.principal_y])
        target_in_frame = bool(
            0.0 <= pixel_xy[0] <= selected.width - 1
            and 0.0 <= pixel_xy[1] <= selected.height - 1
        )

    image, rendered_in_frame, error_diagnostics = _render_with_optical_errors(
        pixel_xy, config=selected, rng=rng,
    )
    target_in_frame = target_in_frame and rendered_in_frame
    return OpticalImageFrame(
        timestamp=float(timestamp), observer_id=str(observer_id),
        target_id=str(target_id), image=image,
        quaternion_i2b_wxyz=quaternion.copy(),
        ideal_normalized_uv=normalized_uv,
        ideal_pixel_xy=pixel_xy, target_in_frame=target_in_frame,
        error_diagnostics=error_diagnostics,
    )


def extract_optical_centroid(frame, *, config=None):
    """Extract a subpixel point-source centroid from a rendered image."""
    selected = config or OpticalCameraConfig()
    return extract_point_source_centroid(
        frame.image, config=selected, target_in_frame=frame.target_in_frame,
    )


def pixel_to_normalized_uv(pixel_xy, config=None):
    selected = config or OpticalCameraConfig()
    pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
    return (pixel - np.array([
        selected.principal_x, selected.principal_y,
    ])) / np.array([
        selected.focal_length_x_pixels,
        selected.focal_length_y_pixels,
    ])


def optical_frame_to_observation_message(
    frame, *, config=None, covariance_calibration=None,
):
    """Convert a raw image frame into the existing normalized-UV interface."""
    selected = config or OpticalCameraConfig()
    pixel_xy, detected = extract_optical_centroid(frame, config=selected)
    frontend_quality = point_source_quality(
        frame.image, pixel_xy, config=selected, detected=detected,
    )
    measurement = (
        pixel_to_normalized_uv(pixel_xy, selected)
        if detected else np.zeros(2, dtype=float)
    )
    covariance = np.diag([
        (
            selected.reported_centroid_sigma_pixels
            / selected.focal_length_x_pixels
        ) ** 2,
        (
            selected.reported_centroid_sigma_pixels
            / selected.focal_length_y_pixels
        ) ** 2,
    ])
    covariance_metadata = {"covariance_source": "reported_centroid_sigma"}
    if detected and covariance_calibration is not None:
        covariance, calibrated_bias, match = covariance_calibration.lookup(
            coordinate_xy=pixel_xy,
            principal_xy=np.array([selected.principal_x, selected.principal_y]),
            peak_snr=frontend_quality["peak_snr"],
        )
        if covariance_calibration.apply_bias_correction:
            measurement = measurement - calibrated_bias
        covariance_metadata = {
            "covariance_source": "optical_empirical_table",
            "optical_covariance_calibration": match,
            "optical_bias_correction_applied": bool(
                covariance_calibration.apply_bias_correction
            ),
            "optical_calibrated_bias": calibrated_bias,
        }
    information_id = (
        f"{frame.observer_id}->{frame.target_id}:optical_image:"
        f"{frame.timestamp:g}"
    )
    return ObservationMessage(
        message_id=information_id,
        physical_observation_id=information_id,
        observer_id=frame.observer_id, target_id=frame.target_id,
        timestamp=frame.timestamp, modality="OPTICAL", frame="BODY",
        measurement=measurement, covariance=covariance,
        valid_flag=bool(detected),
        source_timestamp=frame.timestamp,
        arrival_timestamp=frame.timestamp,
        metadata={
            "sensor_modality": "OPTICAL",
            "measurement_type": "NORMALIZED_IMAGE_COORDINATES",
            "measurement_component": "OPTICAL",
            "raw_source_type": "POINT_SOURCE_IMAGE",
            "raw_image_shape": frame.image.shape,
            "centroid_pixel_xy": pixel_xy,
            "quaternion_i2b_wxyz": frame.quaternion_i2b_wxyz.copy(),
            "optical_error_diagnostics": dict(frame.error_diagnostics),
            **covariance_metadata,
            **frontend_quality,
        },
    )


def _render_with_optical_errors(pixel_xy, *, config, rng=None):
    enabled = bool(
        config.photon_noise_enabled
        or config.stray_light_drift_sigma > 0.0
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
    distorted = principal + offset * (
        1.0 + config.radial_distortion_k1_per_pixel2 * float(offset @ offset)
    )
    jitter = generator.normal(0.0, config.pointing_jitter_sigma_pixels, 2)
    applied = distorted + np.array([
        config.fixed_pixel_bias_x, config.fixed_pixel_bias_y,
    ]) + jitter
    noiseless = replace(config, read_noise_sigma=0.0)
    image, in_frame = render_gaussian_point_source(
        applied, config=noiseless, rng=generator,
    )
    signal = image - config.background
    background_drift = float(generator.normal(0.0, config.stray_light_drift_sigma))
    image = signal + max(config.background + background_drift, 0.0)
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
            config.fixed_pixel_bias_x, config.fixed_pixel_bias_y,
        )),
    }
