from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import ObservationMessage
from orbital_core.attitude import quat_to_dcm_i2b
from adapters.point_source_image import (
    PointSourceImageConfig,
    extract_point_source_centroid,
    render_gaussian_point_source,
)


@dataclass(frozen=True)
class OpticalCameraConfig(PointSourceImageConfig):
    """Minimal pinhole camera and point-source image simulation settings."""

    focal_length_x_pixels: float = 400.0
    focal_length_y_pixels: float = 400.0
    reported_centroid_sigma_pixels: float = 0.2

    def __post_init__(self):
        self.validate_image_settings()
        positive = (
            self.focal_length_x_pixels, self.focal_length_y_pixels,
            self.reported_centroid_sigma_pixels,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Optical camera scales must be finite and positive.")


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

    image, rendered_in_frame = render_gaussian_point_source(
        pixel_xy, config=selected, rng=rng,
    )
    target_in_frame = target_in_frame and rendered_in_frame
    return OpticalImageFrame(
        timestamp=float(timestamp), observer_id=str(observer_id),
        target_id=str(target_id), image=image,
        quaternion_i2b_wxyz=quaternion.copy(),
        ideal_normalized_uv=normalized_uv,
        ideal_pixel_xy=pixel_xy, target_in_frame=target_in_frame,
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


def optical_frame_to_observation_message(frame, *, config=None):
    """Convert a raw image frame into the existing normalized-UV interface."""
    selected = config or OpticalCameraConfig()
    pixel_xy, detected = extract_optical_centroid(frame, config=selected)
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
        },
    )
