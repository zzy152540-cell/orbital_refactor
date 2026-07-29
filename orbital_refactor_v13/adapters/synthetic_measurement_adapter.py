from __future__ import annotations

import numpy as np

from interfaces.data_objects import Observation


Array = np.ndarray


def create_infrared_observations(
    *,
    timestamps: Array,
    relative_position_spri: Array,
    covariance: Array,
    observer_id: str,
    target_id: str,
    rng: np.random.Generator,
    valid_flags: Array | None = None,
) -> list[Observation]:
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    position = np.asarray(relative_position_spri, dtype=float)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    valid = _valid_array(valid_flags, len(timestamps))
    if position.shape != (len(timestamps), 3):
        raise ValueError("relative_position_spri must have shape (N, 3).")

    azimuth = np.arctan2(position[:, 1], position[:, 0])
    elevation = np.arctan2(position[:, 2], np.linalg.norm(position[:, :2], axis=1))
    noise = rng.multivariate_normal(np.zeros(2), covariance, size=len(timestamps))
    measurements = np.column_stack((azimuth, elevation)) + noise
    return _create_observations(
        timestamps=timestamps,
        measurements=measurements,
        covariance=covariance,
        valid=valid,
        observer_id=observer_id,
        target_id=target_id,
        modality="INFRARED",
        measurement_type="AZIMUTH_ELEVATION",
        frame="SPRI",
    )


def create_radar_observations(
    *,
    timestamps: Array,
    relative_position_spri: Array,
    relative_velocity_spri: Array,
    covariance: Array,
    observer_id: str,
    target_id: str,
    rng: np.random.Generator,
    valid_flags: Array | None = None,
) -> list[Observation]:
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    position = np.asarray(relative_position_spri, dtype=float)
    velocity = np.asarray(relative_velocity_spri, dtype=float)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    valid = _valid_array(valid_flags, len(timestamps))
    if position.shape != (len(timestamps), 3) or velocity.shape != (len(timestamps), 3):
        raise ValueError("relative position and velocity must have shape (N, 3).")

    range_value = np.linalg.norm(position, axis=1)
    range_rate = np.sum(position * velocity, axis=1) / np.maximum(range_value, 1e-12)
    noise = rng.multivariate_normal(np.zeros(2), covariance, size=len(timestamps))
    measurements = np.column_stack((range_value, range_rate)) + noise
    return _create_observations(
        timestamps=timestamps,
        measurements=measurements,
        covariance=covariance,
        valid=valid,
        observer_id=observer_id,
        target_id=target_id,
        modality="RADAR",
        measurement_type="RANGE_RANGE_RATE",
        frame="SPRI",
    )


def apply_dropout_windows(valid_flags: Array, timestamps: Array, windows: list[tuple[float, float]] | None) -> Array:
    valid = np.asarray(valid_flags, dtype=bool).copy().reshape(-1)
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if valid.shape != timestamps.shape:
        raise ValueError("valid_flags and timestamps must have the same shape.")
    for start, end in windows or []:
        if end < start:
            raise ValueError("Dropout window end must not precede start.")
        valid[(timestamps >= float(start)) & (timestamps <= float(end))] = False
    return valid


def _create_observations(
    *,
    timestamps: Array,
    measurements: Array,
    covariance: Array,
    valid: Array,
    observer_id: str,
    target_id: str,
    modality: str,
    measurement_type: str,
    frame: str,
) -> list[Observation]:
    return [
        Observation(
            timestamp=float(timestamps[index]),
            observer_id=observer_id,
            target_id=target_id,
            modality=modality,
            source_type="TRADITIONAL",
            measurement=measurements[index].copy(),
            covariance=covariance.copy(),
            confidence=1.0,
            frame=frame,
            valid_flag=bool(valid[index]),
            metadata={"measurement_type": measurement_type},
        )
        for index in range(len(timestamps))
    ]


def _valid_array(valid_flags: Array | None, length: int) -> Array:
    if valid_flags is None:
        return np.ones(length, dtype=bool)
    valid = np.asarray(valid_flags, dtype=bool).reshape(-1)
    if valid.shape != (length,):
        raise ValueError("valid_flags must have shape (N,).")
    return valid


def create_optical_observations(
    *,
    timestamps: Array,
    relative_position_spri: Array,
    covariance: Array,
    observer_id: str,
    target_id: str,
    rng: np.random.Generator,
    valid_flags: Array | None = None,
    field_of_view_limit: float | None = None,
) -> list[Observation]:
    """Create normalized-image-coordinate observations ``[x/z, y/z]``."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    position = np.asarray(relative_position_spri, dtype=float)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    if position.shape != (len(timestamps), 3):
        raise ValueError("relative_position_spri must have shape (N, 3).")
    depth = position[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    ideal = np.column_stack((position[:, 0] / safe_depth, position[:, 1] / safe_depth))
    valid = _valid_array(valid_flags, len(timestamps)) & (depth > 1e-12)
    if field_of_view_limit is not None:
        limit = float(field_of_view_limit)
        if limit <= 0.0:
            raise ValueError("field_of_view_limit must be positive.")
        valid &= np.all(np.abs(ideal) <= limit, axis=1)
    noise = rng.multivariate_normal(np.zeros(2), covariance, size=len(timestamps))
    return _create_observations(
        timestamps=timestamps,
        measurements=ideal + noise,
        covariance=covariance,
        valid=valid,
        observer_id=observer_id,
        target_id=target_id,
        modality="OPTICAL",
        measurement_type="NORMALIZED_IMAGE_COORDINATES",
        frame="SPRI",
    )


def create_nn_state_observations(
    *,
    timestamps: Array,
    relative_state_eci: Array,
    covariance: Array,
    observer_id: str,
    target_id: str,
    rng: np.random.Generator,
    include_velocity: bool = True,
    valid_flags: Array | None = None,
) -> list[Observation]:
    """Create synthetic learning-enhanced optical state observations.

    This is a surrogate for network predictions and does not synthesize images.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    states = np.asarray(relative_state_eci, dtype=float)
    dimension = 6 if include_velocity else 3
    covariance = np.asarray(covariance, dtype=float).reshape(dimension, dimension)
    if states.shape != (len(timestamps), 6):
        raise ValueError("relative_state_eci must have shape (N, 6).")
    ideal = states[:, :dimension]
    noise = rng.multivariate_normal(np.zeros(dimension), covariance, size=len(timestamps))
    valid = _valid_array(valid_flags, len(timestamps))
    observations = _create_observations(
        timestamps=timestamps,
        measurements=ideal + noise,
        covariance=covariance,
        valid=valid,
        observer_id=observer_id,
        target_id=target_id,
        modality="OPTICAL",
        measurement_type="RELATIVE_STATE_ECI" if include_velocity else "RELATIVE_POSITION_ECI",
        frame="ECI",
    )
    for observation in observations:
        observation.source_type = "LEARNING"
    return observations


def visibility_flags(
    *,
    relative_position_spri: Array,
    min_range: float = 0.0,
    max_range: float = np.inf,
    require_positive_z: bool = False,
    field_of_view_limit: float | None = None,
) -> Array:
    """Build deterministic geometry/range visibility flags."""
    position = np.asarray(relative_position_spri, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("relative_position_spri must have shape (N, 3).")
    ranges = np.linalg.norm(position, axis=1)
    valid = (ranges >= float(min_range)) & (ranges <= float(max_range))
    if require_positive_z:
        valid &= position[:, 2] > 1e-12
    if field_of_view_limit is not None:
        safe_z = np.where(np.abs(position[:, 2]) > 1e-12, position[:, 2], 1.0)
        uv = np.column_stack((position[:, 0] / safe_z, position[:, 1] / safe_z))
        valid &= np.all(np.abs(uv) <= float(field_of_view_limit), axis=1)
    return valid
