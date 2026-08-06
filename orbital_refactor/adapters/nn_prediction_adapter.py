from __future__ import annotations

from pathlib import Path

import numpy as np

from interfaces.data_objects import Observation


Array = np.ndarray


def load_aligned_nn_positions(
    predictions_path: str | Path,
    shirt_filenames: list[str],
) -> tuple[Array, Array]:
    """Align predictions.npz to SHIRT frame names using the legacy key contract."""
    path = Path(predictions_path)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    with np.load(path, allow_pickle=True) as data:
        missing = {"image_path", "t_pred"} - set(data.files)
        if missing:
            raise KeyError(f"predictions.npz is missing keys: {sorted(missing)}")
        image_paths = data["image_path"]
        predicted_positions = np.asarray(data["t_pred"], dtype=float)

    if predicted_positions.ndim != 2 or predicted_positions.shape[1] != 3:
        raise ValueError("t_pred must have shape (N, 3).")
    if len(image_paths) != len(predicted_positions):
        raise ValueError("image_path and t_pred lengths do not match.")

    prediction_by_name = {
        Path(str(path_value)).name: predicted_positions[index].reshape(3)
        for index, path_value in enumerate(image_paths)
    }
    aligned = np.zeros((len(shirt_filenames), 3), dtype=float)
    valid = np.zeros(len(shirt_filenames), dtype=bool)
    for index, filename in enumerate(shirt_filenames):
        if filename in prediction_by_name:
            aligned[index] = prediction_by_name[filename]
            valid[index] = True
    return aligned, valid


def build_pseudo_velocity(
    positions: Array,
    valid_positions: Array,
    timestamps: Array,
) -> tuple[Array, Array]:
    """Build pseudo velocity with the same central/forward/backward fallback rules."""
    positions = np.asarray(positions, dtype=float)
    valid_positions = np.asarray(valid_positions, dtype=bool).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if positions.shape != (len(timestamps), 3):
        raise ValueError("positions must have shape (N, 3).")
    if valid_positions.shape != (len(timestamps),):
        raise ValueError("valid_positions must have shape (N,).")

    velocity = np.zeros_like(positions)
    valid_velocity = np.zeros(len(timestamps), dtype=bool)
    for index in range(len(timestamps)):
        if not valid_positions[index]:
            continue
        central = index > 0 and index + 1 < len(timestamps) and valid_positions[index - 1] and valid_positions[index + 1]
        forward = index + 1 < len(timestamps) and valid_positions[index + 1]
        backward = index > 0 and valid_positions[index - 1]
        if central:
            dt = timestamps[index + 1] - timestamps[index - 1]
            if dt > 0.0:
                velocity[index] = (positions[index + 1] - positions[index - 1]) / dt
                valid_velocity[index] = True
                continue
        if forward:
            dt = timestamps[index + 1] - timestamps[index]
            if dt > 0.0:
                velocity[index] = (positions[index + 1] - positions[index]) / dt
                valid_velocity[index] = True
                continue
        if backward:
            dt = timestamps[index] - timestamps[index - 1]
            if dt > 0.0:
                velocity[index] = (positions[index] - positions[index - 1]) / dt
                valid_velocity[index] = True
    return velocity, valid_velocity


def create_nn_observations(
    *,
    timestamps: Array,
    positions: Array,
    valid_positions: Array,
    covariance_position: Array,
    observer_id: str,
    target_id: str,
    frame: str,
    use_pseudo_velocity: bool = False,
    covariance_velocity: Array | None = None,
    confidence: float | Array = 1.0,
) -> list[Observation]:
    """Convert aligned NN predictions into documented Observation objects."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float)
    valid_positions = np.asarray(valid_positions, dtype=bool).reshape(-1)
    covariance_position = np.asarray(covariance_position, dtype=float).reshape(3, 3)
    confidence_values = _confidence_array(confidence, len(timestamps))

    if positions.shape != (len(timestamps), 3):
        raise ValueError("positions must have shape (N, 3).")
    if valid_positions.shape != (len(timestamps),):
        raise ValueError("valid_positions must have shape (N,).")

    if use_pseudo_velocity:
        if covariance_velocity is None:
            raise ValueError("covariance_velocity is required when pseudo velocity is enabled.")
        covariance_velocity = np.asarray(covariance_velocity, dtype=float).reshape(3, 3)
        pseudo_velocity, valid_velocity = build_pseudo_velocity(
            positions, valid_positions, timestamps
        )
        covariance = np.block(
            [[covariance_position, np.zeros((3, 3))], [np.zeros((3, 3)), covariance_velocity]]
        )
    else:
        pseudo_velocity = None
        valid_velocity = np.ones(len(timestamps), dtype=bool)
        covariance = covariance_position

    observations: list[Observation] = []
    for index, timestamp in enumerate(timestamps):
        measurement = (
            np.hstack((positions[index], pseudo_velocity[index]))
            if use_pseudo_velocity
            else positions[index].copy()
        )
        observations.append(
            Observation(
                timestamp=float(timestamp),
                observer_id=observer_id,
                target_id=target_id,
                modality="OPTICAL",
                source_type="LEARNING",
                measurement=measurement,
                covariance=covariance.copy(),
                confidence=float(confidence_values[index]),
                frame=str(frame).upper(),
                valid_flag=bool(valid_positions[index] and valid_velocity[index]),
                metadata={
                    "measurement_type": "RELATIVE_POSITION_VELOCITY" if use_pseudo_velocity else "RELATIVE_POSITION",
                    "prediction_source": str("predictions.npz"),
                },
            )
        )
    return observations


def _confidence_array(confidence: float | Array, length: int) -> Array:
    values = np.asarray(confidence, dtype=float)
    if values.ndim == 0:
        values = np.full(length, float(values), dtype=float)
    values = values.reshape(-1)
    if values.shape != (length,):
        raise ValueError("confidence must be a scalar or shape (N,).")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("confidence values must be in (0, 1].")
    return values
