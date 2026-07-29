from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interfaces.data_objects import InitialState, ModuleInput, Observation
from orbital_core.coordinates import rotate_pri_to_eci
from orbital_core.dynamics import make_process_noise


Array = np.ndarray


@dataclass(frozen=True)
class ShirtOrbitDataset:
    """Orbit-related SHIRT data required by the state-estimation pipeline."""

    trajectory_name: str
    timestamps: Array
    filenames: list[str]
    relative_state_eci: Array
    chief_state_eci: Array
    target_state_eci: Array
    relative_position_spri: Array
    relative_velocity_spri: Array
    q_eci2pri: Array
    preprocess_summary: dict[str, Any]


def load_shirt_orbit_dataset(
    metadata_path: str | Path,
    roe_path: str | Path,
    trajectory_name: str,
) -> ShirtOrbitDataset:
    """Load SHIRT metadata/trajectory JSON without changing legacy conventions.

    The quaternion handling intentionally follows the verified legacy behavior:
    q_eci2pri is numerically used as an active PRI-to-ECI rotation when building
    the ECI relative state. This convention is recorded in preprocess_summary.
    """
    metadata = _load_json(metadata_path)
    roe_data = _load_json(roe_path)
    if not isinstance(roe_data, list) or not roe_data:
        raise ValueError("roe_path must contain a non-empty JSON list.")

    filenames = [str(item["filename"]) for item in roe_data]
    sim_info = metadata["pSim"]
    absolute_state = metadata["sAbsState"]
    relative_state = metadata["tRelState"]

    timestamps = np.arange(len(filenames), dtype=float) * float(sim_info.get("cam_step", 1.0))
    relative_state_spri = np.asarray(relative_state["rv_scom2tcom_spri"], dtype=float)
    chief_state_eci = np.asarray(absolute_state["rv_eci2com_eci"], dtype=float)
    q_eci2pri = np.asarray(absolute_state["q_eci2pri"], dtype=float)

    _require_shape(relative_state_spri, (len(timestamps), 6), "rv_scom2tcom_spri")
    _require_shape(chief_state_eci, (len(timestamps), 6), "rv_eci2com_eci")
    _require_shape(q_eci2pri, (len(timestamps), 4), "q_eci2pri")

    relative_position_spri = relative_state_spri[:, :3]
    relative_velocity_spri = relative_state_spri[:, 3:]
    relative_position_eci = np.empty_like(relative_position_spri)
    relative_velocity_eci = np.empty_like(relative_velocity_spri)
    for index in range(len(timestamps)):
        relative_position_eci[index] = rotate_pri_to_eci(
            relative_position_spri[index], q_eci2pri[index]
        )
        relative_velocity_eci[index] = rotate_pri_to_eci(
            relative_velocity_spri[index], q_eci2pri[index]
        )

    relative_state_eci = np.hstack((relative_position_eci, relative_velocity_eci))
    target_state_eci = chief_state_eci + relative_state_eci

    dt = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    velocity_fd = _finite_difference(relative_position_eci, dt)
    consistency_rmse = float(
        np.sqrt(np.mean(np.sum((relative_velocity_eci - velocity_fd) ** 2, axis=1)))
    )

    return ShirtOrbitDataset(
        trajectory_name=str(trajectory_name),
        timestamps=timestamps,
        filenames=filenames,
        relative_state_eci=relative_state_eci,
        chief_state_eci=chief_state_eci,
        target_state_eci=target_state_eci,
        relative_position_spri=relative_position_spri,
        relative_velocity_spri=relative_velocity_spri,
        q_eci2pri=q_eci2pri,
        preprocess_summary={
            "quaternion_order": "wxyz",
            "quaternion_numeric_use": "pri_to_eci_active",
            "velocity_rotation_rule": "ignore_frame_angular_rate",
            "velocity_consistency_rmse_mps": consistency_rmse,
        },
    )


def build_shirt_module_input(
    dataset: ShirtOrbitDataset,
    *,
    node_id: str,
    target_id: str,
    process_noise_acceleration: float,
    initial_position_std: float,
    initial_velocity_std: float,
    observations: list[Observation],
    filter_config: dict[str, Any] | None = None,
    modalities_config: dict[str, Any] | None = None,
    initial_state_estimate: Array | None = None,
) -> ModuleInput:
    """Create the documented ModuleInput with minimal interface additions."""
    timestamps = dataset.timestamps
    if len(timestamps) < 2:
        raise ValueError("At least two SHIRT samples are required for filtering.")
    dt = float(timestamps[1] - timestamps[0])
    initial_state = (
        np.asarray(initial_state_estimate, dtype=float).reshape(6)
        if initial_state_estimate is not None
        else dataset.relative_state_eci[0].copy()
    )
    covariance = np.diag(
        [initial_position_std**2] * 3 + [initial_velocity_std**2] * 3
    )
    config: dict[str, Any] = {
        "runtime": {
            "timestamps": timestamps,
            "chief_state_history_eci": dataset.chief_state_eci,
            "q_eci2pri_history": dataset.q_eci2pri,
            "node_id": node_id,
            "trajectory_name": dataset.trajectory_name,
            "preprocess_summary": dataset.preprocess_summary,
        },
        "filter": {
            "process_noise": make_process_noise(dt, process_noise_acceleration),
            **(filter_config or {}),
        },
        "modalities": modalities_config or {},
    }
    return ModuleInput(
        initial_state=InitialState(
            target_id=target_id,
            timestamp=float(timestamps[0]),
            state_estimate=initial_state,
            covariance=covariance,
        ),
        sensor_measurements=observations,
        config=config,
    )


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_shape(array: Array, expected: tuple[int, ...], name: str) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}.")


def _finite_difference(position: Array, dt: float) -> Array:
    if dt <= 0.0:
        raise ValueError("Time step must be positive.")
    if len(position) == 1:
        return np.zeros_like(position)
    velocity = np.zeros_like(position)
    velocity[1:-1] = (position[2:] - position[:-2]) / (2.0 * dt)
    velocity[0] = (position[1] - position[0]) / dt
    velocity[-1] = (position[-1] - position[-2]) / dt
    return velocity
