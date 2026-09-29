from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from adapters.external_scene_input import (
    ExternalSceneInitialConditions,
    adapt_external_scene_input,
)

from .data_contracts import MultiTargetInput, ObserverState, TargetInitialState


DEFAULT_OBSERVER_COVARIANCE_DIAGONAL = (10.0, 10.0, 10.0, 0.02, 0.02, 0.02)
DEFAULT_TARGET_COVARIANCE_DIAGONAL = (100.0, 100.0, 100.0, 0.2, 0.2, 0.2)


def load_external_multitarget_input(path: str | Path) -> MultiTargetInput:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return adapt_external_multitarget_input(payload)


def adapt_external_multitarget_input(payload: Mapping[str, Any]) -> MultiTargetInput:
    """Adapt the existing external scene JSON plus ``estimationConfig``.

    The orbital-element ``satellites`` array remains authoritative. The new
    section only assigns observer/target roles and estimator initialization;
    it does not perform IOD or infer an unknown target identity.
    """

    scene = adapt_external_scene_input(payload)
    config = payload.get("estimationConfig")
    if not isinstance(config, Mapping):
        raise ValueError("External multi-target input requires estimationConfig.")
    observer_ids = _identifier_list(config.get("observerIds"), "observerIds")
    targets_payload = config.get("targets")
    if not isinstance(targets_payload, list) or not targets_payload:
        raise ValueError("estimationConfig.targets must be a nonempty array.")
    target_specs = {}
    for value in targets_payload:
        if not isinstance(value, Mapping):
            raise TypeError("Every estimationConfig.targets entry must be an object.")
        node_id = _required_text(value, "nodeId")
        if node_id in target_specs:
            raise ValueError(f"Duplicate target nodeId {node_id!r}.")
        target_specs[node_id] = value
    observer_covariance = _diagonal_covariance(
        config.get(
            "observerCovarianceDiagonal", DEFAULT_OBSERVER_COVARIANCE_DIAGONAL,
        ),
        "observerCovarianceDiagonal",
    )
    default_target_diagonal = config.get(
        "defaultTargetCovarianceDiagonal", DEFAULT_TARGET_COVARIANCE_DIAGONAL,
    )
    return adapt_external_scene_to_multitarget(
        scene,
        observer_ids=observer_ids,
        target_specs=target_specs,
        observer_covariance=observer_covariance,
        default_target_covariance_diagonal=default_target_diagonal,
    )


def adapt_external_scene_to_multitarget(
    scene: ExternalSceneInitialConditions,
    *,
    observer_ids: tuple[str, ...] | list[str],
    target_specs: Mapping[str, Mapping[str, Any]],
    observer_covariance: np.ndarray | None = None,
    default_target_covariance_diagonal=DEFAULT_TARGET_COVARIANCE_DIAGONAL,
) -> MultiTargetInput:
    observers = tuple(map(str, observer_ids))
    targets = dict(target_specs)
    if not observers:
        raise ValueError("At least one observer ID is required.")
    if len(set(observers)) != len(observers):
        raise ValueError("observerIds must be unique.")
    if not targets:
        raise ValueError("At least one target specification is required.")
    overlap = set(observers) & set(targets)
    if overlap:
        raise ValueError(f"Observer and target roles overlap: {sorted(overlap)}")
    known = set(scene.initial_state_by_node)
    unknown = (set(observers) | set(targets)) - known
    if unknown:
        raise ValueError(f"Role configuration references unknown satellites: {sorted(unknown)}")
    observer_cov = (
        _diagonal_covariance(
            DEFAULT_OBSERVER_COVARIANCE_DIAGONAL,
            "observerCovarianceDiagonal",
        )
        if observer_covariance is None
        else _covariance(observer_covariance, "observer_covariance")
    )
    default_target_diagonal = _covariance_diagonal(
        default_target_covariance_diagonal,
        "defaultTargetCovarianceDiagonal",
    )
    timestamp = float(scene.timestamps[0])
    observer_states = {
        node_id: ObserverState(
            observer_id=node_id,
            timestamp=timestamp,
            state_eci=scene.initial_state_by_node[node_id],
            covariance_eci=observer_cov,
        )
        for node_id in observers
    }
    target_states = {}
    sensor_availability = {}
    for node_id, spec in targets.items():
        track_id = str(spec.get("trackId", f"track-{node_id}")).strip()
        if not track_id:
            raise ValueError(f"Target {node_id!r} trackId must be nonempty.")
        diagonal = _covariance_diagonal(
            spec.get("covarianceDiagonal", default_target_diagonal),
            f"targets[{node_id}].covarianceDiagonal",
        )
        target_states[node_id] = TargetInitialState(
            target_id=node_id,
            track_id=track_id,
            timestamp=timestamp,
            state_eci=scene.initial_state_by_node[node_id],
            covariance_eci=np.diag(diagonal),
        )
        availability = spec.get("sensorAvailability", {})
        if not isinstance(availability, Mapping):
            raise TypeError(f"Target {node_id!r} sensorAvailability must be an object.")
        sensor_availability[node_id] = {
            str(key).upper(): bool(value) for key, value in availability.items()
        }
    return MultiTargetInput(
        scene_id=scene.scene_id,
        observer_states=observer_states,
        target_initial_states=target_states,
        config={
            "sceneName": scene.name,
            "startTimeMs": scene.start_time_ms,
            "endTimeMs": scene.end_time_ms,
            "stepSeconds": scene.step_seconds,
            "timestamps": scene.timestamps.copy(),
            "sensorAvailabilityByTarget": sensor_availability,
            "initializationMethod": "EXTERNAL_STRUCTURED_ELEMENTS",
            "initialOrbitDeterminationPerformed": False,
        },
    )


def _identifier_list(value, field_name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"estimationConfig.{field_name} must be a nonempty array.")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"estimationConfig.{field_name} contains an empty ID.")
    return result


def _required_text(payload, field_name):
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string.")
    return value.strip()


def _covariance_diagonal(value, field_name):
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (6,) or np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{field_name} must contain six finite nonnegative variances.")
    return result


def _diagonal_covariance(value, field_name):
    return np.diag(_covariance_diagonal(value, field_name))


def _covariance(value, field_name):
    result = np.asarray(value, dtype=float)
    if result.shape != (6, 6) or np.any(~np.isfinite(result)):
        raise ValueError(f"{field_name} must be a finite 6x6 matrix.")
    if not np.allclose(result, result.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{field_name} must be symmetric.")
    if np.linalg.eigvalsh(result).min() < -1e-10:
        raise ValueError(f"{field_name} must be positive semidefinite.")
    return result
