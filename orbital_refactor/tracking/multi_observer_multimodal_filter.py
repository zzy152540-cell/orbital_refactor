"""Unified EKF update from simultaneous radar and line-of-sight data."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.initial_orbit.data_contracts import IODObservation
from tracking.multi_observer_bearing_filter import _tangent_basis


@dataclass(frozen=True)
class UnifiedMultimodalUpdate:
    timestamp: float
    state_eci: np.ndarray
    covariance_eci: np.ndarray
    observer_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    modality_counts: dict[str, int]
    information_capabilities: tuple[str, ...]
    observation_count: int
    measurement_dimension: int
    nis: float


def unified_multi_observer_multimodal_update(
    prior,
    observations,
    *,
    timestamp: float,
    process_noise_acceleration: float = 1e-4,
) -> UnifiedMultimodalUpdate:
    """Propagate one target prior and jointly update it with RADAR and LOS.

    Each physical observation enters exactly once. RADAR contributes range and
    range rate, while optical/infrared LOS contributes two tangent-plane
    residuals. All Jacobians are evaluated at the same predicted state.
    """

    epoch = float(timestamp)
    values = _apply_radial_proxy_failover_policy(tuple(observations))
    if not values:
        raise ValueError("At least one multimodal observation is required.")
    if any(
        not isinstance(item, (UnlabeledIODObservation, IODObservation))
        for item in values
    ):
        raise TypeError("Unified update requires IOD observations.")
    if any(item.modality not in {"RADAR", "LOS"} for item in values):
        raise ValueError("Unified update accepts RADAR and LOS only.")
    if any(
        not item.valid_flag or not np.isclose(item.timestamp, epoch)
        for item in values
    ):
        raise ValueError("Every observation must be valid and epoch-aligned.")

    state, covariance = _predict_prior(
        prior,
        epoch,
        process_noise_acceleration=process_noise_acceleration,
    )
    jacobians = []
    innovations = []
    covariances = []
    modality_counts: dict[str, int] = {}
    information_capabilities: set[str] = set()
    measurement_ids = []
    for item in values:
        jacobian, innovation, measurement_covariance = _measurement_block(
            item, state,
        )
        jacobians.append(jacobian)
        innovations.append(innovation)
        covariances.append(measurement_covariance)
        source_modality = str(
            item.metadata.get("sourceModality", item.modality)
        ).upper()
        modality_counts[source_modality] = (
            modality_counts.get(source_modality, 0) + 1
        )
        if item.modality == "RADAR" or item.metadata.get("rangeProxyM") is not None:
            information_capabilities.add("RADIAL_RANGE")
        if (
            item.modality == "RADAR"
            or item.metadata.get("rangeRateProxyMps") is not None
        ):
            information_capabilities.add("RADIAL_RANGE_RATE")
        identifier = item.metadata.get("sourceMessageId")
        if identifier is not None:
            measurement_ids.append(str(identifier))

    measurement_matrix = np.vstack(jacobians)
    innovation = np.concatenate(innovations)
    measurement_covariance = _block_diagonal(covariances)
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + measurement_covariance
    )
    innovation_covariance = _symmetrize(innovation_covariance)
    inverse_innovation = np.linalg.pinv(innovation_covariance)
    gain = covariance @ measurement_matrix.T @ inverse_innovation
    updated_state = state + gain @ innovation
    identity = np.eye(6)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = _symmetrize(
        residual_matrix @ covariance @ residual_matrix.T
        + gain @ measurement_covariance @ gain.T
    )
    return UnifiedMultimodalUpdate(
        timestamp=epoch,
        state_eci=updated_state,
        covariance_eci=updated_covariance,
        observer_ids=tuple(sorted({item.observer_id for item in values})),
        measurement_ids=tuple(dict.fromkeys(measurement_ids)),
        modality_counts=dict(sorted(modality_counts.items())),
        information_capabilities=tuple(sorted(information_capabilities)),
        observation_count=len(values),
        measurement_dimension=int(innovation.size),
        nis=float(innovation @ inverse_innovation @ innovation),
    )


def sequential_multi_observer_multimodal_update(
    prior,
    observations,
    *,
    timestamp: float,
    process_noise_acceleration: float = 1e-4,
) -> UnifiedMultimodalUpdate:
    """Apply independent multimodal blocks in a deterministic EKF sequence.

    The target prior is propagated exactly once.  RADAR, INFRARED and OPTICAL
    blocks are then applied in that order and relinearized at the posterior
    produced by the previous block.  This is a conventional no-CANN baseline,
    distinct from the same-linearization stacked update above.
    """

    epoch = float(timestamp)
    values = _apply_radial_proxy_failover_policy(tuple(observations))
    if not values:
        raise ValueError("At least one multimodal observation is required.")
    if any(
        not isinstance(item, (UnlabeledIODObservation, IODObservation))
        for item in values
    ):
        raise TypeError("Sequential update requires IOD observations.")
    if any(item.modality not in {"RADAR", "LOS"} for item in values):
        raise ValueError("Sequential update accepts RADAR and LOS only.")
    if any(
        not item.valid_flag or not np.isclose(item.timestamp, epoch)
        for item in values
    ):
        raise ValueError("Every observation must be valid and epoch-aligned.")

    order = {"RADAR": 0, "INFRARED": 1, "OPTICAL": 2}
    values = tuple(sorted(
        values,
        key=lambda item: (
            order.get(str(item.metadata.get(
                "sourceModality", item.modality,
            )).upper(), 3),
            str(item.observer_id),
            str(item.metadata.get("sourceMessageId", "")),
        ),
    ))
    state, covariance = _predict_prior(
        prior,
        epoch,
        process_noise_acceleration=process_noise_acceleration,
    )
    identity = np.eye(6)
    modality_counts: dict[str, int] = {}
    information_capabilities: set[str] = set()
    measurement_ids = []
    measurement_dimension = 0
    total_nis = 0.0
    for item in values:
        jacobian, innovation, measurement_covariance = _measurement_block(
            item, state,
        )
        innovation_covariance = _symmetrize(
            jacobian @ covariance @ jacobian.T + measurement_covariance
        )
        inverse_innovation = np.linalg.pinv(innovation_covariance)
        gain = covariance @ jacobian.T @ inverse_innovation
        state = state + gain @ innovation
        residual_matrix = identity - gain @ jacobian
        covariance = _symmetrize(
            residual_matrix @ covariance @ residual_matrix.T
            + gain @ measurement_covariance @ gain.T
        )
        total_nis += float(innovation @ inverse_innovation @ innovation)
        measurement_dimension += int(innovation.size)
        source_modality = str(
            item.metadata.get("sourceModality", item.modality)
        ).upper()
        modality_counts[source_modality] = (
            modality_counts.get(source_modality, 0) + 1
        )
        if item.modality == "RADAR" or item.metadata.get("rangeProxyM") is not None:
            information_capabilities.add("RADIAL_RANGE")
        if (
            item.modality == "RADAR"
            or item.metadata.get("rangeRateProxyMps") is not None
        ):
            information_capabilities.add("RADIAL_RANGE_RATE")
        identifier = item.metadata.get("sourceMessageId")
        if identifier is not None:
            measurement_ids.append(str(identifier))

    return UnifiedMultimodalUpdate(
        timestamp=epoch,
        state_eci=state,
        covariance_eci=covariance,
        observer_ids=tuple(sorted({item.observer_id for item in values})),
        measurement_ids=tuple(dict.fromkeys(measurement_ids)),
        modality_counts=dict(sorted(modality_counts.items())),
        information_capabilities=tuple(sorted(information_capabilities)),
        observation_count=len(values),
        measurement_dimension=measurement_dimension,
        nis=total_nis,
    )


def _apply_radial_proxy_failover_policy(observations):
    """Use one radial source per detection when EO proxies are fallbacks.

    RADAR is primary.  When it is unavailable, the lowest-priority-numbered
    electro-optical proxy is retained while every LOS direction measurement
    remains in the update.  The default ``independent`` proxy policy is not
    changed by this helper.
    """

    values = list(observations)
    groups: dict[tuple[str, float, str], list[int]] = {}
    for index, item in enumerate(values):
        group_id = getattr(item, "detection_group_id", None)
        if group_id is None:
            group_id = item.metadata.get("detectionGroupId")
        if group_id is None:
            continue
        key = (item.observer_id, float(item.timestamp), str(group_id))
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        candidates = [
            index for index in indices
            if values[index].metadata.get("radialProxyFusionPolicy")
            == "radar_primary_failover"
            and (
                values[index].metadata.get("rangeProxyM") is not None
                or values[index].metadata.get("rangeRateProxyMps") is not None
            )
        ]
        if not candidates:
            continue
        radar_present = any(values[index].modality == "RADAR" for index in indices)
        selected_index = None
        if not radar_present:
            selected_index = min(
                candidates,
                key=lambda index: (
                    int(values[index].metadata.get(
                        "radialProxyFailoverPriority", 100,
                    )),
                    str(values[index].metadata.get(
                        "sourceModality", values[index].modality,
                    )),
                ),
            )
        for index in candidates:
            metadata = dict(values[index].metadata)
            if index == selected_index:
                metadata["radialProxyUsed"] = True
                metadata["radialProxySelectionReason"] = "EO_FAILOVER_SELECTED"
            else:
                for key in (
                    "rangeProxyM",
                    "rangeProxyVarianceM2",
                    "rangeProxySource",
                    "rangeProxyIndependentNoiseAssumed",
                    "rangeRateProxyMps",
                    "rangeRateProxyVarianceM2ps2",
                    "rangeRateProxySource",
                    "rangeRateProxyIndependentNoiseAssumed",
                ):
                    metadata.pop(key, None)
                metadata["radialProxyUsed"] = False
                metadata["radialProxySelectionReason"] = (
                    "RADAR_PRIMARY" if radar_present else "EO_STANDBY"
                )
            values[index] = replace(values[index], metadata=metadata)
    return tuple(values)


def _measurement_block(item, state):
    relative_position = state[:3] - item.observer_state_eci[:3]
    relative_velocity = state[3:] - item.observer_state_eci[3:]
    distance = float(np.linalg.norm(relative_position))
    if distance <= 0.0:
        raise ValueError("Observer and target positions must differ.")
    line = relative_position / distance
    if item.modality == "RADAR":
        range_rate = float(line @ relative_velocity)
        predicted = np.array([distance, range_rate], dtype=float)
        jacobian = np.zeros((2, 6), dtype=float)
        jacobian[0, :3] = line
        jacobian[1, :3] = (
            relative_velocity - range_rate * line
        ) / distance
        jacobian[1, 3:] = line
        return (
            jacobian,
            np.asarray(item.measurement, dtype=float) - predicted,
            _regularize_covariance(item.covariance),
        )

    tangent = np.vstack(_tangent_basis(line))
    measured_line = np.asarray(item.measurement, dtype=float).reshape(3)
    jacobian = np.zeros((2, 6), dtype=float)
    jacobian[:, :3] = (
        tangent @ (np.eye(3) - np.outer(line, line)) / distance
    )
    line_covariance = item.metadata.get("lineOfSightCovarianceEci")
    tangent_covariance = (
        np.asarray(item.covariance, dtype=float)
        if line_covariance is None
        else tangent
        @ np.asarray(line_covariance, dtype=float).reshape(3, 3)
        @ tangent.T
    )
    los_innovation = tangent @ (measured_line - line)
    range_proxy = item.metadata.get("rangeProxyM")
    range_proxy_variance = item.metadata.get("rangeProxyVarianceM2")
    range_rate_proxy = item.metadata.get("rangeRateProxyMps")
    range_rate_proxy_variance = item.metadata.get(
        "rangeRateProxyVarianceM2ps2"
    )
    if all(value is None for value in (
        range_proxy,
        range_proxy_variance,
        range_rate_proxy,
        range_rate_proxy_variance,
    )):
        return (
            jacobian,
            los_innovation,
            _regularize_covariance(tangent_covariance),
        )
    if (range_proxy is None) != (range_proxy_variance is None):
        raise ValueError(
            "LOS range proxy requires both rangeProxyM and "
            "rangeProxyVarianceM2 metadata."
        )
    if (range_rate_proxy is None) != (range_rate_proxy_variance is None):
        raise ValueError(
            "LOS range-rate proxy requires both rangeRateProxyMps and "
            "rangeRateProxyVarianceM2ps2 metadata."
        )
    proxy_rows = []
    proxy_innovations = []
    proxy_variances = []
    if range_proxy is not None:
        measured_range = float(range_proxy)
        range_variance = float(range_proxy_variance)
        if (
            not np.isfinite(measured_range)
            or measured_range <= 0.0
            or not np.isfinite(range_variance)
            or range_variance <= 0.0
        ):
            raise ValueError(
                "LOS range proxy value and variance must be positive."
            )
        range_row = np.zeros(6, dtype=float)
        range_row[:3] = line
        proxy_rows.append(range_row)
        proxy_innovations.append(measured_range - distance)
        proxy_variances.append(range_variance)
    if range_rate_proxy is not None:
        measured_range_rate = float(range_rate_proxy)
        range_rate_variance = float(range_rate_proxy_variance)
        if (
            not np.isfinite(measured_range_rate)
            or not np.isfinite(range_rate_variance)
            or range_rate_variance <= 0.0
        ):
            raise ValueError(
                "LOS range-rate proxy must be finite with positive variance."
            )
        predicted_range_rate = float(line @ relative_velocity)
        range_rate_row = np.zeros(6, dtype=float)
        range_rate_row[:3] = (
            relative_velocity - predicted_range_rate * line
        ) / distance
        range_rate_row[3:] = line
        proxy_rows.append(range_rate_row)
        proxy_innovations.append(measured_range_rate - predicted_range_rate)
        proxy_variances.append(range_rate_variance)
    augmented_dimension = 2 + len(proxy_rows)
    augmented_jacobian = np.zeros((augmented_dimension, 6), dtype=float)
    augmented_jacobian[:2] = jacobian
    augmented_jacobian[2:] = np.asarray(proxy_rows, dtype=float)
    augmented_covariance = np.zeros(
        (augmented_dimension, augmented_dimension), dtype=float,
    )
    augmented_covariance[:2, :2] = tangent_covariance
    augmented_covariance[2:, 2:] = np.diag(proxy_variances)
    return (
        augmented_jacobian,
        np.concatenate((los_innovation, proxy_innovations)),
        _regularize_covariance(augmented_covariance),
    )


def _predict_prior(prior, timestamp, *, process_noise_acceleration):
    dt = float(timestamp - prior.timestamp)
    if dt <= 0.0:
        raise ValueError("Unified update timestamp must follow the prior timestamp.")
    initial_state = np.asarray(prior.state_eci, dtype=float).reshape(6)
    initial_covariance = np.asarray(prior.covariance_eci, dtype=float).reshape(6, 6)
    propagate = lambda value: rk4_step_absolute(value, dt)
    transition = numerical_jacobian_discrete(propagate, initial_state)
    state = propagate(initial_state)
    covariance = (
        transition @ initial_covariance @ transition.T
        + make_process_noise(dt, process_noise_acceleration)
    )
    return state, _symmetrize(covariance)


def _block_diagonal(matrices):
    sizes = [matrix.shape[0] for matrix in matrices]
    result = np.zeros((sum(sizes), sum(sizes)), dtype=float)
    offset = 0
    for matrix, size in zip(matrices, sizes, strict=True):
        result[offset:offset + size, offset:offset + size] = matrix
        offset += size
    return result


def _regularize_covariance(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Measurement covariance must be square.")
    covariance = _symmetrize(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = np.finfo(float).eps * scale
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _symmetrize(matrix):
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)
