from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from cooperative.network_schmidt_runner import NetworkSchmidtHistory
from interfaces.data_objects import ObservationMessage
from orbital_core.inter_satellite_model import RelativeMeasurementModel

Array = np.ndarray


@dataclass(frozen=True)
class RelativeErrorProjection:
    node_id: str
    neighbor_id: str
    timestamp: float
    modality: str
    information_id: str
    active_state_error_norm: float
    neighbor_state_error_norm: float
    active_projection: Array
    neighbor_projection: Array

    @property
    def total_linearized_projection(self) -> Array:
        return self.active_projection + self.neighbor_projection


@dataclass(frozen=True)
class RelativeUpdateTruthDecomposition:
    node_id: str
    neighbor_id: str
    timestamp: float
    modalities: tuple[str, ...]
    information_ids: tuple[str, ...]
    active_projection: tuple[float, ...]
    neighbor_projection: tuple[float, ...]
    innovation: tuple[float, ...]
    unexplained_innovation: tuple[float, ...]
    active_error_norm_before: float
    active_error_norm_after: float
    neighbor_to_measurement_covariance_trace_ratio: float
    velocity_injection_risk: float
    position_injection_risk: float

    @property
    def active_error_norm_change(self) -> float:
        return self.active_error_norm_after - self.active_error_norm_before


@dataclass(frozen=True)
class RelativeUpdateModalitySummary:
    modality: str
    sample_count: int
    mean_neighbor_to_innovation_ratio: float
    active_error_worsening_fraction: float
    mean_active_error_norm_change: float
    mean_active_projection_norm: float
    mean_neighbor_projection_norm: float
    mean_innovation_norm: float
    mean_neighbor_to_measurement_covariance_trace_ratio: float
    mean_velocity_injection_risk: float
    mean_position_injection_risk: float


def relative_error_projection_diagnostics(
    *,
    history: NetworkSchmidtHistory,
    truth_by_node: Mapping[str, Array],
    observations: Iterable[ObservationMessage],
) -> tuple[RelativeErrorProjection, ...]:
    """Project epoch-posterior active and consider-mean errors to measurement space.

    The consider mean is unchanged by a Schmidt measurement update, while the
    active mean is the epoch posterior. This read-only diagnostic is therefore
    suitable for screening neighbor-error projection, not for reconstructing
    the exact pre-update innovation of each observation.
    """

    time_index = {
        float(timestamp): index
        for index, timestamp in enumerate(history.timestamps)
    }
    records = []
    for observation in observations:
        node_id = observation.observer_id
        neighbor_id = observation.target_id
        if node_id not in history.neighbor_state_history_by_node:
            continue
        if neighbor_id not in history.neighbor_state_history_by_node[node_id]:
            continue
        try:
            index = time_index[float(observation.timestamp)]
        except KeyError:
            continue
        active_mean = history.active_state_history_by_node[node_id][index]
        neighbor_mean = history.neighbor_state_history_by_node[
            node_id
        ][neighbor_id][index]
        active_error = active_mean - np.asarray(
            truth_by_node[node_id][index], dtype=float
        )
        neighbor_error = neighbor_mean - np.asarray(
            truth_by_node[neighbor_id][index], dtype=float
        )
        quaternion = (
            observation.metadata.get("quaternion_i2b_wxyz")
            if observation.frame.upper() == "BODY" else None
        )
        active_jacobian, neighbor_jacobian = RelativeMeasurementModel(
            observation.modality, observation.frame
        ).jacobians(
            active_mean,
            neighbor_mean,
            quaternion_i2b_wxyz=quaternion,
        )
        records.append(RelativeErrorProjection(
            node_id=node_id,
            neighbor_id=neighbor_id,
            timestamp=float(observation.timestamp),
            modality=observation.modality,
            information_id=observation.information_id,
            active_state_error_norm=float(np.linalg.norm(active_error)),
            neighbor_state_error_norm=float(np.linalg.norm(neighbor_error)),
            active_projection=active_jacobian @ active_error,
            neighbor_projection=neighbor_jacobian @ neighbor_error,
        ))
    return tuple(records)


def relative_update_truth_decomposition(
    *,
    history: NetworkSchmidtHistory,
    truth_by_node: Mapping[str, Array],
) -> tuple[RelativeUpdateTruthDecomposition, ...]:
    """Decompose recorded pre-update innovations against trajectory truth."""

    output = []
    for node_id, epochs in history.relative_update_history_by_node.items():
        for index, records in enumerate(epochs):
            timestamp = float(history.timestamps[index])
            active_truth = np.asarray(
                truth_by_node[node_id][index], dtype=float
            )
            for record in records:
                observer = str(record["observer_id"])
                target = str(record["target_id"])
                neighbor_id = target if observer == node_id else observer
                neighbor_truth = np.asarray(
                    truth_by_node[neighbor_id][index], dtype=float
                )
                active_error = record["prior_active_state"] - active_truth
                neighbor_error = (
                    record["prior_neighbor_state"] - neighbor_truth
                )
                active_projection = record["active_jacobian"] @ active_error
                neighbor_projection = (
                    record["neighbor_jacobian"] @ neighbor_error
                )
                innovation = np.asarray(record["innovation"], dtype=float)
                correction = np.asarray(
                    record["active_correction"], dtype=float
                )
                gain = np.asarray(record["active_gain"], dtype=float)
                innovation_covariance = np.asarray(
                    record["innovation_covariance"], dtype=float
                )
                prior_covariance = np.asarray(
                    record["prior_active_covariance"], dtype=float
                )
                output.append(RelativeUpdateTruthDecomposition(
                    node_id=node_id,
                    neighbor_id=neighbor_id,
                    timestamp=timestamp,
                    modalities=tuple(record["modalities"]),
                    information_ids=tuple(record["information_ids"]),
                    active_projection=_float_tuple(active_projection),
                    neighbor_projection=_float_tuple(neighbor_projection),
                    innovation=_float_tuple(innovation),
                    unexplained_innovation=_float_tuple(
                        innovation + active_projection + neighbor_projection
                    ),
                    active_error_norm_before=float(
                        np.linalg.norm(active_error)
                    ),
                    active_error_norm_after=float(
                        np.linalg.norm(active_error + correction)
                    ),
                    neighbor_to_measurement_covariance_trace_ratio=float(
                        np.trace(record["projected_neighbor_covariance"])
                        / max(
                            np.trace(record["nominal_measurement_covariance"]),
                            1e-15,
                        )
                    ),
                    velocity_injection_risk=_subspace_injection_risk(
                        gain[3:], innovation_covariance,
                        prior_covariance[3:, 3:],
                    ),
                    position_injection_risk=_subspace_injection_risk(
                        gain[:3], innovation_covariance,
                        prior_covariance[:3, :3],
                    ),
                ))
    return tuple(output)


def summarize_relative_update_truth_diagnostics(
    records: Iterable[RelativeUpdateTruthDecomposition],
) -> tuple[RelativeUpdateModalitySummary, ...]:
    """Aggregate single-modality diagnostic records without hiding sign."""

    grouped: dict[str, list[RelativeUpdateTruthDecomposition]] = {}
    for record in records:
        key = "+".join(record.modalities)
        grouped.setdefault(key, []).append(record)
    summaries = []
    for modality, values in sorted(grouped.items()):
        active = np.asarray([
            np.linalg.norm(item.active_projection) for item in values
        ])
        neighbor = np.asarray([
            np.linalg.norm(item.neighbor_projection) for item in values
        ])
        innovation = np.asarray([
            np.linalg.norm(item.innovation) for item in values
        ])
        changes = np.asarray([
            item.active_error_norm_change for item in values
        ])
        summaries.append(RelativeUpdateModalitySummary(
            modality=modality,
            sample_count=len(values),
            mean_neighbor_to_innovation_ratio=float(np.mean(
                neighbor / np.maximum(innovation, 1e-15)
            )),
            active_error_worsening_fraction=float(np.mean(changes > 0.0)),
            mean_active_error_norm_change=float(np.mean(changes)),
            mean_active_projection_norm=float(np.mean(active)),
            mean_neighbor_projection_norm=float(np.mean(neighbor)),
            mean_innovation_norm=float(np.mean(innovation)),
            mean_neighbor_to_measurement_covariance_trace_ratio=float(
                np.mean([
                    item.neighbor_to_measurement_covariance_trace_ratio
                    for item in values
                ])
            ),
            mean_velocity_injection_risk=float(np.mean([
                item.velocity_injection_risk for item in values
            ])),
            mean_position_injection_risk=float(np.mean([
                item.position_injection_risk for item in values
            ])),
        ))
    return tuple(summaries)


def _float_tuple(values: Array) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def _subspace_injection_risk(gain, innovation_covariance, prior_covariance):
    injected = gain @ innovation_covariance @ gain.T
    return float(
        np.trace(injected) / max(np.trace(prior_covariance), 1e-15)
    )
