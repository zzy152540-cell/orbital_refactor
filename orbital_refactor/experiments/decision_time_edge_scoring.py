from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import (
    GraphObservation,
    TopologyAction,
    UndirectedEdge,
)


@dataclass(frozen=True)
class DecisionTimeEdgeScore:
    edge: UndirectedEdge
    endpoint_position_uncertainty: float
    projected_position_uncertainty: float
    approximate_trace_reduction: float
    approximate_logdet_reduction: float
    negative_distance: float
    observation_age: float
    recent_mean_nis: float
    negative_recent_mean_nis: float
    nis_calibration_quality: float
    nis_sample_count: float
    negative_consecutive_anomaly_count: float


@dataclass(frozen=True)
class DecisionTimeTopologyScore:
    endpoint_position_uncertainty: float
    projected_position_uncertainty: float
    approximate_trace_reduction: float
    approximate_logdet_reduction: float
    negative_distance: float
    observation_age: float
    recent_mean_nis: float
    negative_recent_mean_nis: float
    nis_calibration_quality: float
    nis_sample_count: float
    negative_consecutive_anomaly_count: float


@dataclass(frozen=True)
class EdgeNisSafetyAssessment:
    edge: UndirectedEdge
    sample_count: int
    maximum_log_calibration_deviation: float | None
    maximum_consecutive_anomaly_count: int
    passes_safety_gate: bool


_MEASUREMENT_DIMENSION_BY_MODALITY = {
    "RANGE": 1,
    "RANGE_RATE": 1,
    "RADAR": 2,
    "AZ_EL": 2,
    "INFRARED": 2,
    "OPTICAL": 2,
}


def assess_edge_nis_safety(
    edge,
    *,
    minimum_sample_count: int = 2,
    maximum_log_calibration_deviation: float = np.log(3.0),
    maximum_consecutive_anomaly_count: int = 0,
) -> EdgeNisSafetyAssessment:
    """Classify an edge without treating missing NIS history as evidence."""

    if minimum_sample_count <= 0:
        raise ValueError("minimum_sample_count must be positive.")
    if maximum_log_calibration_deviation < 0.0:
        raise ValueError("maximum_log_calibration_deviation cannot be negative.")
    if maximum_consecutive_anomaly_count < 0:
        raise ValueError("maximum_consecutive_anomaly_count cannot be negative.")
    count_by_modality = dict(edge.nis_sample_count_by_modality)
    anomaly_by_modality = dict(edge.consecutive_anomaly_count_by_modality)
    deviations = [
        abs(np.log(max(float(nis), 1e-12) / _measurement_dimension(modality)))
        for modality, nis in edge.nis_by_modality
    ]
    sample_count = int(sum(count_by_modality.values()))
    maximum_deviation = max(deviations) if deviations else None
    maximum_anomaly = max(anomaly_by_modality.values(), default=0)
    return EdgeNisSafetyAssessment(
        edge=edge.nodes,
        sample_count=sample_count,
        maximum_log_calibration_deviation=maximum_deviation,
        maximum_consecutive_anomaly_count=maximum_anomaly,
        passes_safety_gate=bool(
            sample_count >= minimum_sample_count
            and maximum_deviation is not None
            and maximum_deviation <= maximum_log_calibration_deviation
            and maximum_anomaly <= maximum_consecutive_anomaly_count
        ),
    )


def _measurement_dimension(modality: str) -> int:
    normalized = str(modality).upper()
    try:
        return _MEASUREMENT_DIMENSION_BY_MODALITY[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported NIS modality: {modality}") from exc


def range_edge_scores(
    observation: GraphObservation,
    *,
    range_sigma: float = 2.0,
) -> tuple[DecisionTimeEdgeScore, ...]:
    """Approximate symmetric RANGE information using decision-time features."""

    if range_sigma <= 0.0:
        raise ValueError("range_sigma must be positive.")
    node_by_id = {node.node_id: node for node in observation.nodes}
    values = []
    for edge in observation.candidate_edges:
        left = node_by_id[edge.nodes[0]]
        right = node_by_id[edge.nodes[1]]
        if left.covariance_diagonal is None or right.covariance_diagonal is None:
            raise ValueError("RANGE edge scoring requires node covariance diagonals.")
        if len(left.state) < 3 or len(right.state) < 3:
            raise ValueError("RANGE edge scoring requires three-position states.")
        if len(left.covariance_diagonal) < 3 or len(right.covariance_diagonal) < 3:
            raise ValueError("RANGE edge scoring requires position covariance.")
        relative = (
            np.asarray(right.state[:3], dtype=float)
            - np.asarray(left.state[:3], dtype=float)
        )
        distance = float(np.linalg.norm(relative))
        if distance <= 0.0:
            raise ValueError("RANGE edge scoring requires nonzero separation.")
        direction = relative / distance
        left_variance = np.asarray(
            left.covariance_diagonal[:3], dtype=float
        )
        right_variance = np.asarray(
            right.covariance_diagonal[:3], dtype=float
        )
        if (
            np.any(left_variance < 0.0)
            or np.any(right_variance < 0.0)
        ):
            raise ValueError("Covariance diagonal cannot be negative.")
        left_projected = float(np.dot(direction ** 2, left_variance))
        right_projected = float(np.dot(direction ** 2, right_variance))
        innovation_variance = (
            left_projected + right_projected + range_sigma ** 2
        )
        left_trace_reduction = float(np.sum(
            (left_variance * direction) ** 2
        ) / innovation_variance)
        right_trace_reduction = float(np.sum(
            (right_variance * direction) ** 2
        ) / innovation_variance)
        left_logdet_reduction = float(np.log1p(
            left_projected / (right_projected + range_sigma ** 2)
        ))
        right_logdet_reduction = float(np.log1p(
            right_projected / (left_projected + range_sigma ** 2)
        ))
        sample_count_by_modality = dict(edge.nis_sample_count_by_modality)
        calibration_values = [
            -abs(np.log(
                max(float(nis), 1e-12) / _measurement_dimension(modality)
            ))
            for modality, nis in edge.nis_by_modality
        ]
        calibration_weights = [
            max(1, sample_count_by_modality.get(modality, 0))
            for modality, _ in edge.nis_by_modality
        ]
        values.append(DecisionTimeEdgeScore(
            edge=edge.nodes,
            endpoint_position_uncertainty=float(
                np.sum(left_variance) + np.sum(right_variance)
            ),
            projected_position_uncertainty=(
                left_projected + right_projected
            ),
            approximate_trace_reduction=(
                left_trace_reduction + right_trace_reduction
            ),
            approximate_logdet_reduction=(
                left_logdet_reduction + right_logdet_reduction
            ),
            negative_distance=-float(edge.distance),
            observation_age=float(
                0.0 if edge.observation_age is None
                else edge.observation_age
            ),
            recent_mean_nis=float(
                np.mean([value for _, value in edge.nis_by_modality])
                if edge.nis_by_modality else 0.0
            ),
            negative_recent_mean_nis=float(
                -np.mean([value for _, value in edge.nis_by_modality])
                if edge.nis_by_modality else 0.0
            ),
            nis_calibration_quality=float(
                np.average(calibration_values, weights=calibration_weights)
                if calibration_values else 0.0
            ),
            nis_sample_count=float(sum(sample_count_by_modality.values())),
            negative_consecutive_anomaly_count=float(-sum(
                value
                for _, value in edge.consecutive_anomaly_count_by_modality
            )),
        ))
    return tuple(values)


def range_topology_score(
    observation: GraphObservation,
    action: TopologyAction,
    *,
    range_sigma: float = 2.0,
) -> DecisionTimeTopologyScore:
    """Sum decision-time edge scores for one legal topology action."""

    score_by_edge = {
        value.edge: value
        for value in range_edge_scores(observation, range_sigma=range_sigma)
    }
    if set(action.active_edges) - set(score_by_edge):
        raise ValueError("Topology action contains a non-candidate edge.")

    def total(name: str) -> float:
        return float(sum(
            getattr(score_by_edge[edge], name) for edge in action.active_edges
        ))

    return DecisionTimeTopologyScore(
        endpoint_position_uncertainty=total(
            "endpoint_position_uncertainty"
        ),
        projected_position_uncertainty=total(
            "projected_position_uncertainty"
        ),
        approximate_trace_reduction=total(
            "approximate_trace_reduction"
        ),
        approximate_logdet_reduction=total(
            "approximate_logdet_reduction"
        ),
        negative_distance=total("negative_distance"),
        observation_age=total("observation_age"),
        recent_mean_nis=total("recent_mean_nis"),
        negative_recent_mean_nis=total("negative_recent_mean_nis"),
        nis_calibration_quality=total("nis_calibration_quality"),
        nis_sample_count=total("nis_sample_count"),
        negative_consecutive_anomaly_count=total(
            "negative_consecutive_anomaly_count"
        ),
    )
