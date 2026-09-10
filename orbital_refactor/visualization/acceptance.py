"""Acceptance comparisons for visualization recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visualization.recording import VisualizationRecordingReader


@dataclass(frozen=True)
class VisualizationPairComparison:
    frame_count: int
    maximum_state_difference: float
    maximum_covariance_difference: float
    maximum_observation_difference: float
    maximum_raw_sensor_difference: float
    algorithm_payloads_identical: bool


def compare_visualization_algorithm_payloads(
    baseline_path, candidate_path,
) -> VisualizationPairComparison:
    """Compare algorithm data while allowing extra sidecar display fields."""
    baseline = VisualizationRecordingReader(baseline_path)
    candidate = VisualizationRecordingReader(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError("Paired visualization recordings have different lengths.")
    maxima = np.zeros(4)
    identical = True
    for left, right in zip(baseline, candidate):
        if (
            left.timestamp != right.timestamp
            or tuple(node.node_id for node in left.nodes)
            != tuple(node.node_id for node in right.nodes)
        ):
            raise ValueError("Paired visualization frames are not aligned.")
        for left_node, right_node in zip(left.nodes, right.nodes):
            maxima[0] = max(maxima[0], _difference(
                left_node.estimate_state, right_node.estimate_state,
            ))
            if left_node.covariance is None or right_node.covariance is None:
                raise ValueError("Full covariance is required for pair acceptance.")
            maxima[1] = max(maxima[1], _difference(
                left_node.covariance, right_node.covariance,
            ))
        left_observations = _observations_by_id(left)
        right_observations = _observations_by_id(right)
        if set(left_observations) != set(right_observations):
            raise ValueError("Paired frames contain different observations.")
        for information_id, left_item in left_observations.items():
            right_item = right_observations[information_id]
            maxima[2] = max(maxima[2], _difference(
                left_item.measurement, right_item.measurement,
            ))
            if (left_item.raw_sensor_data is None) != (
                right_item.raw_sensor_data is None
            ):
                identical = False
            elif left_item.raw_sensor_data is not None:
                maxima[3] = max(maxima[3], _difference(
                    left_item.raw_sensor_data, right_item.raw_sensor_data,
                ))
            identical = identical and (
                left_item.valid == right_item.valid
                and left_item.processing_status == right_item.processing_status
                and left_item.nis == right_item.nis
            )
    identical = bool(identical and np.all(maxima == 0.0))
    return VisualizationPairComparison(
        frame_count=len(baseline),
        maximum_state_difference=float(maxima[0]),
        maximum_covariance_difference=float(maxima[1]),
        maximum_observation_difference=float(maxima[2]),
        maximum_raw_sensor_difference=float(maxima[3]),
        algorithm_payloads_identical=identical,
    )


def _observations_by_id(frame):
    result = {}
    for observation in frame.observations:
        information_id = observation.metadata.get("information_id")
        if not information_id or information_id in result:
            raise ValueError("Visualization observations require unique IDs.")
        result[information_id] = observation
    return result


def _difference(left, right):
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))
