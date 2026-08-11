from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import GraphObservation, TopologyAction
from orbital_core.inter_satellite_model import RelativeMeasurementModel


@dataclass(frozen=True)
class ActionGraphMetrics:
    algebraic_connectivity: float
    minimum_degree: float
    maximum_degree: float
    negative_degree_variance: float
    negative_bridge_count: float
    normalized_information_rank: float
    minimum_positive_information_eigenvalue: float
    information_log_pseudodeterminant: float
    negative_log_information_condition: float


ACTION_GRAPH_FEATURE_NAMES = tuple(ActionGraphMetrics.__dataclass_fields__)


@dataclass(frozen=True)
class ActionPairMetrics:
    added_information_rank: float
    removed_information_rank: float
    added_removed_subspace_complementarity: float
    added_retained_nullspace_fill: float
    removed_retained_unique_information: float


ACTION_PAIR_FEATURE_NAMES = tuple(ActionPairMetrics.__dataclass_fields__)


def action_pair_metrics(
    observation: GraphObservation, action: TopologyAction,
) -> ActionPairMetrics:
    """Compare added and removed measurement subspaces against retained edges."""

    baseline = set(observation.previous_active_edges)
    active = set(action.active_edges)
    added, removed, retained = active - baseline, baseline - active, active & baseline
    added_basis = _row_space_basis(_normalized_jacobian_rows(observation, added))
    removed_basis = _row_space_basis(_normalized_jacobian_rows(observation, removed))
    retained_basis = _row_space_basis(_normalized_jacobian_rows(observation, retained))
    return ActionPairMetrics(
        added_information_rank=float(len(added_basis)),
        removed_information_rank=float(len(removed_basis)),
        added_removed_subspace_complementarity=_subspace_complementarity(
            added_basis, removed_basis
        ),
        added_retained_nullspace_fill=_subspace_complementarity(
            added_basis, retained_basis
        ),
        removed_retained_unique_information=_subspace_complementarity(
            removed_basis, retained_basis
        ),
    )


def action_graph_metrics(
    observation: GraphObservation, action: TopologyAction,
) -> ActionGraphMetrics:
    """Describe one legal action using topology and normalized Jacobian geometry."""

    node_ids = tuple(node.node_id for node in observation.nodes)
    node_index = {node: index for index, node in enumerate(node_ids)}
    candidate_by_edge = {
        edge.nodes: edge for edge in observation.candidate_edges
    }
    if set(action.active_edges) - set(candidate_by_edge):
        raise ValueError("Action contains a non-candidate edge.")
    adjacency = np.zeros((len(node_ids), len(node_ids)))
    degree = np.zeros(len(node_ids))
    rows = []
    state_by_node = {
        node.node_id: np.asarray(node.state, dtype=float)
        for node in observation.nodes
    }
    active_edges = set(action.active_edges)
    for left, right in action.active_edges:
        left_index, right_index = node_index[left], node_index[right]
        adjacency[left_index, right_index] = 1.0
        adjacency[right_index, left_index] = 1.0
        degree[left_index] += 1.0
        degree[right_index] += 1.0
    if observation.measurements:
        measurement_values = observation.measurements
    else:
        measurement_values = tuple(
            (left, right, modality, "ECI", None)
            for left, right in action.active_edges
            for modality in candidate_by_edge[(left, right)].measurement_modalities
            if str(modality).upper() != "OPTICAL"
        )
    for measurement in measurement_values:
        if hasattr(measurement, "observer_id"):
            observer, target = measurement.observer_id, measurement.target_id
            modality, frame = measurement.modality, measurement.frame
            quaternion = measurement.quaternion_i2b_wxyz
        else:
            observer, target, modality, frame, quaternion = measurement
        if tuple(sorted((observer, target))) not in active_edges:
            continue
        observer_jacobian, target_jacobian = RelativeMeasurementModel(
            modality, frame
        ).jacobians(
            state_by_node[observer], state_by_node[target],
            quaternion_i2b_wxyz=(
                None if quaternion is None else np.asarray(quaternion)
            ),
        )
        full = np.zeros((observer_jacobian.shape[0], 6 * len(node_ids)))
        observer_index, target_index = node_index[observer], node_index[target]
        full[:, 6 * observer_index:6 * (observer_index + 1)] = observer_jacobian
        full[:, 6 * target_index:6 * (target_index + 1)] = target_jacobian
        for row in full:
            norm = float(np.linalg.norm(row))
            if norm > 1e-12:
                rows.append(row / norm)
    laplacian = np.diag(degree) - adjacency
    laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
    algebraic = (
        float(laplacian_eigenvalues[1])
        if len(laplacian_eigenvalues) > 1 else 0.0
    )
    information = (
        np.asarray(rows).T @ np.asarray(rows)
        if rows else np.zeros((6 * len(node_ids), 6 * len(node_ids)))
    )
    eigenvalues = np.linalg.eigvalsh(information)
    positive = eigenvalues[eigenvalues > 1e-9]
    condition = (
        float(positive[-1] / positive[0]) if positive.size else 1.0
    )
    return ActionGraphMetrics(
        algebraic_connectivity=algebraic,
        minimum_degree=float(np.min(degree)),
        maximum_degree=float(np.max(degree)),
        negative_degree_variance=-float(np.var(degree)),
        negative_bridge_count=-float(_bridge_count(adjacency)),
        normalized_information_rank=float(positive.size),
        minimum_positive_information_eigenvalue=(
            float(positive[0]) if positive.size else 0.0
        ),
        information_log_pseudodeterminant=(
            float(np.sum(np.log(positive))) if positive.size else 0.0
        ),
        negative_log_information_condition=-float(np.log(condition)),
    )


def _bridge_count(adjacency: np.ndarray) -> int:
    count = 0
    for left in range(len(adjacency)):
        for right in range(left + 1, len(adjacency)):
            if adjacency[left, right] == 0.0:
                continue
            reduced = adjacency.copy()
            reduced[left, right] = reduced[right, left] = 0.0
            if not _connected(reduced):
                count += 1
    return count


def _normalized_jacobian_rows(observation, selected_edges) -> np.ndarray:
    node_ids = tuple(node.node_id for node in observation.nodes)
    node_index = {node: index for index, node in enumerate(node_ids)}
    state_by_node = {
        node.node_id: np.asarray(node.state, dtype=float)
        for node in observation.nodes
    }
    candidate_by_edge = {edge.nodes: edge for edge in observation.candidate_edges}
    selected = set(selected_edges)
    if not selected:
        return np.empty((0, 6 * len(node_ids)))
    if observation.measurements:
        measurement_values = observation.measurements
    else:
        measurement_values = tuple(
            (left, right, modality, "ECI", None)
            for left, right in selected
            for modality in candidate_by_edge[(left, right)].measurement_modalities
            if str(modality).upper() != "OPTICAL"
        )
    rows = []
    for measurement in measurement_values:
        if hasattr(measurement, "observer_id"):
            observer, target = measurement.observer_id, measurement.target_id
            modality, frame = measurement.modality, measurement.frame
            quaternion = measurement.quaternion_i2b_wxyz
        else:
            observer, target, modality, frame, quaternion = measurement
        if tuple(sorted((observer, target))) not in selected:
            continue
        observer_h, target_h = RelativeMeasurementModel(modality, frame).jacobians(
            state_by_node[observer], state_by_node[target],
            quaternion_i2b_wxyz=(
                None if quaternion is None else np.asarray(quaternion)
            ),
        )
        full = np.zeros((observer_h.shape[0], 6 * len(node_ids)))
        observer_index, target_index = node_index[observer], node_index[target]
        full[:, 6 * observer_index:6 * (observer_index + 1)] = observer_h
        full[:, 6 * target_index:6 * (target_index + 1)] = target_h
        for row in full:
            norm = float(np.linalg.norm(row))
            if norm > 1e-12:
                rows.append(row / norm)
    return np.asarray(rows) if rows else np.empty((0, 6 * len(node_ids)))


def _row_space_basis(rows: np.ndarray) -> np.ndarray:
    if not len(rows):
        return np.empty((0, rows.shape[1]))
    _, singular_values, right = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-9))
    return right[:rank]


def _subspace_complementarity(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left):
        return 0.0
    overlap = (
        float(np.sum((left @ right.T) ** 2) / len(left))
        if len(right) else 0.0
    )
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def _connected(adjacency: np.ndarray) -> bool:
    visited, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(np.flatnonzero(adjacency[node]).tolist())
    return len(visited) == len(adjacency)
