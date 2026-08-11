from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import normalized_undirected_edge
from experiments.counterfactual_action_value import (
    ACTION_KINDS,
    CounterfactualActionValueDataset,
)


MODALITIES = ("RANGE", "RANGE_RATE", "RADAR", "AZ_EL", "INFRARED", "OPTICAL")
FRAMES = ("ECI", "BODY", "OTHER")
NODE_FEATURE_NAMES = tuple(
    [*(f"state_{index}" for index in range(6)),
     *(f"covariance_diagonal_{index}" for index in range(6))]
)
CANDIDATE_EDGE_FEATURE_NAMES = (
    "distance", "geometrically_visible", "communication_available",
    "delay", "packet_loss_rate", "observation_age_available",
    "observation_age", "mean_nis", "nis_sample_count",
    "maximum_consecutive_anomaly_count",
    *(f"modality_{value}" for value in MODALITIES),
)
MEASUREMENT_FEATURE_NAMES = (
    *(f"modality_{value}" for value in MODALITIES),
    *(f"frame_{value}" for value in FRAMES),
    "measurement_dimension",
    "covariance_00", "covariance_01", "covariance_10", "covariance_11",
    "quaternion_available", "quaternion_w", "quaternion_x",
    "quaternion_y", "quaternion_z",
)
ACTION_FEATURE_NAMES = (
    *(f"action_{kind}" for kind in ACTION_KINDS),
    "active_edge_count", "added_edge_count", "removed_edge_count",
)
TARGET_NAMES = (
    "position_rmse_reduction", "worst_node_position_rmse_reduction",
    "nees_calibration_improvement", "nees_coverage_calibration_improvement",
    "transmitted_message_cost", "replay_cost", "topology_change_cost",
)


@dataclass(frozen=True)
class GraphActionTensorGroup:
    group_id: tuple[int, int, int, int]
    node_ids: tuple[str, ...]
    node_features: np.ndarray
    candidate_edges: tuple[tuple[str, str], ...]
    candidate_edge_index: np.ndarray
    candidate_edge_features: np.ndarray
    measurement_edge_index: np.ndarray
    measurement_features: np.ndarray
    action_kinds: tuple[str, ...]
    action_features: np.ndarray
    active_edge_mask: np.ndarray
    added_edge_mask: np.ndarray
    removed_edge_mask: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class GraphActionTensorDataset:
    feature_version: str
    node_feature_names: tuple[str, ...]
    candidate_edge_feature_names: tuple[str, ...]
    measurement_feature_names: tuple[str, ...]
    action_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    groups: tuple[GraphActionTensorGroup, ...]


@dataclass(frozen=True)
class GraphActionTensorSplit:
    training: GraphActionTensorDataset
    validation: GraphActionTensorDataset


@dataclass(frozen=True)
class GraphActionTensorThreeWaySplit:
    training: GraphActionTensorDataset
    validation: GraphActionTensorDataset
    test: GraphActionTensorDataset


def split_graph_action_tensor_dataset_by_seed(
    dataset: GraphActionTensorDataset,
    *, training_seeds: tuple[int, ...], validation_seeds: tuple[int, ...],
) -> GraphActionTensorSplit:
    training, validation = set(training_seeds), set(validation_seeds)
    if not training or not validation or training & validation:
        raise ValueError("Training and validation seeds must be nonempty and disjoint.")
    available = {group.group_id[1] for group in dataset.groups}
    if (training | validation) - available:
        raise ValueError("Requested seeds are absent from the tensor dataset.")

    return GraphActionTensorSplit(
        _subset_by_seed(dataset, training),
        _subset_by_seed(dataset, validation),
    )


def split_graph_action_tensor_dataset_three_way(
    dataset: GraphActionTensorDataset,
    *,
    training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
) -> GraphActionTensorThreeWaySplit:
    """Create a strict seed-disjoint train/validation/test partition."""

    partitions = tuple(map(set, (
        training_seeds, validation_seeds, test_seeds,
    )))
    if any(not values for values in partitions):
        raise ValueError("Training, validation, and test seeds must be nonempty.")
    if any(
        partitions[left] & partitions[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Training, validation, and test seeds must be disjoint.")
    requested = set().union(*partitions)
    available = {group.group_id[1] for group in dataset.groups}
    if requested - available:
        raise ValueError("Requested seeds are absent from the tensor dataset.")
    return GraphActionTensorThreeWaySplit(*(
        _subset_by_seed(dataset, values) for values in partitions
    ))


def _subset_by_seed(dataset, seeds):
    return GraphActionTensorDataset(
        feature_version=dataset.feature_version,
        node_feature_names=dataset.node_feature_names,
        candidate_edge_feature_names=dataset.candidate_edge_feature_names,
        measurement_feature_names=dataset.measurement_feature_names,
        action_feature_names=dataset.action_feature_names,
        target_names=dataset.target_names,
        groups=tuple(
            group for group in dataset.groups if group.group_id[1] in seeds
        ),
    )


def build_graph_action_tensor_dataset(
    dataset: CounterfactualActionValueDataset,
) -> GraphActionTensorDataset:
    """Convert causal graph/action records into immutable NumPy tensors."""

    records_by_group = {}
    for record, group in zip(dataset.records, dataset.group_by_row):
        records_by_group.setdefault(group, []).append(record)
    groups = tuple(
        _build_group(group, observation, records_by_group[group])
        for group, observation in dataset.observation_by_group
    )
    return GraphActionTensorDataset(
        feature_version="v14.3-graph-action-causal",
        node_feature_names=NODE_FEATURE_NAMES,
        candidate_edge_feature_names=CANDIDATE_EDGE_FEATURE_NAMES,
        measurement_feature_names=MEASUREMENT_FEATURE_NAMES,
        action_feature_names=ACTION_FEATURE_NAMES,
        target_names=TARGET_NAMES,
        groups=groups,
    )


def _build_group(group, observation, records):
    node_ids = tuple(node.node_id for node in observation.nodes)
    node_index = {node: index for index, node in enumerate(node_ids)}
    node_features = []
    for node in observation.nodes:
        if len(node.state) != 6 or node.covariance_diagonal is None or (
            len(node.covariance_diagonal) != 6
        ):
            raise ValueError("Graph tensor nodes require six-state covariance data.")
        node_features.append((*node.state, *node.covariance_diagonal))
    candidate_edges = tuple(edge.nodes for edge in observation.candidate_edges)
    candidate_index = {edge: index for index, edge in enumerate(candidate_edges)}
    candidate_edge_features = []
    for edge in observation.candidate_edges:
        nis_values = [value for _, value in edge.nis_by_modality]
        candidate_edge_features.append((
            edge.distance, float(edge.geometrically_visible),
            float(edge.communication_available), edge.delay,
            edge.packet_loss_rate, float(edge.observation_age is not None),
            0.0 if edge.observation_age is None else edge.observation_age,
            float(np.mean(nis_values)) if nis_values else 0.0,
            float(sum(value for _, value in edge.nis_sample_count_by_modality)),
            float(max((value for _, value
                       in edge.consecutive_anomaly_count_by_modality), default=0)),
            *(_one_hot(edge.measurement_modalities, MODALITIES)),
        ))
    measurement_edge_index, measurement_features = [], []
    for measurement in observation.measurements:
        covariance = np.asarray(measurement.covariance, dtype=float)
        if covariance.shape not in {(1, 1), (2, 2)}:
            raise ValueError("Graph tensor supports one- or two-dimensional measurements.")
        padded = np.zeros((2, 2))
        padded[:covariance.shape[0], :covariance.shape[1]] = covariance
        quaternion = (
            np.zeros(4) if measurement.quaternion_i2b_wxyz is None
            else np.asarray(measurement.quaternion_i2b_wxyz, dtype=float)
        )
        frame = measurement.frame.upper()
        frame_label = frame if frame in {"ECI", "BODY"} else "OTHER"
        measurement_edge_index.append((
            node_index[measurement.observer_id], node_index[measurement.target_id]
        ))
        measurement_features.append((
            *_one_hot((measurement.modality,), MODALITIES),
            *_one_hot((frame_label,), FRAMES),
            float(covariance.shape[0]), *padded.reshape(-1),
            float(measurement.quaternion_i2b_wxyz is not None), *quaternion,
        ))
    action_features, active_masks, added_masks, removed_masks, targets = (
        [], [], [], [], []
    )
    for record in records:
        active = {normalized_undirected_edge(*edge) for edge in record.active_edges}
        added = {normalized_undirected_edge(*edge) for edge in record.added_edges}
        removed = {normalized_undirected_edge(*edge) for edge in record.removed_edges}
        action_features.append((
            *(1.0 if record.action_kind == kind else 0.0 for kind in ACTION_KINDS),
            float(len(active)), float(len(added)), float(len(removed)),
        ))
        active_masks.append([float(edge in active) for edge in candidate_edges])
        added_masks.append([float(edge in added) for edge in candidate_edges])
        removed_masks.append([float(edge in removed) for edge in candidate_edges])
        targets.append(tuple(float(getattr(record, name)) for name in TARGET_NAMES))
        if (added | removed) - set(candidate_index):
            raise ValueError("Action mask references a non-candidate edge.")
    return GraphActionTensorGroup(
        group_id=group, node_ids=node_ids,
        node_features=_readonly(node_features), candidate_edges=candidate_edges,
        candidate_edge_index=_readonly([
            (node_index[left], node_index[right]) for left, right in candidate_edges
        ], transpose=True, dtype=np.int64),
        candidate_edge_features=_readonly(candidate_edge_features),
        measurement_edge_index=_readonly(
            measurement_edge_index, transpose=True, dtype=np.int64,
            empty_rows=2,
        ),
        measurement_features=_readonly(
            measurement_features, empty_columns=len(MEASUREMENT_FEATURE_NAMES)
        ),
        action_kinds=tuple(record.action_kind for record in records),
        action_features=_readonly(action_features),
        active_edge_mask=_readonly(active_masks),
        added_edge_mask=_readonly(added_masks),
        removed_edge_mask=_readonly(removed_masks),
        targets=_readonly(targets),
    )


def _one_hot(values, vocabulary):
    normalized = {str(value).upper() for value in values}
    return tuple(float(value in normalized) for value in vocabulary)


def _readonly(values, *, transpose=False, dtype=float, empty_rows=None,
              empty_columns=None):
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        if empty_rows is not None:
            array = np.empty((empty_rows, 0), dtype=dtype)
        elif empty_columns is not None:
            array = np.empty((0, empty_columns), dtype=dtype)
    elif transpose:
        array = array.T
    array.setflags(write=False)
    return array
