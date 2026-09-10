from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cooperative.v15_cann_policy_tensor import (
    CANN_AVAILABILITY_FEATURE_NAMES,
    CANN_NODE_FEATURE_NAMES,
)
from experiments.topology_snapshot_counterfactual import (
    SnapshotActionTensorDataset,
)


@dataclass(frozen=True)
class CANNFeatureInformation:
    feature_name: str
    observed_range: float
    oracle_gain_correlation: float | None
    action_spread_correlation: float | None


@dataclass(frozen=True)
class CANNSnapshotInformationAudit:
    group_count: int
    node_count_range: tuple[int, int]
    varying_feature_count: int
    oracle_non_keep_rate: float
    mean_oracle_gain: float
    features: tuple[CANNFeatureInformation, ...]


def audit_cann_snapshot_information(
    dataset: SnapshotActionTensorDataset,
) -> CANNSnapshotInformationAudit:
    """Audit whether v15.1 CANN inputs vary with action-value headroom.

    This is a screening diagnostic, not evidence that a feature causes an
    action gain.  CANN values are pooled per graph because they are node-level
    context shared by all legal actions at one decision snapshot.
    """
    if not dataset.groups:
        raise ValueError("CANN information audit requires a nonempty dataset.")
    indices = []
    for name in CANN_NODE_FEATURE_NAMES:
        if name not in dataset.node_feature_names:
            raise ValueError("Dataset does not contain v15.1 CANN features.")
        indices.append(dataset.node_feature_names.index(name))
    pooled, oracle_gains, spreads, non_keep = [], [], [], []
    for group in dataset.groups:
        values = group.policy_tensor.node_features[:, indices]
        pooled.append(np.mean(values, axis=0))
        gains = group.targets[:, group.target_names.index(
            "position_rmse_reduction_vs_keep"
        )]
        oracle_index = int(np.argmax(gains))
        oracle_gains.append(float(gains[oracle_index]))
        spreads.append(float(np.ptp(gains)))
        non_keep.append(group.action_kinds[oracle_index] != "keep")
    pooled = np.asarray(pooled)
    oracle_gains = np.asarray(oracle_gains)
    spreads = np.asarray(spreads)
    features = tuple(CANNFeatureInformation(
        feature_name=name,
        observed_range=float(np.ptp(pooled[:, index])),
        oracle_gain_correlation=_correlation(pooled[:, index], oracle_gains),
        action_spread_correlation=_correlation(pooled[:, index], spreads),
    ) for index, name in enumerate(CANN_NODE_FEATURE_NAMES))
    counts = [len(group.policy_tensor.node_ids) for group in dataset.groups]
    return CANNSnapshotInformationAudit(
        group_count=len(dataset.groups),
        node_count_range=(min(counts), max(counts)),
        varying_feature_count=sum(item.observed_range > 0.0 for item in features),
        oracle_non_keep_rate=float(np.mean(non_keep)),
        mean_oracle_gain=float(np.mean(oracle_gains)),
        features=features,
    )


def strip_cann_snapshot_features(
    dataset: SnapshotActionTensorDataset,
) -> SnapshotActionTensorDataset:
    """Create the exactly paired v15.0 prefix view without relabeling."""
    first_name = CANN_NODE_FEATURE_NAMES[0]
    if first_name not in dataset.node_feature_names:
        raise ValueError("Dataset does not contain v15.1 CANN features.")
    width = dataset.node_feature_names.index(first_name)
    groups = []
    for group in dataset.groups:
        features = group.policy_tensor.node_features[:, :width].copy()
        features.setflags(write=False)
        tensor = replace(
            group.policy_tensor,
            schema_version="v15.0-policy-normalized",
            node_feature_names=dataset.node_feature_names[:width],
            node_features=features,
        )
        groups.append(replace(group, policy_tensor=tensor))
    return replace(
        dataset,
        feature_version="v15.0-online-snapshot-action-value-paired",
        node_feature_names=dataset.node_feature_names[:width],
        groups=tuple(groups),
    )


def mask_cann_snapshot_features(
    dataset: SnapshotActionTensorDataset, *, retained_feature_names=(),
) -> SnapshotActionTensorDataset:
    """Keep the v15.1 shape while zeroing unselected CANN columns."""
    retained = set(retained_feature_names)
    cann_names = set(CANN_NODE_FEATURE_NAMES)
    availability_names = set(CANN_AVAILABILITY_FEATURE_NAMES)
    unknown = retained - cann_names - availability_names
    if unknown:
        raise ValueError(f"Unknown CANN feature names: {sorted(unknown)}")
    if CANN_NODE_FEATURE_NAMES[0] not in dataset.node_feature_names:
        raise ValueError("Dataset does not contain v15.1 CANN features.")
    zero_indices = [
        index for index, name in enumerate(dataset.node_feature_names)
        if (name in cann_names or name in availability_names)
        and name not in retained
    ]
    groups = []
    for group in dataset.groups:
        features = group.policy_tensor.node_features.copy()
        features[:, zero_indices] = 0.0
        features.setflags(write=False)
        groups.append(replace(
            group, policy_tensor=replace(
                group.policy_tensor, node_features=features,
            ),
        ))
    label = "zero" if not retained else "subset"
    return replace(
        dataset, feature_version=dataset.feature_version + f"-{label}",
        groups=tuple(groups),
    )


def shuffle_cann_snapshot_features(
    dataset: SnapshotActionTensorDataset, *, random_seed=0,
) -> SnapshotActionTensorDataset:
    """Destroy graph-label alignment while preserving CANN distributions."""
    if CANN_NODE_FEATURE_NAMES[0] not in dataset.node_feature_names:
        raise ValueError("Dataset does not contain v15.1 CANN features.")
    cann_names = set(CANN_NODE_FEATURE_NAMES) | set(
        CANN_AVAILABILITY_FEATURE_NAMES
    )
    indices = [index for index, name in enumerate(dataset.node_feature_names)
               if name in cann_names]
    grouped = {}
    for index, group in enumerate(dataset.groups):
        grouped.setdefault(len(group.policy_tensor.node_ids), []).append(index)
    source_by_target = {}
    rng = np.random.default_rng(int(random_seed))
    for group_indices in grouped.values():
        if len(group_indices) < 2:
            raise ValueError(
                "CANN shuffle needs at least two graphs per node-count stratum."
            )
        shuffled = np.asarray(group_indices)[rng.permutation(len(group_indices))]
        offset = int(rng.integers(1, len(group_indices)))
        sources = np.roll(shuffled, offset)
        source_by_target.update(zip(shuffled.tolist(), sources.tolist()))
    groups = []
    for target_index, group in enumerate(dataset.groups):
        source = dataset.groups[source_by_target[target_index]]
        features = group.policy_tensor.node_features.copy()
        features[:, indices] = source.policy_tensor.node_features[:, indices]
        features.setflags(write=False)
        groups.append(replace(
            group, policy_tensor=replace(
                group.policy_tensor, node_features=features,
            ),
        ))
    return replace(
        dataset,
        feature_version=dataset.feature_version + "-shuffled",
        groups=tuple(groups),
    )


def _correlation(left, right):
    if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
