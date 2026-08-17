from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.topology_control_baselines import EnvironmentPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from cooperative.v15_policy_tensor import V15PolicyTensor


@dataclass(frozen=True)
class SnapshotActionValueRecord:
    seed: int
    decision_epoch: int
    action_id: int
    action_kind: str
    added_edges: str
    removed_edges: str
    lookahead_steps: int
    final_position_rmse: float
    position_rmse_reduction_vs_keep: float
    transmitted_messages: float
    replay_count: float
    resynchronization_count: float
    topology_switch_count: float


@dataclass(frozen=True)
class SnapshotActionTensorGroup:
    seed: int
    decision_epoch: int
    policy_tensor: V15PolicyTensor
    action_kinds: tuple[str, ...]
    action_features: np.ndarray
    active_edge_mask: np.ndarray
    added_edge_mask: np.ndarray
    removed_edge_mask: np.ndarray
    target_names: tuple[str, ...]
    targets: np.ndarray


@dataclass(frozen=True)
class SnapshotActionTensorDataset:
    feature_version: str
    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    action_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    groups: tuple[SnapshotActionTensorGroup, ...]


@dataclass(frozen=True)
class SnapshotActionTensorSplit:
    training: SnapshotActionTensorDataset
    validation: SnapshotActionTensorDataset


SNAPSHOT_TARGET_NAMES = (
    "position_rmse_reduction_vs_keep", "transmitted_messages",
    "replay_count", "resynchronization_count", "topology_switch_count",
)
SNAPSHOT_ACTION_FEATURE_NAMES = (
    "action_keep", "action_add", "action_swap", "action_remove",
    "active_edge_count", "added_edge_count", "removed_edge_count",
)


def evaluate_topology_action_snapshot(
    environment: TopologyControlEnvironment, *, seed: int,
    decision_epoch: int, baseline_policy: EnvironmentPolicy,
    lookahead_steps: int = 2, condition_seed: int | None = None,
) -> tuple[SnapshotActionValueRecord, ...]:
    """Label every legal action at one causal online-environment snapshot."""

    if decision_epoch < 0 or lookahead_steps < 1:
        raise ValueError("Decision epoch and lookahead must be nonnegative/positive.")
    state = environment.reset(seed=seed, condition_seed=condition_seed)
    for _ in range(decision_epoch):
        result = environment.step(baseline_policy.select_action(state))
        if result.terminated or result.truncated:
            raise ValueError("Decision epoch lies beyond the episode horizon.")
        state = result.state
    return _evaluate_current_snapshot(
        environment, state, seed=seed, decision_epoch=decision_epoch,
        lookahead_steps=lookahead_steps,
    )


def _evaluate_current_snapshot(
    environment, state, *, seed, decision_epoch, lookahead_steps,
):
    outcomes = []
    for action, allowed in zip(
        state.action_space.actions, state.action_space.legal_mask
    ):
        if not allowed:
            continue
        branch = deepcopy(environment)
        result = branch.step(action.action_id)
        transmitted = result.constraint_costs.transmitted_messages
        replay = result.constraint_costs.replay_count
        resynchronization = result.constraint_costs.resynchronization_count
        switches = result.constraint_costs.topology_switch
        for _ in range(1, lookahead_steps):
            if result.terminated or result.truncated:
                break
            result = branch.step(0)
            transmitted += result.constraint_costs.transmitted_messages
            replay += result.constraint_costs.replay_count
            resynchronization += result.constraint_costs.resynchronization_count
            switches += result.constraint_costs.topology_switch
        outcomes.append((action, dict(result.diagnostics)["position_rmse"],
                         transmitted, replay, resynchronization, switches))
    keep_rmse = next(value[1] for value in outcomes if value[0].kind == "keep")
    return tuple(SnapshotActionValueRecord(
        seed=int(seed), decision_epoch=int(decision_epoch),
        action_id=action.action_id, action_kind=action.kind,
        added_edges=_edges_text(action.added_edges),
        removed_edges=_edges_text(action.removed_edges),
        lookahead_steps=int(lookahead_steps), final_position_rmse=float(rmse),
        position_rmse_reduction_vs_keep=float(keep_rmse - rmse),
        transmitted_messages=float(transmitted), replay_count=float(replay),
        resynchronization_count=float(resynchronization),
        topology_switch_count=float(switches),
    ) for action, rmse, transmitted, replay, resynchronization, switches in outcomes)


def build_topology_action_snapshot_tensor(
    environment: TopologyControlEnvironment, *, seed: int,
    decision_epoch: int, baseline_policy: EnvironmentPolicy,
    lookahead_steps: int = 2,
) -> tuple[SnapshotActionTensorGroup, tuple[SnapshotActionValueRecord, ...]]:
    """Build one no-truth input group aligned with causal future labels."""

    state = environment.reset(seed=seed)
    for _ in range(decision_epoch):
        result = environment.step(baseline_policy.select_action(state))
        if result.terminated or result.truncated:
            raise ValueError("Decision epoch lies beyond the episode horizon.")
        state = result.state
    return _build_current_snapshot_tensor(
        environment, state, seed=seed, decision_epoch=decision_epoch,
        lookahead_steps=lookahead_steps,
    )


def build_online_snapshot_action_tensor(state, *, seed=-1, decision_epoch=-1):
    """Build deployable action inputs from current state without future labels."""

    legal_actions = tuple(
        action for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        ) if allowed
    )
    edge_index = {
        edge: index for index, edge in enumerate(
            state.policy_tensor.candidate_edges
        )
    }
    active_masks, added_masks, removed_masks, action_features = [], [], [], []
    for action in legal_actions:
        active = set(action.topology.active_edges)
        added, removed = set(action.added_edges), set(action.removed_edges)
        active_masks.append(_edge_mask(edge_index, active))
        added_masks.append(_edge_mask(edge_index, added))
        removed_masks.append(_edge_mask(edge_index, removed))
        action_features.append((
            float(action.kind == "keep"), float(action.kind == "add"),
            float(action.kind == "swap"), float(action.kind == "remove"),
            float(len(active)), float(len(added)), float(len(removed)),
        ))
    arrays = tuple(map(_readonly, (
        action_features, active_masks, added_masks, removed_masks,
    )))
    return SnapshotActionTensorGroup(
        seed=int(seed), decision_epoch=int(decision_epoch),
        policy_tensor=state.policy_tensor,
        action_kinds=tuple(action.kind for action in legal_actions),
        action_features=arrays[0], active_edge_mask=arrays[1],
        added_edge_mask=arrays[2], removed_edge_mask=arrays[3],
        target_names=SNAPSHOT_TARGET_NAMES,
        targets=_readonly(np.zeros((len(legal_actions), len(SNAPSHOT_TARGET_NAMES)))),
    ), tuple(action.action_id for action in legal_actions)


def _build_current_snapshot_tensor(
    environment, state, *, seed, decision_epoch, lookahead_steps,
):
    records = _evaluate_current_snapshot(
        environment, state, seed=seed, decision_epoch=decision_epoch,
        lookahead_steps=lookahead_steps,
    )
    legal_actions = tuple(
        action for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        ) if allowed
    )
    if tuple(record.action_id for record in records) != tuple(
        action.action_id for action in legal_actions
    ):
        raise RuntimeError("Snapshot action labels do not align with actions.")
    edge_index = {
        edge: index for index, edge in enumerate(
            state.policy_tensor.candidate_edges
        )
    }
    active_masks, added_masks, removed_masks, action_features = [], [], [], []
    for action in legal_actions:
        active = set(action.topology.active_edges)
        added, removed = set(action.added_edges), set(action.removed_edges)
        active_masks.append(_edge_mask(edge_index, active))
        added_masks.append(_edge_mask(edge_index, added))
        removed_masks.append(_edge_mask(edge_index, removed))
        action_features.append((
            float(action.kind == "keep"), float(action.kind == "add"),
            float(action.kind == "swap"), float(action.kind == "remove"),
            float(len(active)), float(len(added)), float(len(removed)),
        ))
    targets = np.asarray([
        tuple(float(getattr(record, name)) for name in SNAPSHOT_TARGET_NAMES)
        for record in records
    ])
    arrays = tuple(map(_readonly, (
        action_features, active_masks, added_masks, removed_masks, targets,
    )))
    return SnapshotActionTensorGroup(
        seed=int(seed), decision_epoch=int(decision_epoch),
        policy_tensor=state.policy_tensor,
        action_kinds=tuple(action.kind for action in legal_actions),
        action_features=arrays[0], active_edge_mask=arrays[1],
        added_edge_mask=arrays[2], removed_edge_mask=arrays[3],
        target_names=SNAPSHOT_TARGET_NAMES, targets=arrays[4],
    ), records


def build_topology_snapshot_tensor_dataset(
    environment: TopologyControlEnvironment, *, seeds: Iterable[int],
    decision_epochs: Iterable[int], baseline_policy: EnvironmentPolicy,
    lookahead_steps: int = 2,
) -> SnapshotActionTensorDataset:
    """Generate aligned online snapshots without mixing future labels into inputs."""

    seed_values = tuple(int(value) for value in seeds)
    epoch_values = tuple(int(value) for value in decision_epochs)
    if not seed_values or not epoch_values:
        raise ValueError("Snapshot dataset seeds and decision epochs must be nonempty.")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Snapshot dataset seeds must be unique.")
    if len(set(epoch_values)) != len(epoch_values):
        raise ValueError("Snapshot decision epochs must be unique.")
    if any(epoch < 0 for epoch in epoch_values):
        raise ValueError("Snapshot decision epochs must be nonnegative.")
    ordered_epochs = tuple(sorted(epoch_values))
    groups = []
    for seed in seed_values:
        branch = deepcopy(environment)
        state = branch.reset(seed=seed)
        current_epoch = 0
        for epoch in ordered_epochs:
            while current_epoch < epoch:
                result = branch.step(baseline_policy.select_action(state))
                if result.terminated or result.truncated:
                    raise ValueError("Decision epoch lies beyond the episode horizon.")
                state = result.state
                current_epoch += 1
            groups.append(_build_current_snapshot_tensor(
                branch, state, seed=seed, decision_epoch=epoch,
                lookahead_steps=lookahead_steps,
            )[0])
    groups = tuple(groups)
    reference = groups[0].policy_tensor
    for group in groups[1:]:
        tensor = group.policy_tensor
        if (
            tensor.node_feature_names != reference.node_feature_names
            or tensor.edge_feature_names != reference.edge_feature_names
            or tensor.global_feature_names != reference.global_feature_names
            or group.target_names != groups[0].target_names
        ):
            raise ValueError("V15 snapshot tensor schemas differ across groups.")
    return SnapshotActionTensorDataset(
        feature_version="v15.0-online-snapshot-action-value",
        node_feature_names=reference.node_feature_names,
        edge_feature_names=reference.edge_feature_names,
        global_feature_names=reference.global_feature_names,
        action_feature_names=SNAPSHOT_ACTION_FEATURE_NAMES,
        target_names=SNAPSHOT_TARGET_NAMES,
        groups=groups,
    )


def split_topology_snapshot_dataset_by_seed(
    dataset: SnapshotActionTensorDataset, *, training_seeds: Iterable[int],
    validation_seeds: Iterable[int],
) -> SnapshotActionTensorSplit:
    training, validation = set(training_seeds), set(validation_seeds)
    if not training or not validation or training & validation:
        raise ValueError("Training and validation seeds must be nonempty and disjoint.")
    available = {group.seed for group in dataset.groups}
    if (training | validation) - available:
        raise ValueError("Requested seeds are absent from the snapshot dataset.")

    def subset(seeds):
        return SnapshotActionTensorDataset(
            feature_version=dataset.feature_version,
            node_feature_names=dataset.node_feature_names,
            edge_feature_names=dataset.edge_feature_names,
            global_feature_names=dataset.global_feature_names,
            action_feature_names=dataset.action_feature_names,
            target_names=dataset.target_names,
            groups=tuple(group for group in dataset.groups if group.seed in seeds),
        )

    return SnapshotActionTensorSplit(subset(training), subset(validation))


def merge_topology_snapshot_tensor_datasets(
    datasets: Iterable[SnapshotActionTensorDataset],
) -> SnapshotActionTensorDataset:
    """Merge restartable shards while enforcing one common feature schema."""

    values = tuple(datasets)
    if not values:
        raise ValueError("At least one snapshot dataset is required for merging.")
    reference = values[0]
    schema = (
        reference.feature_version, reference.node_feature_names,
        reference.edge_feature_names, reference.global_feature_names,
        reference.action_feature_names, reference.target_names,
    )
    groups, identities = [], set()
    for dataset in values:
        if schema != (
            dataset.feature_version, dataset.node_feature_names,
            dataset.edge_feature_names, dataset.global_feature_names,
            dataset.action_feature_names, dataset.target_names,
        ):
            raise ValueError("Snapshot dataset shard schemas differ.")
        for group in dataset.groups:
            identity = (group.seed, group.decision_epoch)
            if identity in identities:
                raise ValueError(f"Duplicate snapshot group {identity}.")
            identities.add(identity)
            groups.append(group)
    return SnapshotActionTensorDataset(
        feature_version=reference.feature_version,
        node_feature_names=reference.node_feature_names,
        edge_feature_names=reference.edge_feature_names,
        global_feature_names=reference.global_feature_names,
        action_feature_names=reference.action_feature_names,
        target_names=reference.target_names,
        groups=tuple(sorted(groups, key=lambda group: (
            group.seed, group.decision_epoch
        ))),
    )


def save_topology_snapshot_tensor_dataset(
    dataset: SnapshotActionTensorDataset, output_path: str | Path,
) -> Path:
    """Persist variable-size snapshot groups in one compressed, pickle-free NPZ."""

    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Snapshot tensor dataset path must use the .npz suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_version": dataset.feature_version,
        "node_feature_names": dataset.node_feature_names,
        "edge_feature_names": dataset.edge_feature_names,
        "global_feature_names": dataset.global_feature_names,
        "action_feature_names": dataset.action_feature_names,
        "target_names": dataset.target_names,
        "group_count": len(dataset.groups),
    }
    arrays = {"manifest": np.asarray(json.dumps(manifest))}
    for index, group in enumerate(dataset.groups):
        prefix = f"group_{index:05d}_"
        tensor = group.policy_tensor
        arrays.update({
            prefix + "identity": np.asarray((group.seed, group.decision_epoch)),
            prefix + "node_ids": np.asarray(tensor.node_ids),
            prefix + "node_features": tensor.node_features,
            prefix + "candidate_edges": np.asarray(tensor.candidate_edges),
            prefix + "edge_index": tensor.edge_index,
            prefix + "edge_features": tensor.edge_features,
            prefix + "global_features": tensor.global_features,
            prefix + "action_kinds": np.asarray(group.action_kinds),
            prefix + "action_features": group.action_features,
            prefix + "active_edge_mask": group.active_edge_mask,
            prefix + "added_edge_mask": group.added_edge_mask,
            prefix + "removed_edge_mask": group.removed_edge_mask,
            prefix + "targets": group.targets,
        })
    np.savez_compressed(path, **arrays)
    return path


def load_topology_snapshot_tensor_dataset(
    input_path: str | Path,
) -> SnapshotActionTensorDataset:
    """Load and validate a pickle-free V15 snapshot tensor dataset."""

    path = Path(input_path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        groups = []
        for index in range(int(manifest["group_count"])):
            prefix = f"group_{index:05d}_"
            identity = archive[prefix + "identity"]
            tensor = V15PolicyTensor(
                schema_version="v15.0-policy-normalized",
                node_ids=tuple(str(value) for value in archive[prefix + "node_ids"]),
                node_feature_names=tuple(manifest["node_feature_names"]),
                node_features=_readonly(archive[prefix + "node_features"]),
                candidate_edges=tuple(
                    tuple(str(value) for value in row)
                    for row in archive[prefix + "candidate_edges"]
                ),
                edge_index=_readonly_int(archive[prefix + "edge_index"]),
                edge_feature_names=tuple(manifest["edge_feature_names"]),
                edge_features=_readonly(archive[prefix + "edge_features"]),
                global_feature_names=tuple(manifest["global_feature_names"]),
                global_features=_readonly(archive[prefix + "global_features"]),
            )
            groups.append(SnapshotActionTensorGroup(
                seed=int(identity[0]), decision_epoch=int(identity[1]),
                policy_tensor=tensor,
                action_kinds=tuple(
                    str(value) for value in archive[prefix + "action_kinds"]
                ),
                action_features=_readonly(archive[prefix + "action_features"]),
                active_edge_mask=_readonly(archive[prefix + "active_edge_mask"]),
                added_edge_mask=_readonly(archive[prefix + "added_edge_mask"]),
                removed_edge_mask=_readonly(archive[prefix + "removed_edge_mask"]),
                target_names=tuple(manifest["target_names"]),
                targets=_readonly(archive[prefix + "targets"]),
            ))
    dataset = SnapshotActionTensorDataset(
        feature_version=str(manifest["feature_version"]),
        node_feature_names=tuple(manifest["node_feature_names"]),
        edge_feature_names=tuple(manifest["edge_feature_names"]),
        global_feature_names=tuple(manifest["global_feature_names"]),
        action_feature_names=tuple(manifest["action_feature_names"]),
        target_names=tuple(manifest["target_names"]), groups=tuple(groups),
    )
    _validate_snapshot_tensor_dataset(dataset)
    return dataset


def _validate_snapshot_tensor_dataset(dataset):
    for group in dataset.groups:
        tensor = group.policy_tensor
        action_count = len(group.action_kinds)
        edge_count = len(tensor.candidate_edges)
        if tensor.node_feature_names != dataset.node_feature_names or (
            tensor.edge_feature_names != dataset.edge_feature_names
        ) or tensor.global_feature_names != dataset.global_feature_names:
            raise ValueError("Stored snapshot feature schema is inconsistent.")
        if group.target_names != dataset.target_names or (
            group.action_features.shape != (
                action_count, len(dataset.action_feature_names)
            )
        ) or group.targets.shape != (action_count, len(dataset.target_names)):
            raise ValueError("Stored snapshot action/target schema is inconsistent.")
        if any(mask.shape != (action_count, edge_count) for mask in (
            group.active_edge_mask, group.added_edge_mask,
            group.removed_edge_mask,
        )):
            raise ValueError("Stored snapshot edge masks are inconsistent.")


def export_snapshot_action_values(
    records: Iterable[SnapshotActionValueRecord], output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(records)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(SnapshotActionValueRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(value) for value in values)
    return path


def _edges_text(edges):
    return ";".join(f"{left}|{right}" for left, right in edges)


def _edge_mask(edge_index, selected):
    if set(selected) - set(edge_index):
        raise ValueError("Snapshot action references a non-candidate edge.")
    return tuple(float(edge in selected) for edge in edge_index)


def _readonly(values):
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _readonly_int(values):
    array = np.asarray(values, dtype=np.int64)
    array.setflags(write=False)
    return array
