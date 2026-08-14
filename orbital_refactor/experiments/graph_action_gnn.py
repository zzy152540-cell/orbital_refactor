from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from experiments.graph_action_tensor_dataset import (
    GraphActionTensorDataset,
    GraphActionTensorGroup,
)
from experiments.topology_snapshot_counterfactual import (
    SnapshotActionTensorDataset,
    SnapshotActionTensorGroup,
)


@dataclass(frozen=True)
class TorchGraphActionGroup:
    node_features: torch.Tensor
    candidate_edge_index: torch.Tensor
    candidate_edge_features: torch.Tensor
    measurement_edge_index: torch.Tensor
    measurement_features: torch.Tensor
    action_features: torch.Tensor
    active_edge_mask: torch.Tensor
    added_edge_mask: torch.Tensor
    removed_edge_mask: torch.Tensor
    targets: torch.Tensor
    action_kind_index: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphActionPrediction:
    utility: torch.Tensor
    risk_logit: torch.Tensor
    type_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphActionLoss:
    total: torch.Tensor
    utility: torch.Tensor
    risk: torch.Tensor
    ranking: torch.Tensor


@dataclass(frozen=True)
class SingleGroupOverfitResult:
    initial_loss: float
    final_loss: float
    target_best_action: int
    predicted_best_action: int
    final_utility_correlation: float | None


@dataclass(frozen=True)
class GraphActionEvaluation:
    group_count: int
    mean_loss: float
    exact_action_match_rate: float
    action_kind_match_rate: float
    mean_selected_position_rmse_reduction: float
    positive_selected_gain_rate: float
    selected_nees_violation_rate: float
    utility_correlation: float | None


@dataclass(frozen=True)
class GraphActionTrainingResult:
    model: "GraphActionValueNetwork"
    best_epoch: int
    epochs_run: int
    initial_training: GraphActionEvaluation
    initial_validation: GraphActionEvaluation
    final_training: GraphActionEvaluation
    best_validation: GraphActionEvaluation


@dataclass(frozen=True)
class SnapshotGraphActionEvaluation:
    group_count: int
    mean_loss: float
    exact_action_match_rate: float
    action_kind_match_rate: float
    mean_selected_position_rmse_reduction: float
    mean_oracle_regret: float
    positive_selected_gain_rate: float
    utility_correlation: float | None
    conditional_action_match_rate: float = 0.0
    conditional_top3_match_rate: float = 0.0


@dataclass(frozen=True)
class SnapshotGraphActionTrainingResult:
    model: "GraphActionValueNetwork"
    best_epoch: int
    epochs_run: int
    initial_training: SnapshotGraphActionEvaluation
    initial_validation: SnapshotGraphActionEvaluation
    final_training: SnapshotGraphActionEvaluation
    best_validation: SnapshotGraphActionEvaluation


def save_snapshot_action_checkpoint(
    result: SnapshotGraphActionTrainingResult,
    dataset: SnapshotActionTensorDataset,
    output_path: str | Path,
    *, configuration: dict,
) -> Path:
    """Save a warm-start-compatible hierarchical snapshot checkpoint."""

    if configuration.get("loss_mode") != "hierarchical":
        raise ValueError("PPO warm start requires a hierarchical checkpoint.")
    if not dataset.groups:
        raise ValueError("Checkpoint dataset cannot be empty.")
    sample = torch_snapshot_action_group(dataset.groups[0])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": result.model.state_dict(),
        "configuration": dict(configuration),
        "feature_version": dataset.feature_version,
        "node_feature_count": sample.node_features.shape[1],
        "edge_feature_count": sample.candidate_edge_features.shape[1],
        "global_feature_count": len(dataset.global_feature_names),
        "action_feature_count": sample.action_features.shape[1],
    }, path)
    return path


def torch_graph_action_group(
    group: GraphActionTensorGroup,
    *,
    device: str | torch.device = "cpu",
) -> TorchGraphActionGroup:
    """Convert immutable NumPy tensors without using any outcome for scaling."""

    def floating(values, normalize=False):
        tensor = torch.as_tensor(np.array(values, copy=True), dtype=torch.float32)
        if normalize and tensor.numel() and tensor.shape[0] > 1:
            mean = tensor.mean(dim=0, keepdim=True)
            scale = tensor.std(dim=0, unbiased=False, keepdim=True)
            tensor = (tensor - mean) / torch.where(scale > 1e-6, scale, 1.0)
        return tensor.to(device)

    return TorchGraphActionGroup(
        node_features=floating(group.node_features, normalize=True),
        candidate_edge_index=torch.as_tensor(
            np.array(group.candidate_edge_index, copy=True), dtype=torch.long,
            device=device,
        ),
        candidate_edge_features=floating(
            group.candidate_edge_features, normalize=True
        ),
        measurement_edge_index=torch.as_tensor(
            np.array(group.measurement_edge_index, copy=True), dtype=torch.long,
            device=device,
        ),
        measurement_features=floating(
            group.measurement_features, normalize=True
        ),
        action_features=floating(group.action_features, normalize=True),
        active_edge_mask=floating(group.active_edge_mask),
        added_edge_mask=floating(group.added_edge_mask),
        removed_edge_mask=floating(group.removed_edge_mask),
        targets=floating(group.targets),
        action_kind_index=None,
    )


def torch_snapshot_action_group(
    group: SnapshotActionTensorGroup,
    *,
    device: str | torch.device = "cpu",
) -> TorchGraphActionGroup:
    """Adapt one online V15 snapshot to the existing action-value GNN.

    Candidate links are duplicated in both directions for message passing.  This
    is an interface/overfit bridge until observation-message edges are exposed
    separately by the deployment-safe policy observation.
    """

    def floating(values, normalize=False):
        tensor = torch.as_tensor(np.array(values, copy=True), dtype=torch.float32)
        if normalize and tensor.numel() and tensor.shape[0] > 1:
            mean = tensor.mean(dim=0, keepdim=True)
            scale = tensor.std(dim=0, unbiased=False, keepdim=True)
            tensor = (tensor - mean) / torch.where(scale > 1e-6, scale, 1.0)
        return tensor.to(device)

    tensor = group.policy_tensor
    edge_index = np.array(tensor.edge_index, copy=True)
    directed_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)
    directed_features = np.concatenate(
        (tensor.edge_features, tensor.edge_features), axis=0
    )
    normalized_actions = floating(group.action_features, normalize=True)
    global_features = floating(tensor.global_features).unsqueeze(0).expand(
        len(group.action_features), -1
    )
    action_features = torch.cat((normalized_actions, global_features), dim=1)
    return TorchGraphActionGroup(
        node_features=floating(tensor.node_features, normalize=True),
        candidate_edge_index=torch.as_tensor(
            edge_index, dtype=torch.long, device=device,
        ),
        candidate_edge_features=floating(
            tensor.edge_features, normalize=True
        ),
        measurement_edge_index=torch.as_tensor(
            directed_index, dtype=torch.long, device=device,
        ),
        measurement_features=floating(directed_features, normalize=True),
        action_features=action_features,
        active_edge_mask=floating(group.active_edge_mask),
        added_edge_mask=floating(group.added_edge_mask),
        removed_edge_mask=floating(group.removed_edge_mask),
        targets=floating(group.targets),
        action_kind_index=torch.as_tensor(
            tuple(_snapshot_kind_index(kind) for kind in group.action_kinds),
            dtype=torch.long, device=device,
        ),
    )


class GraphActionValueNetwork(nn.Module):
    """Small directed-message GNN with shared graph and per-action decoding."""

    def __init__(
        self, *, node_feature_count: int, candidate_edge_feature_count: int,
        measurement_feature_count: int, action_feature_count: int,
        hidden_size: int = 32, message_passing_steps: int = 2,
        explicit_action_pairing: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or message_passing_steps <= 0:
            raise ValueError("GNN hidden size and message steps must be positive.")
        self.node_encoder = _mlp(node_feature_count, hidden_size, hidden_size)
        self.measurement_encoder = _mlp(
            measurement_feature_count, hidden_size, hidden_size
        )
        self.message_networks = nn.ModuleList([
            _mlp(2 * hidden_size, hidden_size, hidden_size)
            for _ in range(message_passing_steps)
        ])
        self.node_updates = nn.ModuleList([
            _mlp(2 * hidden_size, hidden_size, hidden_size)
            for _ in range(message_passing_steps)
        ])
        self.candidate_encoder = _mlp(
            2 * hidden_size + candidate_edge_feature_count,
            hidden_size, hidden_size,
        )
        self.explicit_action_pairing = bool(explicit_action_pairing)
        if self.explicit_action_pairing:
            self.action_pair_encoder = _mlp(
                4 * hidden_size, hidden_size, hidden_size
            )
        decoder_input = 2 * hidden_size + 3 * hidden_size + action_feature_count
        if self.explicit_action_pairing:
            decoder_input += hidden_size
        self.action_decoder = _mlp(decoder_input, hidden_size, hidden_size)
        self.utility_head = nn.Linear(hidden_size, 1)
        self.risk_head = nn.Linear(hidden_size, 1)
        self.type_head = nn.Linear(hidden_size, 1)

    def forward(self, group: TorchGraphActionGroup) -> GraphActionPrediction:
        nodes = self.node_encoder(group.node_features)
        measurement_edges = self.measurement_encoder(group.measurement_features)
        source, target = group.measurement_edge_index
        for message_network, update in zip(
            self.message_networks, self.node_updates
        ):
            messages = message_network(torch.cat((nodes[source], measurement_edges), 1))
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(0, target, messages)
            degree = torch.zeros((len(nodes), 1), device=nodes.device)
            degree.index_add_(0, target, torch.ones(
                (len(target), 1), device=nodes.device
            ))
            aggregate = aggregate / degree.clamp_min(1.0)
            nodes = nodes + update(torch.cat((nodes, aggregate), 1))
        left, right = group.candidate_edge_index
        candidates = self.candidate_encoder(torch.cat((
            nodes[left], nodes[right], group.candidate_edge_features,
        ), 1))
        graph = torch.cat((nodes.mean(0), nodes.max(0).values), 0)
        action_count = group.action_features.shape[0]
        graph = graph.unsqueeze(0).expand(action_count, -1)
        active = _masked_mean(group.active_edge_mask, candidates)
        added = _masked_mean(group.added_edge_mask, candidates)
        removed = _masked_mean(group.removed_edge_mask, candidates)
        components = [graph, active, added, removed]
        if self.explicit_action_pairing:
            components.append(self.action_pair_encoder(torch.cat((
                added, removed, added - removed, added * removed,
            ), 1)))
        components.append(group.action_features)
        action_embedding = self.action_decoder(torch.cat(tuple(components), 1))
        type_logits = None
        if group.action_kind_index is not None:
            scores = self.type_head(action_embedding).squeeze(1)
            type_logits = torch.stack(tuple(
                scores[group.action_kind_index == kind].mean()
                if torch.any(group.action_kind_index == kind)
                else scores.new_tensor(-1.0e4)
                for kind in range(4)
            ))
        return GraphActionPrediction(
            utility=self.utility_head(action_embedding).squeeze(1),
            risk_logit=self.risk_head(action_embedding).squeeze(1),
            type_logits=type_logits,
        )


def graph_action_multitask_loss(
    prediction: GraphActionPrediction,
    targets: torch.Tensor,
    *, risk_weight: float = 0.2, ranking_weight: float = 0.2,
) -> GraphActionLoss:
    realized = targets[:, 0]
    scale = realized.abs().max().clamp_min(1e-3)
    normalized = realized / scale
    utility_loss = functional.mse_loss(prediction.utility, normalized)
    risk_target = (targets[:, 2] < 0.0).to(targets.dtype)
    risk_loss = functional.binary_cross_entropy_with_logits(
        prediction.risk_logit, risk_target
    )
    differences = normalized[:, None] - normalized[None, :]
    valid = differences.abs() > 1e-7
    if torch.any(valid):
        predicted = prediction.utility[:, None] - prediction.utility[None, :]
        ranking_loss = functional.softplus(
            -torch.sign(differences[valid]) * predicted[valid]
        ).mean()
    else:
        ranking_loss = prediction.utility.sum() * 0.0
    total = utility_loss + risk_weight * risk_loss + ranking_weight * ranking_loss
    return GraphActionLoss(total, utility_loss, risk_loss, ranking_loss)


def overfit_single_graph_action_group(
    group: GraphActionTensorGroup,
    *, steps: int = 600, learning_rate: float = 3e-3,
    hidden_size: int = 32, random_seed: int = 0,
) -> SingleGroupOverfitResult:
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("Overfit steps and learning rate must be positive.")
    torch.manual_seed(random_seed)
    values = torch_graph_action_group(group)
    model = GraphActionValueNetwork(
        node_feature_count=values.node_features.shape[1],
        candidate_edge_feature_count=values.candidate_edge_features.shape[1],
        measurement_feature_count=values.measurement_features.shape[1],
        action_feature_count=values.action_features.shape[1],
        hidden_size=hidden_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = graph_action_multitask_loss(model(values), values.targets)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = graph_action_multitask_loss(model(values), values.targets)
        loss.total.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(values)
        final = graph_action_multitask_loss(prediction, values.targets)
        target = values.targets[:, 0].cpu().numpy()
        predicted = prediction.utility.cpu().numpy()
    correlation = (
        float(np.corrcoef(target, predicted)[0, 1])
        if np.ptp(target) > 0.0 and np.ptp(predicted) > 0.0 else None
    )
    return SingleGroupOverfitResult(
        initial_loss=float(initial.total), final_loss=float(final.total),
        target_best_action=int(np.argmax(target)),
        predicted_best_action=int(np.argmax(predicted)),
        final_utility_correlation=correlation,
    )


def overfit_single_snapshot_action_group(
    group: SnapshotActionTensorGroup,
    *, steps: int = 600, learning_rate: float = 3e-3,
    hidden_size: int = 32, random_seed: int = 0,
) -> SingleGroupOverfitResult:
    """Check that the V15 snapshot tensors can drive the existing GNN.

    Only the first target (RMSE reduction versus keep) supervises this smoke
    test.  Resource columns remain available for later constrained/RL losses.
    """

    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("Overfit steps and learning rate must be positive.")
    torch.manual_seed(random_seed)
    values = torch_snapshot_action_group(group)
    model = GraphActionValueNetwork(
        node_feature_count=values.node_features.shape[1],
        candidate_edge_feature_count=values.candidate_edge_features.shape[1],
        measurement_feature_count=values.measurement_features.shape[1],
        action_feature_count=values.action_features.shape[1],
        hidden_size=hidden_size,
    )
    target = values.targets[:, 0]
    scale = target.abs().max().clamp_min(1e-3)
    normalized = target / scale

    def loss_value(prediction):
        return _snapshot_utility_loss(prediction.utility, normalized)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = loss_value(model(values))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_value(model(values))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = model(values).utility.cpu().numpy()
        final = loss_value(model(values))
        realized = target.cpu().numpy()
    correlation = (
        float(np.corrcoef(realized, predicted)[0, 1])
        if np.ptp(realized) > 0.0 and np.ptp(predicted) > 0.0 else None
    )
    return SingleGroupOverfitResult(
        initial_loss=float(initial), final_loss=float(final),
        target_best_action=int(np.argmax(realized)),
        predicted_best_action=int(np.argmax(predicted)),
        final_utility_correlation=correlation,
    )


def train_snapshot_action_network(
    training: SnapshotActionTensorDataset,
    validation: SnapshotActionTensorDataset,
    *, epochs: int = 200, learning_rate: float = 1e-3,
    hidden_size: int = 32, message_passing_steps: int = 2,
    patience: int = 40, random_seed: int = 0,
    explicit_action_pairing: bool = True,
    loss_mode: str = "decision",
) -> SnapshotGraphActionTrainingResult:
    """Train V15 snapshot utility with strict seed-disjoint validation."""

    if epochs <= 0 or patience <= 0 or learning_rate <= 0.0:
        raise ValueError("Training epochs, patience, and learning rate must be positive.")
    if loss_mode not in {"regression_ranking", "decision", "hierarchical"}:
        raise ValueError("Unsupported snapshot loss mode.")
    if not training.groups or not validation.groups:
        raise ValueError("Training and validation snapshot sets must be nonempty.")
    training_seeds = {group.seed for group in training.groups}
    validation_seeds = {group.seed for group in validation.groups}
    if training_seeds & validation_seeds:
        raise ValueError("Snapshot training and validation seeds must be disjoint.")
    schemas = (
        training.node_feature_names, training.edge_feature_names,
        training.global_feature_names, training.action_feature_names,
    )
    if schemas != (
        validation.node_feature_names, validation.edge_feature_names,
        validation.global_feature_names, validation.action_feature_names,
    ):
        raise ValueError("Snapshot training and validation schemas differ.")
    torch.manual_seed(random_seed)
    training_groups = tuple(torch_snapshot_action_group(group)
                            for group in training.groups)
    validation_groups = tuple(torch_snapshot_action_group(group)
                              for group in validation.groups)
    sample = training_groups[0]
    model = GraphActionValueNetwork(
        node_feature_count=sample.node_features.shape[1],
        candidate_edge_feature_count=sample.candidate_edge_features.shape[1],
        measurement_feature_count=sample.measurement_features.shape[1],
        action_feature_count=sample.action_features.shape[1],
        hidden_size=hidden_size, message_passing_steps=message_passing_steps,
        explicit_action_pairing=explicit_action_pairing,
    )
    initial_training = evaluate_snapshot_action_network(
        model, training.groups, training_groups, loss_mode=loss_mode
    )
    initial_validation = evaluate_snapshot_action_network(
        model, validation.groups, validation_groups, loss_mode=loss_mode
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_epoch = deepcopy(model.state_dict()), 0
    best_regret = initial_validation.mean_oracle_regret
    best_loss, stale, epochs_run = initial_validation.mean_loss, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for index in torch.randperm(len(training_groups)).tolist():
            optimizer.zero_grad()
            values = training_groups[index]
            realized = values.targets[:, 0]
            normalized = realized / realized.abs().max().clamp_min(1e-3)
            prediction = model(values)
            loss = _snapshot_training_loss(
                prediction, values, normalized, mode=loss_mode
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        metrics = evaluate_snapshot_action_network(
            model, validation.groups, validation_groups, loss_mode=loss_mode
        )
        epochs_run = epoch
        better_regret = metrics.mean_oracle_regret < best_regret - 1e-9
        tied_regret = abs(metrics.mean_oracle_regret - best_regret) <= 1e-9
        if better_regret or (tied_regret and metrics.mean_loss < best_loss - 1e-7):
            best_regret = metrics.mean_oracle_regret
            best_loss, best_epoch = metrics.mean_loss, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return SnapshotGraphActionTrainingResult(
        model=model, best_epoch=best_epoch, epochs_run=epochs_run,
        initial_training=initial_training,
        initial_validation=initial_validation,
        final_training=evaluate_snapshot_action_network(
            model, training.groups, training_groups, loss_mode=loss_mode
        ),
        best_validation=evaluate_snapshot_action_network(
            model, validation.groups, validation_groups, loss_mode=loss_mode
        ),
    )


def evaluate_snapshot_action_network(
    model, numpy_groups, torch_groups, *, loss_mode="decision",
):
    if len(numpy_groups) != len(torch_groups):
        raise ValueError("NumPy and Torch snapshot groups must align.")
    model.eval()
    losses, exact, kinds, selected_gains, regrets = [], [], [], [], []
    conditional_exact, conditional_top3 = [], []
    target_values, predicted_values = [], []
    with torch.no_grad():
        for numpy_group, values in zip(numpy_groups, torch_groups):
            output = model(values)
            prediction = output.utility
            realized_tensor = values.targets[:, 0]
            normalized = realized_tensor / realized_tensor.abs().max().clamp_min(1e-3)
            losses.append(float(_snapshot_training_loss(
                output, values, normalized, mode=loss_mode
            )))
            realized = realized_tensor.cpu().numpy()
            predicted = prediction.cpu().numpy()
            selected = (
                _select_snapshot_action(output, values, predicted)
                if loss_mode == "hierarchical" else int(np.argmax(predicted))
            )
            oracle = float(np.max(realized))
            best = np.flatnonzero(np.isclose(realized, oracle))
            exact.append(selected in set(best.tolist()))
            best_kinds = {numpy_group.action_kinds[index] for index in best}
            kinds.append(numpy_group.action_kinds[selected] in best_kinds)
            selected_gains.append(float(realized[selected]))
            regrets.append(oracle - float(realized[selected]))
            target_values.extend(realized.tolist())
            predicted_values.extend(predicted.tolist())
            oracle_action = int(np.argmax(realized))
            oracle_kind = int(values.action_kind_index[oracle_action])
            kind_candidates = np.flatnonzero(
                values.action_kind_index.cpu().numpy() == oracle_kind
            )
            conditional_order = kind_candidates[
                np.argsort(predicted[kind_candidates])[::-1]
            ]
            conditional_exact.append(oracle_action == conditional_order[0])
            conditional_top3.append(oracle_action in conditional_order[:3])
    correlation = (
        float(np.corrcoef(target_values, predicted_values)[0, 1])
        if np.ptp(target_values) > 0.0 and np.ptp(predicted_values) > 0.0
        else None
    )
    gains = np.asarray(selected_gains)
    return SnapshotGraphActionEvaluation(
        group_count=len(numpy_groups), mean_loss=float(np.mean(losses)),
        exact_action_match_rate=float(np.mean(exact)),
        action_kind_match_rate=float(np.mean(kinds)),
        mean_selected_position_rmse_reduction=float(np.mean(gains)),
        mean_oracle_regret=float(np.mean(regrets)),
        positive_selected_gain_rate=float(np.mean(gains > 0.0)),
        utility_correlation=correlation,
        conditional_action_match_rate=float(np.mean(conditional_exact)),
        conditional_top3_match_rate=float(np.mean(conditional_top3)),
    )


def _snapshot_training_loss(prediction, values, normalized_target, *, mode):
    utility_loss = _snapshot_utility_loss(
        prediction.utility, normalized_target,
        mode="decision" if mode == "hierarchical" else mode,
    )
    if mode != "hierarchical":
        return utility_loss
    oracle_action = int(torch.argmax(normalized_target))
    oracle_kind = values.action_kind_index[oracle_action].unsqueeze(0)
    type_loss = functional.cross_entropy(
        prediction.type_logits.unsqueeze(0), oracle_kind
    )
    kind_mask = values.action_kind_index == oracle_kind.item()
    within_kind = prediction.utility[kind_mask]
    within_target = normalized_target[kind_mask]
    conditional = -(functional.softmax(within_target / 0.15, dim=0)
                    * functional.log_softmax(within_kind, dim=0)).sum()
    return utility_loss + type_loss + conditional


def _select_snapshot_action(output, values, predicted):
    if output.type_logits is None:
        return int(np.argmax(predicted))
    selected_kind = int(torch.argmax(output.type_logits))
    candidates = np.flatnonzero(
        values.action_kind_index.cpu().numpy() == selected_kind
    )
    if not len(candidates):
        return int(np.argmax(predicted))
    return int(candidates[np.argmax(predicted[candidates])])


def _snapshot_kind_index(kind):
    try:
        return ("keep", "add", "swap", "remove").index(str(kind).lower())
    except ValueError as error:
        raise ValueError(f"Unsupported snapshot action kind {kind!r}.") from error


def _snapshot_utility_loss(utility, normalized_target, *, mode="regression_ranking"):
    regression = functional.mse_loss(utility, normalized_target)
    if mode == "decision":
        temperature = 0.15
        soft_target = functional.softmax(normalized_target / temperature, dim=0)
        listwise = -(soft_target * functional.log_softmax(utility, dim=0)).sum()
        positive = (normalized_target > 0.0).to(normalized_target.dtype)
        positive_loss = functional.binary_cross_entropy_with_logits(
            utility, positive
        )
        return 0.15 * regression + listwise + 0.25 * positive_loss
    if mode != "regression_ranking":
        raise ValueError("Unsupported snapshot utility loss mode.")
    differences = normalized_target[:, None] - normalized_target[None, :]
    valid = differences.abs() > 1e-7
    if not torch.any(valid):
        return regression
    predicted = utility[:, None] - utility[None, :]
    ranking = functional.softplus(
        -torch.sign(differences[valid]) * predicted[valid]
    ).mean()
    return regression + 0.2 * ranking


def train_graph_action_network(
    training: GraphActionTensorDataset,
    validation: GraphActionTensorDataset,
    *, epochs: int = 200, learning_rate: float = 1e-3,
    hidden_size: int = 32, message_passing_steps: int = 2,
    patience: int = 40, random_seed: int = 0,
) -> GraphActionTrainingResult:
    """Train graph-by-graph with seed-disjoint validation and early stopping."""

    if epochs <= 0 or patience <= 0 or learning_rate <= 0.0:
        raise ValueError("Training epochs, patience, and learning rate must be positive.")
    if not training.groups or not validation.groups:
        raise ValueError("Training and validation graph sets must be nonempty.")
    training_seeds = {group.group_id[1] for group in training.groups}
    validation_seeds = {group.group_id[1] for group in validation.groups}
    if training_seeds & validation_seeds:
        raise ValueError("GNN training and validation seeds must be disjoint.")
    schemas = (
        training.node_feature_names,
        training.candidate_edge_feature_names,
        training.measurement_feature_names,
        training.action_feature_names,
    )
    if schemas != (
        validation.node_feature_names,
        validation.candidate_edge_feature_names,
        validation.measurement_feature_names,
        validation.action_feature_names,
    ):
        raise ValueError("GNN training and validation schemas differ.")
    torch.manual_seed(random_seed)
    model = GraphActionValueNetwork(
        node_feature_count=len(training.node_feature_names),
        candidate_edge_feature_count=len(training.candidate_edge_feature_names),
        measurement_feature_count=len(training.measurement_feature_names),
        action_feature_count=len(training.action_feature_names),
        hidden_size=hidden_size,
        message_passing_steps=message_passing_steps,
    )
    training_groups = tuple(torch_graph_action_group(group)
                            for group in training.groups)
    validation_groups = tuple(torch_graph_action_group(group)
                              for group in validation.groups)
    initial_training = evaluate_graph_action_network(
        model, training.groups, training_groups
    )
    initial_validation = evaluate_graph_action_network(
        model, validation.groups, validation_groups
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Epoch zero is a valid model-selection candidate. This guarantees that
    # early stopping never returns weights that are worse than initialization
    # on the seed-disjoint validation split.
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = initial_validation.mean_loss
    stale = 0
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(training_groups)).tolist()
        for index in order:
            optimizer.zero_grad()
            values = training_groups[index]
            loss = graph_action_multitask_loss(model(values), values.targets)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        validation_metrics = evaluate_graph_action_network(
            model, validation.groups, validation_groups
        )
        epochs_run = epoch
        if validation_metrics.mean_loss < best_loss - 1e-7:
            best_loss = validation_metrics.mean_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return GraphActionTrainingResult(
        model=model, best_epoch=best_epoch, epochs_run=epochs_run,
        initial_training=initial_training,
        initial_validation=initial_validation,
        final_training=evaluate_graph_action_network(
            model, training.groups, training_groups
        ),
        best_validation=evaluate_graph_action_network(
            model, validation.groups, validation_groups
        ),
    )


def evaluate_graph_action_network(model, numpy_groups, torch_groups):
    if len(numpy_groups) != len(torch_groups):
        raise ValueError("NumPy and Torch graph groups must align.")
    model.eval()
    losses, exact, kind_matches, selected_gains = [], [], [], []
    selected_nees, target_values, predicted_values = [], [], []
    with torch.no_grad():
        for numpy_group, values in zip(numpy_groups, torch_groups):
            prediction = model(values)
            loss = graph_action_multitask_loss(prediction, values.targets)
            losses.append(float(loss.total))
            realized = values.targets[:, 0].cpu().numpy()
            predicted = prediction.utility.cpu().numpy()
            selected = int(np.argmax(predicted))
            oracle = float(np.max(realized))
            best = np.flatnonzero(np.isclose(realized, oracle))
            exact.append(selected in set(best.tolist()))
            best_kinds = {numpy_group.action_kinds[index] for index in best}
            kind_matches.append(numpy_group.action_kinds[selected] in best_kinds)
            selected_gains.append(float(realized[selected]))
            selected_nees.append(float(values.targets[selected, 2]) < 0.0)
            target_values.extend(realized.tolist())
            predicted_values.extend(predicted.tolist())
    correlation = (
        float(np.corrcoef(target_values, predicted_values)[0, 1])
        if np.ptp(target_values) > 0.0 and np.ptp(predicted_values) > 0.0
        else None
    )
    gains = np.asarray(selected_gains)
    return GraphActionEvaluation(
        group_count=len(numpy_groups),
        mean_loss=float(np.mean(losses)) if losses else 0.0,
        exact_action_match_rate=float(np.mean(exact)) if exact else 0.0,
        action_kind_match_rate=(
            float(np.mean(kind_matches)) if kind_matches else 0.0
        ),
        mean_selected_position_rmse_reduction=(
            float(np.mean(gains)) if gains.size else 0.0
        ),
        positive_selected_gain_rate=(
            float(np.mean(gains > 0.0)) if gains.size else 0.0
        ),
        selected_nees_violation_rate=(
            float(np.mean(selected_nees)) if selected_nees else 0.0
        ),
        utility_correlation=correlation,
    )


def _mlp(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, output_size), nn.ReLU(),
    )


def _masked_mean(mask, values):
    count = mask.sum(1, keepdim=True).clamp_min(1.0)
    return mask @ values / count
