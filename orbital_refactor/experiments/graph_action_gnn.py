from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from experiments.graph_action_tensor_dataset import (
    GraphActionTensorDataset,
    GraphActionTensorGroup,
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


@dataclass(frozen=True)
class GraphActionPrediction:
    utility: torch.Tensor
    risk_logit: torch.Tensor


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
    )


class GraphActionValueNetwork(nn.Module):
    """Small directed-message GNN with shared graph and per-action decoding."""

    def __init__(
        self, *, node_feature_count: int, candidate_edge_feature_count: int,
        measurement_feature_count: int, action_feature_count: int,
        hidden_size: int = 32, message_passing_steps: int = 2,
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
        decoder_input = 2 * hidden_size + 3 * hidden_size + action_feature_count
        self.action_decoder = _mlp(decoder_input, hidden_size, hidden_size)
        self.utility_head = nn.Linear(hidden_size, 1)
        self.risk_head = nn.Linear(hidden_size, 1)

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
        action_embedding = self.action_decoder(torch.cat((
            graph,
            _masked_mean(group.active_edge_mask, candidates),
            _masked_mean(group.added_edge_mask, candidates),
            _masked_mean(group.removed_edge_mask, candidates),
            group.action_features,
        ), 1))
        return GraphActionPrediction(
            utility=self.utility_head(action_embedding).squeeze(1),
            risk_logit=self.risk_head(action_embedding).squeeze(1),
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
