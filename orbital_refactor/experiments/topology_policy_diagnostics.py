from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import torch

from experiments.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)


@dataclass(frozen=True)
class TopologyPolicyMarginRecord:
    condition_seed: int
    noise_seed: int
    decision_index: int
    legal_action_count: int
    selected_action_id: int
    selected_action_kind: str
    selected_probability: float
    log_probability_margin: float
    reference_action_id: int
    reference_selected_probability: float
    reference_log_probability_margin: float
    policy_kl_from_reference: float
    type_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class RobustPolicyOverrideRecord:
    condition_seed: int
    decision_epoch: int
    policy_action_index: int
    reference_action_index: int
    policy_action_kind: str
    reference_action_kind: str
    changed_action: bool
    robust_gain_over_reference: float


def audit_policy_overrides_against_robust_targets(
    dataset, policy_model, reference_model,
) -> tuple[RobustPolicyOverrideRecord, ...]:
    """Score policy overrides with noise-aggregated counterfactual targets."""

    if not dataset.groups:
        raise ValueError("Policy override audit requires a nonempty dataset.")
    try:
        gain_index = dataset.target_names.index(
            "position_rmse_reduction_vs_keep"
        )
    except ValueError as error:
        raise ValueError("Robust dataset omits the position-gain target.") from error
    records = []
    for group in dataset.groups:
        values = torch_snapshot_action_group(group)
        with torch.no_grad():
            policy_action = int(policy_model(values).distribution.mode().item())
            reference_action = int(
                reference_model(values).distribution.mode().item()
            )
        targets = group.targets[:, gain_index]
        records.append(RobustPolicyOverrideRecord(
            condition_seed=int(group.seed),
            decision_epoch=int(group.decision_epoch),
            policy_action_index=policy_action,
            reference_action_index=reference_action,
            policy_action_kind=group.action_kinds[policy_action],
            reference_action_kind=group.action_kinds[reference_action],
            changed_action=policy_action != reference_action,
            robust_gain_over_reference=float(
                targets[policy_action] - targets[reference_action]
            ),
        ))
    return tuple(records)


def audit_topology_policy_margins(
    environment,
    policy_model,
    reference_model,
    *,
    condition_seeds: Iterable[int],
    noise_seeds: Iterable[int],
) -> tuple[TopologyPolicyMarginRecord, ...]:
    """Compare a policy with its reference on the policy's deterministic path."""

    conditions = _unique_nonempty(condition_seeds, "condition")
    noises = _unique_nonempty(noise_seeds, "noise")
    records = []
    for condition_seed in conditions:
        for noise_seed in noises:
            branch = deepcopy(environment)
            state = branch.reset(seed=noise_seed, condition_seed=condition_seed)
            decision_index = 0
            while True:
                snapshot, action_ids = build_online_snapshot_action_tensor(state)
                group = torch_snapshot_action_group(snapshot)
                with torch.no_grad():
                    policy = policy_model(group).distribution
                    reference = reference_model(group).distribution
                policy_log = policy.action_log_probabilities
                reference_log = reference.action_log_probabilities
                selected = int(policy.mode().item())
                reference_selected = int(reference.mode().item())
                ordered = torch.topk(policy_log, k=min(2, len(policy_log))).values
                reference_ordered = torch.topk(
                    reference_log, k=min(2, len(reference_log))
                ).values
                margin = (
                    float((ordered[0] - ordered[1]).item())
                    if len(ordered) == 2 else float("inf")
                )
                reference_margin = (
                    float((reference_ordered[0] - reference_ordered[1]).item())
                    if len(reference_ordered) == 2 else float("inf")
                )
                probabilities = policy_log.exp()
                reference_probabilities = reference_log.exp()
                kl = torch.sum(
                    probabilities * (policy_log - reference_log)
                )
                records.append(TopologyPolicyMarginRecord(
                    condition_seed=condition_seed,
                    noise_seed=noise_seed,
                    decision_index=decision_index,
                    legal_action_count=len(action_ids),
                    selected_action_id=int(action_ids[selected]),
                    selected_action_kind=snapshot.action_kinds[selected],
                    selected_probability=float(probabilities[selected].item()),
                    log_probability_margin=margin,
                    reference_action_id=int(action_ids[reference_selected]),
                    reference_selected_probability=float(
                        reference_probabilities[reference_selected].item()
                    ),
                    reference_log_probability_margin=reference_margin,
                    policy_kl_from_reference=float(kl.item()),
                    type_probabilities=tuple(
                        float(value) for value in policy.type_probabilities.tolist()
                    ),
                ))
                step = branch.step(int(action_ids[selected]))
                if step.terminated or step.truncated:
                    break
                state = step.state
                decision_index += 1
    return tuple(records)


def _unique_nonempty(values, label):
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"Policy-audit {label} seeds must be unique/nonempty.")
    return result
