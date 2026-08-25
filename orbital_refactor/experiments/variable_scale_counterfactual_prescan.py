from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np

from experiments.topology_ppo_stage1 import build_stage1_environment
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def run_variable_scale_counterfactual_prescan(
    *,
    condition_seeds: tuple[int, ...],
    noise_seed: int = 0,
    decision_indices: tuple[int, ...] = (0, 3, 6, 9),
    horizon_decisions: int = 2,
    maximum_actions_per_kind: int = 4,
    reference_policy=None,
) -> dict[str, object]:
    """Audit bounded local action opportunities along an always-keep trajectory.

    Truth-derived rewards are used only for this offline audit. The policy input
    and reference-policy decision remain deployment-safe GraphObservation data.
    """

    if not condition_seeds or horizon_decisions < 1:
        raise ValueError("Prescan conditions and horizon must be nonempty and positive.")
    if maximum_actions_per_kind < 1:
        raise ValueError("At least one action per non-keep kind must be evaluated.")
    requested_decisions = set(int(value) for value in decision_indices)
    curriculum = VariableScaleTopologyCurriculum()
    records = []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        environment = build_stage1_environment(configuration)
        state = environment.reset(
            seed=noise_seed, condition_seed=condition_seed,
        )
        decision_index = 0
        while True:
            if decision_index in requested_decisions:
                records.append(_audit_state(
                    environment, state,
                    condition_seed=condition_seed,
                    noise_seed=noise_seed,
                    decision_index=decision_index,
                    node_count=configuration.node_count,
                    horizon_decisions=horizon_decisions,
                    maximum_actions_per_kind=maximum_actions_per_kind,
                    reference_policy=reference_policy,
                ))
            step = environment.step(0)
            state = step.state
            decision_index += 1
            if step.terminated or step.truncated:
                break
    return {
        "condition_seeds": list(condition_seeds),
        "noise_seed": int(noise_seed),
        "decision_indices": sorted(requested_decisions),
        "horizon_decisions": int(horizon_decisions),
        "maximum_actions_per_kind": int(maximum_actions_per_kind),
        "records": records,
        "summary_by_node_count": _summaries(records),
        "summary_by_decision_index": _summaries(records, key="decision_index"),
        "overall": _signal_summary(records),
    }

def _audit_state(
    environment, state, *, condition_seed, noise_seed, decision_index,
    node_count, horizon_decisions, maximum_actions_per_kind, reference_policy,
):
    action_space = state.action_space
    selected_ids = _bounded_action_ids(
        action_space, maximum_actions_per_kind,
    )
    reference_action_id = None
    if reference_policy is not None:
        reference_action_id = int(reference_policy.select_action(state))
        if bool(action_space.legal_mask[reference_action_id]):
            selected_ids = tuple(sorted(set(selected_ids) | {reference_action_id}))
    outcomes = []
    for action_id in selected_ids:
        action = action_space.actions[action_id]
        trial = deepcopy(environment)
        cumulative_reward = 0.0
        for horizon_index in range(horizon_decisions):
            step = trial.step(action_id if horizon_index == 0 else 0)
            cumulative_reward += float(step.reward)
            if step.terminated or step.truncated:
                break
        outcomes.append({
            "action_id": int(action_id),
            "kind": action.kind,
            "added_edges": [list(edge) for edge in action.added_edges],
            "removed_edges": [list(edge) for edge in action.removed_edges],
            "cumulative_reward": cumulative_reward,
        })
    keep_reward = next(
        item["cumulative_reward"] for item in outcomes if item["kind"] == "keep"
    )
    for item in outcomes:
        item["gain_over_keep"] = item["cumulative_reward"] - keep_reward
    nonkeep = [item for item in outcomes if item["kind"] != "keep"]
    best_nonkeep = max(nonkeep, key=lambda item: item["gain_over_keep"])
    legal_counts = Counter(
        action.kind
        for action, legal in zip(action_space.actions, action_space.legal_mask)
        if legal
    )
    positive_counts = Counter(
        item["kind"] for item in nonkeep if item["gain_over_keep"] > 0.0
    )
    timestamp = float(state.observation.timestamp)
    active_events = sum(
        start <= timestamp <= end
        for (left, right), (start, end, _loss, _delay)
        in environment._episode_conditions["dynamic_link_events_by_link"].items()
        if left < right
    )
    reference = next(
        (item for item in outcomes if item["action_id"] == reference_action_id),
        None,
    )
    return {
        "condition_seed": int(condition_seed),
        "noise_seed": int(noise_seed),
        "node_count": int(node_count),
        "decision_index": int(decision_index),
        "timestamp": timestamp,
        "active_dynamic_link_event_count": int(active_events),
        "legal_action_kind_counts": dict(sorted(legal_counts.items())),
        "evaluated_action_count": len(outcomes),
        "keep_cumulative_reward": float(keep_reward),
        "positive_nonkeep_kind_counts": dict(sorted(positive_counts.items())),
        "best_nonkeep": best_nonkeep,
        "reference_action": reference,
        "outcomes": outcomes,
    }


def _bounded_action_ids(action_space, maximum_per_kind):
    by_kind = defaultdict(list)
    for action, legal in zip(action_space.actions, action_space.legal_mask):
        if legal:
            by_kind[action.kind].append(action.action_id)
    selected = [0]
    for kind in ("add", "swap", "remove"):
        values = by_kind[kind]
        if len(values) <= maximum_per_kind:
            selected.extend(values)
            continue
        indices = np.linspace(
            0, len(values) - 1, maximum_per_kind, dtype=int,
        )
        selected.extend(values[int(index)] for index in indices)
    return tuple(sorted(set(selected)))


def _summaries(records, *, key="node_count"):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return {
        str(group_value): {
            "audited_decision_count": len(values),
            "positive_best_nonkeep_count": sum(
                item["best_nonkeep"]["gain_over_keep"] > 0.0
                for item in values
            ),
            "positive_later_best_nonkeep_count": sum(
                item["decision_index"] > 0
                and item["best_nonkeep"]["gain_over_keep"] > 0.0
                for item in values
            ),
            "best_nonkeep_kind_counts": dict(sorted(Counter(
                item["best_nonkeep"]["kind"] for item in values
            ).items())),
            "mean_best_nonkeep_gain": float(np.mean([
                item["best_nonkeep"]["gain_over_keep"] for item in values
            ])),
            **_signal_summary(values),
        }
        for group_value, values in sorted(grouped.items())
    }


def _signal_summary(records):
    keep = np.asarray([
        item["keep_cumulative_reward"] for item in records
    ], dtype=float)
    all_nonkeep = np.asarray([
        outcome["gain_over_keep"]
        for item in records for outcome in item["outcomes"]
        if outcome["kind"] != "keep"
    ], dtype=float)
    best = np.asarray([
        item["best_nonkeep"]["gain_over_keep"] for item in records
    ], dtype=float)
    reference = np.asarray([
        item["reference_action"]["gain_over_keep"]
        for item in records if item["reference_action"] is not None
    ], dtype=float)
    keep_rms = float(np.sqrt(np.mean(keep ** 2)))
    nonkeep_rms = float(np.sqrt(np.mean(all_nonkeep ** 2)))
    return {
        "keep_reward_mean": float(np.mean(keep)),
        "keep_reward_rms": keep_rms,
        "all_nonkeep_gain_mean": float(np.mean(all_nonkeep)),
        "all_nonkeep_gain_mean_absolute": float(np.mean(np.abs(all_nonkeep))),
        "all_nonkeep_gain_rms": nonkeep_rms,
        "nonkeep_to_keep_rms_ratio": (
            0.0 if keep_rms <= 1.0e-15 else nonkeep_rms / keep_rms
        ),
        "best_nonkeep_gain_mean": float(np.mean(best)),
        "best_nonkeep_positive_fraction": float(np.mean(best > 0.0)),
        "reference_gain_mean": (
            None if not len(reference) else float(np.mean(reference))
        ),
        "reference_gain_positive_fraction": (
            None if not len(reference) else float(np.mean(reference > 0.0))
        ),
    }
