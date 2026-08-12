from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cooperative.topology_policy import UndirectedEdge
from experiments.short_horizon_topology_counterfactual import (
    run_short_horizon_topology_counterfactual,
)


@dataclass(frozen=True)
class OnlineActionStabilityRecord:
    seed: int
    future_seed: int
    best_action_kind: str
    best_active_edges: tuple[UndirectedEdge, ...]
    action_ranking: tuple[tuple[str, tuple[UndirectedEdge, ...]], ...]
    best_position_rmse: float
    keep_position_rmse: float
    best_rmse_reduction: float
    best_mean_nees: float
    best_resynchronization_count: int
    stale_topology_message_count: int
    protocol_rejection_count: int


@dataclass(frozen=True)
class OnlineCounterfactualStabilityScan:
    packet_loss: float
    communication_delay: float
    records: tuple[OnlineActionStabilityRecord, ...]
    best_action_counts: tuple[
        tuple[str, tuple[UndirectedEdge, ...], int], ...
    ]
    unique_best_action_count: int
    mean_best_rmse_reduction: float
    positive_best_gain_rate: float
    total_stale_topology_message_count: int
    total_protocol_rejection_count: int


def run_online_counterfactual_stability_scan(
    *, seeds: Iterable[int], future_seed_offset: int = 100,
    node_count: int = 3, decision_epoch: int = 2,
    horizon_epochs: int = 3, dt: float = 2.0,
    relative_modalities: tuple[str, ...] = (
        "RANGE", "RANGE_RATE", "AZ_EL",
    ),
    batch_relative_observations: bool = True,
    packet_loss: float = 0.0, communication_delay: float = 0.0,
) -> OnlineCounterfactualStabilityScan:
    """Scan online topology-action stability across fixed-prefix seeds."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique.")
    records = []
    action_counts = {}
    for seed in seed_values:
        future_seed = int(future_seed_offset) + seed
        result = run_short_horizon_topology_counterfactual(
            node_count=node_count, seed=seed, future_seed=future_seed,
            decision_epoch=decision_epoch, horizon_epochs=horizon_epochs,
            dt=dt, relative_modalities=relative_modalities,
            future_batch_relative_observations=batch_relative_observations,
            packet_loss=packet_loss,
            communication_delay=communication_delay,
            backend="online_orchestrator",
        )
        ordered = tuple(sorted(
            result.rollouts, key=lambda item: item.metrics.position_rmse
        ))
        best = ordered[0]
        keep = next(
            item for item in result.rollouts if item.action.kind == "keep"
        )
        stale = protocol = 0
        for rollout in result.rollouts:
            for _, _, outcome, count in rollout.state_message_outcome_counts:
                if outcome == "inactive_topology_link":
                    stale += int(count)
                elif outcome != "accepted":
                    protocol += int(count)
        action_key = (best.action.kind, best.action.topology.active_edges)
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        records.append(OnlineActionStabilityRecord(
            seed=seed, future_seed=future_seed,
            best_action_kind=best.action.kind,
            best_active_edges=best.action.topology.active_edges,
            action_ranking=tuple(
                (item.action.kind, item.action.topology.active_edges)
                for item in ordered
            ),
            best_position_rmse=float(best.metrics.position_rmse),
            keep_position_rmse=float(keep.metrics.position_rmse),
            best_rmse_reduction=float(
                keep.metrics.position_rmse - best.metrics.position_rmse
            ),
            best_mean_nees=float(best.metrics.mean_nees),
            best_resynchronization_count=int(
                best.metrics.resynchronization_count
            ),
            stale_topology_message_count=stale,
            protocol_rejection_count=protocol,
        ))
    reductions = np.asarray(
        [record.best_rmse_reduction for record in records], dtype=float
    )
    return OnlineCounterfactualStabilityScan(
        packet_loss=float(packet_loss),
        communication_delay=float(communication_delay),
        records=tuple(records),
        best_action_counts=tuple(
            (kind, edges, count)
            for (kind, edges), count in sorted(
                action_counts.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ),
        unique_best_action_count=len(action_counts),
        mean_best_rmse_reduction=float(np.mean(reductions)),
        positive_best_gain_rate=float(np.mean(reductions > 0.0)),
        total_stale_topology_message_count=sum(
            record.stale_topology_message_count for record in records
        ),
        total_protocol_rejection_count=sum(
            record.protocol_rejection_count for record in records
        ),
    )
