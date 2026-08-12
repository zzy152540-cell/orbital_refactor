from __future__ import annotations

from dataclasses import replace

import numpy as np

from cooperative.multi_neighbor_replay_coordinator import (
    ReplayPerformanceStats,
)
from cooperative.network_schmidt_runner import (
    NetworkSchmidtHistory,
    _relative_update_record,
)


def network_history_from_orchestrator(
    orchestrator,
) -> NetworkSchmidtHistory:
    """Convert completed online steps to the shared history schema."""

    steps = orchestrator.history_snapshot()
    if not steps:
        raise ValueError("Orchestrator history is empty.")
    nodes = tuple(orchestrator.topology.node_ids)
    times = np.asarray([step.timestamp for step in steps], dtype=float)
    active_states = {
        node: np.asarray([
            step.result_by_node[node].state.active_state for step in steps
        ])
        for node in nodes
    }
    active_covariances = {
        node: np.asarray([
            step.result_by_node[node].state.active_covariance for step in steps
        ])
        for node in nodes
    }
    neighbor_states = {
        node: {
            neighbor: np.asarray([
                step.result_by_node[node].state.neighbor_state_by_id[neighbor]
                for step in steps
            ])
            for neighbor in orchestrator.topology.neighbors(node)
        }
        for node in nodes
    }
    cross = {
        node: {
            neighbor: np.asarray([
                step.result_by_node[node].state.active_cross_covariance(
                    neighbor
                )
                for step in steps
            ])
            for neighbor in orchestrator.topology.neighbors(node)
        }
        for node in nodes
    }
    joint = {
        node: np.asarray([
            step.result_by_node[node].state.joint_covariance for step in steps
        ])
        for node in nodes
    }
    relative_updates = {
        node: [
            tuple(
                _relative_update_record(
                    update,
                    observations,
                )
                for observations, update in step.result_by_node[
                    node
                ].relative_update_results
            )
            for step in steps
        ]
        for node in nodes
    }
    empty_per_epoch = {node: [{} for _ in steps] for node in nodes}
    return NetworkSchmidtHistory(
        timestamps=times,
        active_state_history_by_node=active_states,
        active_covariance_history_by_node=active_covariances,
        active_cross_covariance_history_by_node=cross,
        neighbor_state_history_by_node=neighbor_states,
        joint_covariance_history_by_node=joint,
        nis_history_by_node={
            node: [
                dict(step.result_by_node[node].nis_by_information_id)
                for step in steps
            ]
            for node in nodes
        },
        local_dimension_by_node={
            node: steps[-1].result_by_node[node].state.dimension
            for node in nodes
        },
        refresh_diagnostics=_refresh_counts(steps),
        refresh_diagnostic_records=tuple(
            record for step in steps
            for record in step.message_diagnostic_records
        ),
        replay_performance_by_node={
            node: replace(
                orchestrator.sessions[node].coordinator.performance
            )
            for node in nodes
        },
        integrity_history_by_node={
            node: list(empty_per_epoch[node]) for node in nodes
        },
        modality_history_by_node={
            node: list(empty_per_epoch[node]) for node in nodes
        },
        joint_nis_history_by_node={
            node: [() for _ in steps] for node in nodes
        },
        relative_update_history_by_node=relative_updates,
    )


def _refresh_counts(steps):
    counts = {}
    for step in steps:
        for record in step.message_diagnostic_records:
            key = (
                "accepted" if bool(record.get("accepted"))
                else str(record.get("reason", "unknown"))
            )
            counts[key] = counts.get(key, 0) + 1
    return counts
