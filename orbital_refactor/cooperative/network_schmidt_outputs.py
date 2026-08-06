from __future__ import annotations

from dataclasses import dataclass

from cooperative.multi_neighbor_replay_coordinator import ReplayPerformanceStats
from interfaces.data_objects import (
    AbnormalEvent,
    FusionStatus,
    ModuleOutput,
    RuntimeStatus,
    StateOutput,
)
from orbital_core.dynamics import accel_two_body_j2
from orbital_core.quality import quality_score_from_covariance


@dataclass(frozen=True)
class NetworkRuntimeDiagnostics:
    node_id: str
    neighbor_count: int
    replay_count: int
    replay_batch_count: int
    replay_fallback_count: int
    maximum_replay_seconds: float
    maximum_retained_journal_count: int
    configured_neighbors: tuple[str, ...]
    link_health_by_neighbor: dict[str, str]
    last_receive_timestamp_by_neighbor: dict[str, float | None]
    losses_before_last_delivery_by_neighbor: dict[str, int]
    maximum_consecutive_losses_by_neighbor: dict[str, int]
    recovery_delivery_count_by_neighbor: dict[str, int]
    resynchronization_required_by_neighbor: dict[str, bool]
    message_rejection_counts_by_reason: dict[str, int]
    maximum_checkpoint_count: int
    maximum_pinned_checkpoint_count: int
    maximum_resync_required_count: int
    current_topology_version: int
    topology_transition_count: int
    active_neighbors: tuple[str, ...]
    inactive_configured_neighbors: tuple[str, ...]


@dataclass(frozen=True)
class NetworkModuleOutput:
    module_output: ModuleOutput
    network_diagnostics: NetworkRuntimeDiagnostics


def network_module_output(
    history, *, node_id: str, processing_time: float,
    link_timeout: float | None, topology_version: int,
    topology_transition_count: int, active_neighbors_by_node,
) -> NetworkModuleOutput:
    """Convert one final network posterior to the formal output schema."""

    if link_timeout is not None and link_timeout <= 0.0:
        raise ValueError("link_timeout must be positive when provided.")
    final_state = history.active_state_history_by_node[node_id][-1]
    final_covariance = history.active_covariance_history_by_node[node_id][-1]
    final_integrity = history.integrity_history_by_node[node_id][-1]
    final_modalities = history.modality_history_by_node[node_id][-1]
    all_modalities = {
        modality
        for epoch in history.modality_history_by_node[node_id]
        for modality in epoch.values()
    }
    modality_valid_flags = {
        modality: False for modality in sorted(all_modalities)
    }
    for information_id, integrity in final_integrity.items():
        modality = final_modalities[information_id]
        modality_valid_flags[modality] = (
            modality_valid_flags[modality] or integrity.accepted
        )
    active_modalities = [
        modality for modality, valid in modality_valid_flags.items() if valid
    ]
    abnormal_events = _integrity_abnormal_events(history, node_id)
    replay = history.replay_performance_by_node.get(
        node_id, ReplayPerformanceStats()
    )
    if replay.fallback_count:
        abnormal_events.append(AbnormalEvent(
            timestamp=float(history.timestamps[-1]),
            event_type="REPLAY_FALLBACK",
            severity="ERROR",
            description=f"Replay fallback count={replay.fallback_count}.",
            node_id=node_id,
            target_id=node_id,
        ))
    observation_count = sum(
        len(epoch) for epoch in history.integrity_history_by_node[node_id]
    )
    status = (
        "DEGRADED" if replay.fallback_count
        or any(event.severity == "ERROR" for event in abnormal_events)
        else "OK" if active_modalities else "PREDICTION_ONLY"
    )
    module_output = ModuleOutput(
        state_output=StateOutput(
            timestamp=float(history.timestamps[-1]),
            target_id=node_id,
            position_estimate=final_state[:3].copy(),
            velocity_estimate=final_state[3:].copy(),
            acceleration_estimate=accel_two_body_j2(final_state[:3]),
            covariance=final_covariance.copy(),
            valid_flag=True,
            confidence_level=quality_score_from_covariance(final_covariance),
        ),
        fusion_status=FusionStatus(
            modality_valid_flags=modality_valid_flags,
            active_nodes=list(history.node_ids),
        ),
        abnormal_events=abnormal_events,
        runtime_status=RuntimeStatus(
            processing_time=float(processing_time),
            observation_count=observation_count,
            active_modality_count=len(active_modalities),
            active_node_count=len(history.node_ids),
            status=status,
        ),
    )
    configured_neighbors = tuple(sorted(
        history.active_cross_covariance_history_by_node[node_id]
    ))
    active_neighbors = tuple(sorted(
        (active_neighbors_by_node or {}).get(node_id, configured_neighbors)
    ))
    diagnostics = NetworkRuntimeDiagnostics(
        node_id=node_id,
        neighbor_count=history.local_dimension_by_node[node_id] // 6 - 1,
        replay_count=replay.replay_count,
        replay_batch_count=replay.batch_count,
        replay_fallback_count=replay.fallback_count,
        maximum_replay_seconds=replay.maximum_replay_seconds,
        maximum_retained_journal_count=replay.maximum_retained_journal_count,
        **link_runtime_diagnostics(
            history, node_id=node_id, link_timeout=link_timeout
        ),
        maximum_checkpoint_count=replay.maximum_checkpoint_count,
        maximum_pinned_checkpoint_count=replay.maximum_pinned_checkpoint_count,
        maximum_resync_required_count=replay.maximum_resync_required_count,
        current_topology_version=int(topology_version),
        topology_transition_count=int(topology_transition_count),
        active_neighbors=active_neighbors,
        inactive_configured_neighbors=tuple(
            neighbor for neighbor in configured_neighbors
            if neighbor not in active_neighbors
        ),
    )
    return NetworkModuleOutput(module_output, diagnostics)


def _integrity_abnormal_events(history, node_id):
    events = []
    for index, timestamp in enumerate(history.timestamps):
        modalities = history.modality_history_by_node[node_id][index]
        for information_id, integrity in (
            history.integrity_history_by_node[node_id][index].items()
        ):
            if not integrity.anomalous:
                continue
            rejected = integrity.status == "HARD_REJECTED"
            events.append(AbnormalEvent(
                timestamp=float(timestamp),
                event_type=(
                    "OBSERVATION_REJECTED" if rejected
                    else "OBSERVATION_DOWNWEIGHTED"
                ),
                severity="ERROR" if rejected else "WARNING",
                description=(
                    f"{information_id} integrity status={integrity.status}; "
                    f"raw_nis={integrity.raw_nis}; "
                    f"processed_nis={integrity.processed_nis}; "
                    "covariance_scale="
                    f"{integrity.measurement_covariance_scale}."
                ),
                node_id=node_id,
                target_id=node_id,
                modality=modalities[information_id],
            ))
    return events


def link_runtime_diagnostics(
    history, *, node_id: str, link_timeout: float | None,
) -> dict[str, object]:
    """Summarize final health and transport history for configured links."""

    configured = tuple(sorted(
        history.active_cross_covariance_history_by_node[node_id]
    ))
    records = [
        record for record in history.refresh_diagnostic_records
        if record.get("receiver_id") == node_id
    ]
    last_receive = {neighbor: None for neighbor in configured}
    losses = {neighbor: 0 for neighbor in configured}
    maximum_losses = {neighbor: 0 for neighbor in configured}
    recovery_deliveries = {neighbor: 0 for neighbor in configured}
    resync = {neighbor: False for neighbor in configured}
    rejection_counts = {}
    for record in records:
        neighbor = str(record.get("source_id"))
        if neighbor not in last_receive:
            continue
        if bool(record.get("accepted")):
            arrival = record.get("arrival_timestamp")
            receive_time = (
                record.get("current_timestamp") if arrival is None else arrival
            )
            last_receive[neighbor] = float(receive_time)
            losses[neighbor] = int(
                record.get("consecutive_losses_before_delivery", 0)
            )
            maximum_losses[neighbor] = max(
                maximum_losses[neighbor], losses[neighbor]
            )
            if losses[neighbor] > 0:
                recovery_deliveries[neighbor] += 1
        else:
            reason = str(record.get("reason", "unknown"))
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        resync[neighbor] = neighbor in set(
            record.get("resync_required_neighbors", ())
        )
    final_timestamp = float(history.timestamps[-1])
    health = {}
    for neighbor in configured:
        if resync[neighbor]:
            health[neighbor] = "RESYNC_REQUIRED"
        elif link_timeout is not None and (
            last_receive[neighbor] is None
            or final_timestamp - float(last_receive[neighbor]) > link_timeout
        ):
            health[neighbor] = "LOST"
        elif last_receive[neighbor] is not None:
            health[neighbor] = "HEALTHY"
        else:
            health[neighbor] = "UNKNOWN"
    return {
        "configured_neighbors": configured,
        "link_health_by_neighbor": health,
        "last_receive_timestamp_by_neighbor": last_receive,
        "losses_before_last_delivery_by_neighbor": losses,
        "maximum_consecutive_losses_by_neighbor": maximum_losses,
        "recovery_delivery_count_by_neighbor": recovery_deliveries,
        "resynchronization_required_by_neighbor": resync,
        "message_rejection_counts_by_reason": rejection_counts,
    }
