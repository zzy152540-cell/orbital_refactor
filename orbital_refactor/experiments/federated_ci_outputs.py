from __future__ import annotations

from cooperative.network_schmidt_runner import (
    NetworkModuleOutput,
    NetworkRuntimeDiagnostics,
)
from interfaces.data_objects import (
    FusionStatus,
    ModuleOutput,
    RuntimeStatus,
    StateOutput,
)
from orbital_core.dynamics import accel_two_body_j2
from orbital_core.measurement_semantics import PHYSICAL_SENSOR_MODALITIES
from orbital_core.quality import quality_score_from_covariance


def build_federated_module_outputs(
    *, timestamps, fused_state_by_node, fused_covariance_by_node,
    fusion_weights_by_node, modality_validity_by_node, local_histories,
    processing_time, topology_version, active_neighbors_by_node,
):
    """Adapt federated CI histories to the standard network output contract."""

    outputs = {}
    node_ids = list(fused_state_by_node)
    local_outputs = {
        modality: history.to_module_outputs(
            processing_time=processing_time,
            topology_version=topology_version,
            topology_transition_count=topology_version,
            active_neighbors_by_node=active_neighbors_by_node,
        )
        for modality, history in local_histories.items()
    }
    for node_id in node_ids:
        state = fused_state_by_node[node_id][-1]
        covariance = fused_covariance_by_node[node_id][-1]
        weights = fusion_weights_by_node[node_id][-1]
        valid_flags = modality_validity_by_node[node_id][-1]
        active_modalities = [
            modality for modality, valid in valid_flags.items() if valid
        ]
        abnormal_events = [
            event
            for modality in PHYSICAL_SENSOR_MODALITIES
            for event in local_outputs[modality][
                node_id
            ].module_output.abnormal_events
        ]
        local_diagnostics = [
            local_outputs[modality][node_id].network_diagnostics
            for modality in PHYSICAL_SENSOR_MODALITIES
        ]
        observation_count = sum(
            local_outputs[modality][
                node_id
            ].module_output.runtime_status.observation_count
            for modality in PHYSICAL_SENSOR_MODALITIES
        )
        status = (
            "DEGRADED" if any(
                event.severity == "ERROR" for event in abnormal_events
            )
            else "OK" if active_modalities
            else "NAVIGATION_ONLY" if any(
                tracked_modality == "ABSOLUTE_POSITION"
                for tracked_modality in local_histories[
                    PHYSICAL_SENSOR_MODALITIES[0]
                ].modality_history_by_node[node_id][-1].values()
            )
            else "PREDICTION_ONLY"
        )
        module_output = ModuleOutput(
            state_output=StateOutput(
                timestamp=float(timestamps[-1]),
                target_id=node_id,
                position_estimate=state[:3].copy(),
                velocity_estimate=state[3:].copy(),
                acceleration_estimate=accel_two_body_j2(state[:3]),
                covariance=covariance.copy(),
                valid_flag=True,
                confidence_level=quality_score_from_covariance(covariance),
            ),
            fusion_status=FusionStatus(
                modality_weights=dict(weights),
                modality_valid_flags=dict(valid_flags),
                active_nodes=node_ids,
            ),
            abnormal_events=abnormal_events,
            runtime_status=RuntimeStatus(
                processing_time=float(processing_time),
                observation_count=observation_count,
                active_modality_count=len(active_modalities),
                active_node_count=len(node_ids),
                status=status,
            ),
        )
        diagnostics = _combine_network_diagnostics(
            node_id=node_id,
            local_diagnostics=local_diagnostics,
            topology_version=topology_version,
            active_neighbors=active_neighbors_by_node[node_id],
        )
        outputs[node_id] = NetworkModuleOutput(module_output, diagnostics)
    return outputs


def _combine_network_diagnostics(
    *, node_id, local_diagnostics, topology_version, active_neighbors,
):
    reference = local_diagnostics[0]
    return NetworkRuntimeDiagnostics(
        node_id=node_id,
        neighbor_count=reference.neighbor_count,
        replay_count=sum(item.replay_count for item in local_diagnostics),
        replay_batch_count=sum(
            item.replay_batch_count for item in local_diagnostics
        ),
        replay_fallback_count=sum(
            item.replay_fallback_count for item in local_diagnostics
        ),
        maximum_replay_seconds=max(
            item.maximum_replay_seconds for item in local_diagnostics
        ),
        maximum_retained_journal_count=max(
            item.maximum_retained_journal_count for item in local_diagnostics
        ),
        configured_neighbors=reference.configured_neighbors,
        link_health_by_neighbor=dict(reference.link_health_by_neighbor),
        last_receive_timestamp_by_neighbor=dict(
            reference.last_receive_timestamp_by_neighbor
        ),
        losses_before_last_delivery_by_neighbor=dict(
            reference.losses_before_last_delivery_by_neighbor
        ),
        maximum_consecutive_losses_by_neighbor=dict(
            reference.maximum_consecutive_losses_by_neighbor
        ),
        recovery_delivery_count_by_neighbor=dict(
            reference.recovery_delivery_count_by_neighbor
        ),
        resynchronization_required_by_neighbor=dict(
            reference.resynchronization_required_by_neighbor
        ),
        message_rejection_counts_by_reason={
            reason: sum(
                item.message_rejection_counts_by_reason.get(reason, 0)
                for item in local_diagnostics
            )
            for reason in {
                key
                for item in local_diagnostics
                for key in item.message_rejection_counts_by_reason
            }
        },
        maximum_checkpoint_count=max(
            item.maximum_checkpoint_count for item in local_diagnostics
        ),
        maximum_pinned_checkpoint_count=max(
            item.maximum_pinned_checkpoint_count for item in local_diagnostics
        ),
        maximum_resync_required_count=max(
            item.maximum_resync_required_count for item in local_diagnostics
        ),
        current_topology_version=int(topology_version),
        topology_transition_count=int(topology_version),
        active_neighbors=tuple(sorted(active_neighbors)),
        inactive_configured_neighbors=tuple(
            neighbor
            for neighbor in reference.configured_neighbors
            if neighbor not in active_neighbors
        ),
    )
