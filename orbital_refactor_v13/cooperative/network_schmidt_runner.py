from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_absolute_position_update,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.topology import NetworkTopology
from cooperative.schmidt_refresh import (
    exact_transport_eligibility,
    refresh_consider_neighbor,
)
from cooperative.multi_neighbor_replay_coordinator import (
    MultiNeighborReplayCoordinator,
    ReplayPerformanceStats,
)
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from interfaces.data_objects import (
    AbnormalEvent,
    AbsolutePositionObservation,
    FusionStatus,
    ModuleOutput,
    ObservationMessage,
    RuntimeStatus,
    StateMessage,
    StateOutput,
)
from orbital_core.dynamics import accel_two_body_j2
from orbital_core.measurement_integrity import (
    MeasurementIntegrityDiagnostics,
    MeasurementIntegrityPolicy,
)
from orbital_core.quality import quality_score_from_covariance

Array = np.ndarray


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
    resynchronization_required_by_neighbor: dict[str, bool]
    message_rejection_counts_by_reason: dict[str, int]
    maximum_checkpoint_count: int
    maximum_pinned_checkpoint_count: int
    maximum_resync_required_count: int


@dataclass(frozen=True)
class NetworkModuleOutput:
    module_output: ModuleOutput
    network_diagnostics: NetworkRuntimeDiagnostics


@dataclass(frozen=True)
class NetworkSchmidtHistory:
    timestamps: Array
    active_state_history_by_node: dict[str, Array]
    active_covariance_history_by_node: dict[str, Array]
    active_cross_covariance_history_by_node: dict[str, dict[str, Array]]
    joint_covariance_history_by_node: dict[str, Array]
    nis_history_by_node: dict[str, list[dict[str, float]]]
    local_dimension_by_node: dict[str, int]
    refresh_diagnostics: dict[str, int]
    refresh_diagnostic_records: tuple[dict[str, object], ...]
    replay_performance_by_node: dict[str, ReplayPerformanceStats]
    integrity_history_by_node: dict[
        str, list[dict[str, MeasurementIntegrityDiagnostics]]
    ]
    modality_history_by_node: dict[str, list[dict[str, str]]]

    @property
    def node_ids(self) -> list[str]:
        return list(self.active_state_history_by_node)

    def to_module_outputs(
        self, *, processing_time: float = 0.0,
        link_timeout: float | None = None,
    ) -> dict[str, NetworkModuleOutput]:
        """Convert final per-node posteriors to the shared formal output schema."""

        return {
            node_id: _network_module_output(
                self, node_id=node_id, processing_time=processing_time,
                link_timeout=link_timeout,
            )
            for node_id in self.node_ids
        }


def run_network_schmidt_filter(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
    initial_covariance_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    observation_messages: Iterable[ObservationMessage],
    absolute_position_observations: Iterable[
        AbsolutePositionObservation
    ] = (),
    observation_usage: str = "observer_only",
    process_noise_acceleration: float = 1e-4,
    consider_refresh_mode: str = "propagate_only",
    state_messages_by_receiver: Mapping[str, Iterable[StateMessage]] | None = None,
    replay_history_window: float | None = None,
    expected_lineage_by_link: Mapping[tuple[str, str], str] | None = None,
    max_pinned_age: float | None = None,
    max_retained_events: int | None = None,
    nis_gate_threshold_by_modality: Mapping[str, float] | None = None,
    nis_inflation_threshold_by_modality: Mapping[str, float] | None = None,
    maximum_measurement_covariance_scale_by_modality: Mapping[str, float] | None = None,
    integrity_policy_by_modality: Mapping[
        str, MeasurementIntegrityPolicy
    ] | None = None,
) -> NetworkSchmidtHistory:
    """Run one local multi-neighbor Schmidt filter at every topology node.

    Each node owns a different local augmented covariance. Consider-state
    means are propagated but never corrected inside that node's filter. The
    default routes each physical observation only to its actual observer.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    if observation_usage not in {"observer_only", "both_endpoints"}:
        raise ValueError(
            "observation_usage must be 'observer_only' or 'both_endpoints'."
        )
    if consider_refresh_mode not in {"propagate_only", "safe_rescale", "zero_cross", "exact_if_compatible", "exact_transport_event_replay"}:
        raise ValueError("Unsupported network consider_refresh_mode.")
    if state_messages_by_receiver and consider_refresh_mode != "exact_transport_event_replay":
        raise ValueError("State messages require exact_transport_event_replay mode.")
    node_ids = topology.node_ids
    states = _initial_values(initial_state_by_node, node_ids, (6,), "state")
    covariances = _initial_values(
        initial_covariance_by_node, node_ids, (6, 6), "covariance"
    )
    local_states = {
        node_id: initialize_multi_neighbor_schmidt(
            timestamp=float(times[0]),
            active_node_id=node_id,
            active_state=states[node_id],
            active_covariance=covariances[node_id],
            neighbor_state_by_id={
                neighbor_id: states[neighbor_id]
                for neighbor_id in topology.neighbors(node_id)
            },
            neighbor_covariance_by_id={
                neighbor_id: covariances[neighbor_id]
                for neighbor_id in topology.neighbors(node_id)
            },
        )
        for node_id in node_ids
    }
    coordinators = (
        {
            node_id: MultiNeighborReplayCoordinator(
                local_states[node_id],
                process_noise_acceleration=process_noise_acceleration,
                history_window=replay_history_window,
                max_pinned_age=max_pinned_age,
                max_retained_events=max_retained_events,
                nis_gate_threshold_by_modality=nis_gate_threshold_by_modality,
                nis_inflation_threshold_by_modality=nis_inflation_threshold_by_modality,
                maximum_measurement_covariance_scale_by_modality=(
                    maximum_measurement_covariance_scale_by_modality
                ),
                integrity_policy_by_modality=integrity_policy_by_modality,
            )
            for node_id in node_ids
        }
        if consider_refresh_mode == "exact_transport_event_replay"
        else {}
    )
    pending_state_messages = _prepare_state_messages(
        state_messages_by_receiver or {}, node_ids, topology
    )
    observations_by_time_and_owner = _route_observations(
        observation_messages,
        times=times,
        topology=topology,
        observation_usage=observation_usage,
        allow_delayed=(consider_refresh_mode == "exact_transport_event_replay"),
    )
    absolute_observations_by_time_and_node = _route_absolute_observations(
        absolute_position_observations, times=times, node_ids=set(node_ids),
        allow_delayed=(consider_refresh_mode == "exact_transport_event_replay"),
    )
    active_states = {
        node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids
    }
    active_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float) for node_id in node_ids
    }
    joint_covariances = {
        node_id: np.zeros(
            (times.size, local_states[node_id].dimension, local_states[node_id].dimension),
            dtype=float,
        )
        for node_id in node_ids
    }
    active_cross_covariances = {
        node_id: {
            neighbor_id: np.zeros((times.size, 6, 6), dtype=float)
            for neighbor_id in topology.neighbors(node_id)
        }
        for node_id in node_ids
    }
    nis_history = {node_id: [] for node_id in node_ids}
    integrity_history = {node_id: [] for node_id in node_ids}
    modality_history = {node_id: [] for node_id in node_ids}
    consecutive_anomalies = {node_id: {} for node_id in node_ids}
    refresh_diagnostics = {"accepted": 0, "reference_covariance_mismatch": 0,
                           "reference_mean_mismatch": 0}
    refresh_diagnostic_records = []
    previous_active = {
        node_id: (local_states[node_id].active_state.copy(),
                  local_states[node_id].active_covariance)
        for node_id in node_ids
    }

    for index, timestamp in enumerate(times):
        if consider_refresh_mode == "exact_transport_event_replay":
            if index > 0:
                for coordinator in coordinators.values():
                    coordinator.advance(float(timestamp))
            for node_id in node_ids:
                remaining = []
                available = []
                for message in pending_state_messages[node_id]:
                    arrival = message.timestamp if message.arrival_timestamp is None else message.arrival_timestamp
                    if float(arrival) <= float(timestamp):
                        available.append((
                            message,
                            (expected_lineage_by_link or {}).get(
                                (node_id, str(message.source_node_id))
                            ),
                        ))
                    else:
                        remaining.append(message)
                outcomes = coordinators[node_id].apply_state_messages(tuple(available))
                for (message, _), outcome in zip(available, outcomes):
                    key = "accepted" if outcome.accepted else outcome.reason
                    refresh_diagnostics[key] = refresh_diagnostics.get(key, 0) + 1
                    checkpoints = coordinators[node_id].checkpoint_timestamps
                    refresh_diagnostic_records.append({
                        "receiver_id": node_id,
                        "source_id": str(message.source_node_id),
                        "current_timestamp": float(timestamp),
                        "message_timestamp": float(message.timestamp),
                        "reference_timestamp": message.reference_timestamp,
                        "arrival_timestamp": message.arrival_timestamp,
                        "lineage_id": message.lineage_id,
                        "information_ids": "|".join(message.information_ids),
                        "transport_event_count": len(message.transport_events),
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "checkpoint_count": len(checkpoints),
                        "oldest_checkpoint": min(checkpoints) if checkpoints else None,
                        "newest_checkpoint": max(checkpoints) if checkpoints else None,
                        "pinned_checkpoint_count": coordinators[node_id].pinned_checkpoint_count,
                        "oldest_pinned_timestamp": coordinators[node_id].oldest_pinned_timestamp,
                        "resync_required_count": len(
                            coordinators[node_id].resynchronization_requirements
                        ),
                        **message.metadata,
                    })
                pending_state_messages[node_id] = remaining
                local_states[node_id] = coordinators[node_id].state
        elif index > 0:
            local_states = {
                node_id: multi_neighbor_schmidt_predict(
                    local_states[node_id], float(timestamp),
                    process_noise_acceleration=process_noise_acceleration,
                )
                for node_id in node_ids
            }
            if consider_refresh_mode != "propagate_only":
                snapshots = {
                    node_id: (local_states[node_id].active_state.copy(),
                              local_states[node_id].active_covariance)
                    for node_id in node_ids
                }
                for node_id in node_ids:
                    for neighbor_id in local_states[node_id].neighbor_ids:
                        mean, covariance = snapshots[neighbor_id]
                        if consider_refresh_mode == "exact_if_compatible":
                            reference_mean, reference_covariance = previous_active[neighbor_id]
                            # Compare against the receiver block before this epoch's
                            # prediction, reconstructed by applying the same model.
                            dt_epoch = float(timestamp) - float(times[index - 1])
                            transition = numerical_jacobian_discrete(
                                lambda value: rk4_step_absolute(value, dt_epoch),
                                reference_mean,
                            )
                            noise = make_process_noise(dt_epoch, process_noise_acceleration)
                            predicted_reference_covariance = transition @ reference_covariance @ transition.T + noise
                            eligible, reason = exact_transport_eligibility(
                                local_states[node_id], neighbor_id=neighbor_id,
                                reference_covariance=predicted_reference_covariance,
                                reference_mean=rk4_step_absolute(reference_mean, dt_epoch),
                            )
                            if eligible:
                                # It is already at the exactly transported value;
                                # reapplying would transport twice. Acceptance is
                                # recorded as a provenance/compatibility assertion.
                                refresh_diagnostics["accepted"] += 1
                            else:
                                refresh_diagnostics[reason] += 1
                        else:
                            local_states[node_id] = refresh_consider_neighbor(
                                local_states[node_id], neighbor_id=neighbor_id,
                                neighbor_state=mean, neighbor_covariance=covariance,
                                mode=consider_refresh_mode,
                            )
        for node_id in node_ids:
            state = local_states[node_id]
            epoch_nis = {}
            epoch_integrity = {}
            epoch_modalities = {}
            current_observations = observations_by_time_and_owner[
                float(timestamp)
            ].get(node_id, [])
            current_absolute_observations = (
                absolute_observations_by_time_and_node[float(timestamp)].get(
                    node_id, []
                )
            )
            current_modalities = {
                observation.modality for observation in current_observations
            }
            if current_absolute_observations:
                current_modalities.add("ABSOLUTE_POSITION")
            for modality in tuple(consecutive_anomalies[node_id]):
                if modality not in current_modalities:
                    consecutive_anomalies[node_id][modality] = 0
            for observation in current_absolute_observations:
                if consider_refresh_mode == "exact_transport_event_replay":
                    value = coordinators[
                        node_id
                    ].apply_delayed_absolute_observation(observation)
                    state = coordinators[node_id].state
                    if value is not None:
                        epoch_nis[observation.information_id] = value
                        epoch_integrity[observation.information_id] = (
                            coordinators[node_id].integrity_by_information_id[
                                observation.information_id
                            ]
                        )
                        epoch_modalities[observation.information_id] = (
                            "ABSOLUTE_POSITION"
                        )
                else:
                    update = multi_neighbor_schmidt_absolute_position_update(
                        state, observation,
                        nis_gate_threshold=(
                            nis_gate_threshold_by_modality or {}
                        ).get("ABSOLUTE_POSITION"),
                        nis_inflation_threshold=(
                            nis_inflation_threshold_by_modality or {}
                        ).get("ABSOLUTE_POSITION"),
                        maximum_measurement_covariance_scale=(
                            maximum_measurement_covariance_scale_by_modality or {}
                        ).get("ABSOLUTE_POSITION", 1.0),
                        integrity_policy=(
                            integrity_policy_by_modality or {}
                        ).get("ABSOLUTE_POSITION"),
                    )
                    state = update.state
                    epoch_nis[observation.information_id] = update.nis
                    epoch_integrity[observation.information_id] = update.integrity
                    epoch_modalities[observation.information_id] = (
                        "ABSOLUTE_POSITION"
                    )
            for observation in current_observations:
                if consider_refresh_mode == "exact_transport_event_replay":
                    value = coordinators[node_id].apply_delayed_observation(
                        observation
                    )
                    state = coordinators[node_id].state
                    if value is not None:
                        epoch_nis[observation.information_id] = value
                        epoch_integrity[observation.information_id] = (
                            coordinators[node_id].integrity_by_information_id[
                                observation.information_id
                            ]
                        )
                        epoch_modalities[observation.information_id] = (
                            observation.modality
                        )
                else:
                    update = multi_neighbor_schmidt_update(
                        state, observation,
                        nis_gate_threshold=(nis_gate_threshold_by_modality or {}).get(
                            observation.modality
                        ),
                        nis_inflation_threshold=(
                            nis_inflation_threshold_by_modality or {}
                        ).get(observation.modality),
                        maximum_measurement_covariance_scale=(
                            maximum_measurement_covariance_scale_by_modality or {}
                        ).get(observation.modality, 1.0),
                        integrity_policy=(
                            integrity_policy_by_modality or {}
                        ).get(observation.modality),
                    )
                    state = update.state
                    epoch_nis[observation.information_id] = update.nis
                    epoch_integrity[observation.information_id] = update.integrity
                    epoch_modalities[observation.information_id] = (
                        observation.modality
                    )
            for observation_id, integrity in tuple(epoch_integrity.items()):
                modality = epoch_modalities[observation_id]
                previous_count = consecutive_anomalies[node_id].get(modality, 0)
                count = previous_count + 1 if integrity.anomalous else 0
                consecutive_anomalies[node_id][modality] = count
                epoch_integrity[observation_id] = (
                    integrity.with_consecutive_anomaly_count(count)
                )
            local_states[node_id] = state
            active_states[node_id][index] = state.active_state
            active_covariances[node_id][index] = state.active_covariance
            joint_covariances[node_id][index] = state.joint_covariance
            for neighbor_id in state.neighbor_ids:
                active_cross_covariances[node_id][neighbor_id][index] = (
                    state.active_cross_covariance(neighbor_id)
                )
            nis_history[node_id].append(epoch_nis)
            integrity_history[node_id].append(epoch_integrity)
            modality_history[node_id].append(epoch_modalities)
        previous_active = {
            node_id: (local_states[node_id].active_state.copy(),
                      local_states[node_id].active_covariance)
            for node_id in node_ids
        }

    return NetworkSchmidtHistory(
        timestamps=times.copy(),
        active_state_history_by_node=active_states,
        active_covariance_history_by_node=active_covariances,
        active_cross_covariance_history_by_node=active_cross_covariances,
        joint_covariance_history_by_node=joint_covariances,
        nis_history_by_node=nis_history,
        local_dimension_by_node={
            node_id: local_states[node_id].dimension for node_id in node_ids
        },
        refresh_diagnostics=refresh_diagnostics,
        refresh_diagnostic_records=tuple(refresh_diagnostic_records),
        replay_performance_by_node={
            node_id: replace(coordinator.performance)
            for node_id, coordinator in coordinators.items()
        },
        integrity_history_by_node=integrity_history,
        modality_history_by_node=modality_history,
    )


def _network_module_output(
    history: NetworkSchmidtHistory, *, node_id: str, processing_time: float,
    link_timeout: float | None,
) -> NetworkModuleOutput:
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
    modality_valid_flags = {modality: False for modality in sorted(all_modalities)}
    for information_id, integrity in final_integrity.items():
        modality = final_modalities[information_id]
        modality_valid_flags[modality] = (
            modality_valid_flags[modality] or integrity.accepted
        )
    active_modalities = [
        modality for modality, valid in modality_valid_flags.items() if valid
    ]
    abnormal_events = []
    for index, timestamp in enumerate(history.timestamps):
        modalities = history.modality_history_by_node[node_id][index]
        for information_id, integrity in (
            history.integrity_history_by_node[node_id][index].items()
        ):
            if not integrity.anomalous:
                continue
            rejected = integrity.status == "HARD_REJECTED"
            abnormal_events.append(AbnormalEvent(
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
                    f"covariance_scale={integrity.measurement_covariance_scale}."
                ),
                node_id=node_id, target_id=node_id,
                modality=modalities[information_id],
            ))
    replay = history.replay_performance_by_node.get(
        node_id, ReplayPerformanceStats()
    )
    if replay.fallback_count:
        abnormal_events.append(AbnormalEvent(
            timestamp=float(history.timestamps[-1]),
            event_type="REPLAY_FALLBACK",
            severity="ERROR",
            description=f"Replay fallback count={replay.fallback_count}.",
            node_id=node_id, target_id=node_id,
        ))
    observation_count = sum(
        len(epoch) for epoch in history.integrity_history_by_node[node_id]
    )
    confidence = quality_score_from_covariance(final_covariance)
    status = (
        "DEGRADED" if replay.fallback_count
        or any(event.severity == "ERROR" for event in abnormal_events)
        else "OK" if active_modalities else "PREDICTION_ONLY"
    )
    module_output = ModuleOutput(
        state_output=StateOutput(
            timestamp=float(history.timestamps[-1]), target_id=node_id,
            position_estimate=final_state[:3].copy(),
            velocity_estimate=final_state[3:].copy(),
            acceleration_estimate=accel_two_body_j2(final_state[:3]),
            covariance=final_covariance.copy(), valid_flag=True,
            confidence_level=confidence,
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
            active_node_count=len(history.node_ids), status=status,
        ),
    )
    diagnostics = NetworkRuntimeDiagnostics(
        node_id=node_id,
        neighbor_count=history.local_dimension_by_node[node_id] // 6 - 1,
        replay_count=replay.replay_count,
        replay_batch_count=replay.batch_count,
        replay_fallback_count=replay.fallback_count,
        maximum_replay_seconds=replay.maximum_replay_seconds,
        maximum_retained_journal_count=replay.maximum_retained_journal_count,
        **_link_runtime_diagnostics(
            history, node_id=node_id, link_timeout=link_timeout,
        ),
        maximum_checkpoint_count=replay.maximum_checkpoint_count,
        maximum_pinned_checkpoint_count=(
            replay.maximum_pinned_checkpoint_count
        ),
        maximum_resync_required_count=(
            replay.maximum_resync_required_count
        ),
    )
    return NetworkModuleOutput(module_output, diagnostics)


def _link_runtime_diagnostics(
    history: NetworkSchmidtHistory, *, node_id: str,
    link_timeout: float | None,
) -> dict[str, object]:
    configured = tuple(sorted(
        history.active_cross_covariance_history_by_node[node_id]
    ))
    records = [
        record for record in history.refresh_diagnostic_records
        if record.get("receiver_id") == node_id
    ]
    last_receive = {neighbor: None for neighbor in configured}
    losses = {neighbor: 0 for neighbor in configured}
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
        else:
            reason = str(record.get("reason", "unknown"))
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        resync[neighbor] = bool(record.get("resync_required_count", 0))
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
        "resynchronization_required_by_neighbor": resync,
        "message_rejection_counts_by_reason": rejection_counts,
    }


def _route_observations(
    observations: Iterable[ObservationMessage],
    *,
    times: Array,
    topology: NetworkTopology,
    observation_usage: str,
    allow_delayed: bool = False,
) -> dict[float, dict[str, list[ObservationMessage]]]:
    result = {float(timestamp): {} for timestamp in times}
    seen_message_ids: set[str] = set()
    for observation in observations:
        source_timestamp = float(observation.timestamp)
        if source_timestamp not in result:
            raise ValueError("Observation timestamp is not in timestamps.")
        observer, target = str(observation.observer_id), str(observation.target_id)
        if target not in topology.neighbors(observer):
            raise ValueError("Observation endpoints must share a topology edge.")
        if observation.message_id in seen_message_ids:
            raise ValueError("Observation message_id values must be globally unique.")
        seen_message_ids.add(observation.message_id)
        owners = (
            (observer, target)
            if (
                observation_usage == "both_endpoints"
                and observation.metadata.get("shared_delivery", True)
            )
            else (observer,)
        )
        for owner in owners:
            route_timestamp = source_timestamp
            if owner != observer and observation.arrival_timestamp is not None:
                route_timestamp = float(observation.arrival_timestamp)
            if route_timestamp > float(times[-1]):
                continue
            if route_timestamp not in result:
                raise ValueError("Observation arrival timestamp is not in timestamps.")
            if not allow_delayed and route_timestamp > source_timestamp:
                raise ValueError(
                    "Delayed shared observations require exact event replay."
                )
            result[route_timestamp].setdefault(owner, []).append(observation)
    for per_owner in result.values():
        for owner, messages in per_owner.items():
            messages.sort(key=lambda item: item.information_id)
            unique = {}
            for message in messages:
                unique.setdefault(message.information_id, message)
            per_owner[owner] = list(unique.values())
    return result


def _route_absolute_observations(
    observations: Iterable[AbsolutePositionObservation], *, times: Array,
    node_ids: set[str], allow_delayed: bool,
) -> dict[float, dict[str, list[AbsolutePositionObservation]]]:
    result = {float(timestamp): {} for timestamp in times}
    seen_ids = set()
    for observation in observations:
        if not observation.valid_flag:
            continue
        node_id = str(observation.satellite_id)
        if node_id not in node_ids:
            raise ValueError("Absolute observation satellite is not a topology node.")
        source_timestamp = float(observation.timestamp)
        if source_timestamp not in result:
            raise ValueError("Absolute observation timestamp is not in timestamps.")
        if observation.information_id in seen_ids:
            raise ValueError("Absolute observation IDs must be globally unique.")
        seen_ids.add(observation.information_id)
        route_timestamp = (
            source_timestamp if observation.arrival_timestamp is None
            else float(observation.arrival_timestamp)
        )
        if route_timestamp > float(times[-1]):
            continue
        if route_timestamp not in result:
            raise ValueError(
                "Absolute observation arrival timestamp is not in timestamps."
            )
        if not allow_delayed and route_timestamp > source_timestamp:
            raise ValueError("Delayed absolute observations require exact event replay.")
        result[route_timestamp].setdefault(node_id, []).append(observation)
    for per_node in result.values():
        for values in per_node.values():
            values.sort(key=lambda item: item.information_id)
    return result


def _initial_values(
    values: Mapping[str, Array],
    node_ids: tuple[str, ...],
    shape: tuple[int, ...],
    name: str,
) -> dict[str, Array]:
    if set(values) != set(node_ids):
        raise ValueError(f"Initial {name} keys must match topology node IDs.")
    result = {}
    for node_id, value in values.items():
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            raise ValueError(f"Initial {name} for {node_id} must have shape {shape}.")
        result[node_id] = array.copy()
    return result


def _prepare_state_messages(
    messages_by_receiver: Mapping[str, Iterable[StateMessage]],
    node_ids: tuple[str, ...], topology: NetworkTopology,
) -> dict[str, list[StateMessage]]:
    unknown = set(messages_by_receiver) - set(node_ids)
    if unknown:
        raise ValueError("State-message receiver IDs must belong to the topology.")
    result = {node_id: [] for node_id in node_ids}
    for receiver_id, messages in messages_by_receiver.items():
        for message in messages:
            if str(message.source_node_id) not in topology.neighbors(receiver_id):
                raise ValueError("State-message source must be a receiver neighbor.")
            result[receiver_id].append(message)
        result[receiver_id].sort(key=lambda item: (
            float(item.timestamp if item.arrival_timestamp is None else item.arrival_timestamp),
            float(item.timestamp), str(item.source_node_id),
        ))
    return result
