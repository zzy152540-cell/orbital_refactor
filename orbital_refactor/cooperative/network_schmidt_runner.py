from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_batch_update,
    multi_neighbor_schmidt_absolute_position_update,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.runner_utils import validated_initial_values
from cooperative.network_schmidt_inputs import (
    prepare_state_messages as _prepare_state_messages,
    route_absolute_observations as _route_absolute_observations,
    route_relative_observations as _route_observations,
)
from cooperative.network_schmidt_outputs import (
    NetworkModuleOutput,
    NetworkRuntimeDiagnostics,
    link_runtime_diagnostics as _link_runtime_diagnostics,
    network_module_output as _network_module_output,
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
from cooperative.neighbor_measurement_quality import (
    NeighborLinkQuality,
    NeighborMeasurementQualityPolicy,
)
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from interfaces.data_objects import (
    AbsolutePositionObservation,
    ObservationMessage,
    StateMessage,
)
from orbital_core.measurement_integrity import (
    MeasurementIntegrityDiagnostics,
    MeasurementIntegrityPolicy,
)

Array = np.ndarray


@dataclass(frozen=True)
class NetworkSchmidtHistory:
    timestamps: Array
    active_state_history_by_node: dict[str, Array]
    active_covariance_history_by_node: dict[str, Array]
    active_cross_covariance_history_by_node: dict[str, dict[str, Array]]
    neighbor_state_history_by_node: dict[str, dict[str, Array]]
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
    joint_nis_history_by_node: dict[str, list[tuple[dict[str, object], ...]]]
    relative_update_history_by_node: dict[
        str, list[tuple[dict[str, object], ...]]
    ]

    @property
    def node_ids(self) -> list[str]:
        return list(self.active_state_history_by_node)

    def to_module_outputs(
        self, *, processing_time: float = 0.0,
        link_timeout: float | None = None,
        topology_version: int = 0,
        topology_transition_count: int = 0,
        active_neighbors_by_node: Mapping[
            str, tuple[str, ...]
        ] | None = None,
    ) -> dict[str, NetworkModuleOutput]:
        """Convert final per-node posteriors to the shared formal output schema."""

        return {
            node_id: _network_module_output(
                self, node_id=node_id, processing_time=processing_time,
                link_timeout=link_timeout,
                topology_version=topology_version,
                topology_transition_count=topology_transition_count,
                active_neighbors_by_node=active_neighbors_by_node,
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
    topology_version_by_timestamp: Mapping[float, int] | None = None,
    active_neighbors_by_timestamp: Mapping[
        float, Mapping[str, tuple[str, ...]]
    ] | None = None,
    relative_observation_order: tuple[str, ...] | None = None,
    relative_observation_order_start_time: float | None = None,
    batch_relative_observations: bool = False,
    batch_relative_observations_start_time: float | None = None,
    neighbor_linearization_state_by_node_and_time: Mapping[
        tuple[str, float], Array
    ] | None = None,
    neighbor_uncertainty_inflation_by_modality: Mapping[
        str, float
    ] | None = None,
    neighbor_measurement_quality_policy: (
        NeighborMeasurementQualityPolicy | None
    ) = None,
    neighbor_link_quality_by_node_and_time: Mapping[
        tuple[str, str, float], NeighborLinkQuality
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
    states = validated_initial_values(
        initial_state_by_node, node_ids, (6,), "state"
    )
    covariances = validated_initial_values(
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
                relative_observation_order=relative_observation_order,
                relative_observation_order_start_time=(
                    relative_observation_order_start_time
                ),
                batch_relative_observations=batch_relative_observations,
                batch_relative_observations_start_time=(
                    batch_relative_observations_start_time
                ),
                neighbor_linearization_state_by_node_and_time=(
                    neighbor_linearization_state_by_node_and_time
                ),
                neighbor_uncertainty_inflation_by_modality=(
                    neighbor_uncertainty_inflation_by_modality
                ),
                neighbor_measurement_quality_policy=(
                    neighbor_measurement_quality_policy
                ),
                neighbor_link_quality_by_node_and_time=(
                    neighbor_link_quality_by_node_and_time
                ),
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
        modality_update_order=relative_observation_order,
        modality_update_order_start_time=(
            relative_observation_order_start_time
        ),
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
    neighbor_states = {
        node_id: {
            neighbor_id: np.zeros((times.size, 6), dtype=float)
            for neighbor_id in topology.neighbors(node_id)
        }
        for node_id in node_ids
    }
    nis_history = {node_id: [] for node_id in node_ids}
    integrity_history = {node_id: [] for node_id in node_ids}
    modality_history = {node_id: [] for node_id in node_ids}
    joint_nis_history = {node_id: [] for node_id in node_ids}
    relative_update_history = {node_id: [] for node_id in node_ids}
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
                rejected = []
                current_version = int(
                    (topology_version_by_timestamp or {}).get(
                        float(timestamp), 0
                    )
                )
                current_active_neighbors = set(
                    (active_neighbors_by_timestamp or {}).get(
                        float(timestamp), {}
                    ).get(node_id, topology.neighbors(node_id))
                )
                for message in pending_state_messages[node_id]:
                    arrival = message.timestamp if message.arrival_timestamp is None else message.arrival_timestamp
                    if float(arrival) <= float(timestamp):
                        source_id = str(message.source_node_id)
                        message_version = int(
                            message.metadata.get("topology_version", 0)
                        )
                        if source_id not in current_active_neighbors:
                            rejected.append((message, "inactive_topology_link"))
                        elif message_version != current_version:
                            rejected.append((message, "topology_version_mismatch"))
                        else:
                            available.append((
                                message,
                                (expected_lineage_by_link or {}).get(
                                    (node_id, source_id)
                                ),
                            ))
                    else:
                        remaining.append(message)
                outcomes = coordinators[node_id].apply_state_messages(tuple(available))
                recorded = [
                    (message, outcome.accepted, outcome.reason)
                    for (message, _), outcome in zip(available, outcomes)
                ] + [
                    (message, False, reason) for message, reason in rejected
                ]
                for message, accepted, reason in recorded:
                    key = "accepted" if accepted else reason
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
                        "accepted": accepted,
                        "reason": reason,
                        "checkpoint_count": len(checkpoints),
                        "oldest_checkpoint": min(checkpoints) if checkpoints else None,
                        "newest_checkpoint": max(checkpoints) if checkpoints else None,
                        "pinned_checkpoint_count": coordinators[node_id].pinned_checkpoint_count,
                        "oldest_pinned_timestamp": coordinators[node_id].oldest_pinned_timestamp,
                        "resync_required_count": len(
                            coordinators[node_id].resynchronization_requirements
                        ),
                        "resync_required_neighbors": tuple(sorted({
                            neighbor
                            for neighbor, _ in coordinators[
                                node_id
                            ].resynchronization_requirements
                        })),
                        **message.metadata,
                        "expected_topology_version": current_version,
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
            epoch_joint_nis = []
            epoch_relative_updates = []
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
            observation_batches = _relative_observation_batches(
                current_observations,
                enabled=batch_relative_observations,
                start_time=batch_relative_observations_start_time,
            )
            for observation_batch in observation_batches:
                neighbor_id = _observation_counterpart(
                    node_id, observation_batch[0]
                )
                neighbor_linearization_state = (
                    neighbor_linearization_state_by_node_and_time or {}
                ).get((neighbor_id, float(timestamp)))
                dynamic_inflation_by_modality = {
                    item.modality: _neighbor_inflation(
                        modality=item.modality,
                        node_id=node_id,
                        neighbor_id=neighbor_id,
                        timestamp=float(timestamp),
                        fixed_by_modality=(
                            neighbor_uncertainty_inflation_by_modality
                        ),
                        policy=neighbor_measurement_quality_policy,
                        quality_by_node_and_time=(
                            neighbor_link_quality_by_node_and_time
                        ),
                    )
                    for item in observation_batch
                }
                if len(observation_batch) > 1:
                    if consider_refresh_mode == "exact_transport_event_replay":
                        update = coordinators[node_id].apply_observation_batch(
                            observation_batch,
                            neighbor_linearization_state=(
                                neighbor_linearization_state
                            ),
                        )
                        state = coordinators[node_id].state
                    else:
                        update = multi_neighbor_schmidt_batch_update(
                            state, observation_batch,
                            neighbor_linearization_state=(
                                neighbor_linearization_state
                            ),
                            neighbor_uncertainty_inflation_by_modality=(
                                dynamic_inflation_by_modality
                            ),
                        )
                        state = update.state
                    epoch_joint_nis.append({
                        "observer_id": observation_batch[0].observer_id,
                        "target_id": observation_batch[0].target_id,
                        "information_ids": tuple(
                            item.information_id for item in observation_batch
                        ),
                        "modalities": tuple(
                            item.modality for item in observation_batch
                        ),
                        "dimension": int(update.innovation.size),
                        "raw_nis": float(update.raw_nis),
                        "processed_nis": float(update.nis),
                        "measurement_covariance_scale": float(
                            update.measurement_covariance_scale
                        ),
                    })
                    epoch_relative_updates.append(
                        _relative_update_record(
                            update, observation_batch
                        )
                    )
                    continue
                observation = observation_batch[0]
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
                        update = coordinators[node_id].last_relative_update
                        if (
                            update is not None
                            and observation.information_id
                            in update.state.information_ids
                            and np.isclose(
                                observation.timestamp, timestamp
                            )
                        ):
                            epoch_relative_updates.append(
                                _relative_update_record(
                                    update, (observation,)
                                )
                            )
                else:
                    update = multi_neighbor_schmidt_update(
                        state, observation,
                        neighbor_linearization_state=(
                            neighbor_linearization_state
                        ),
                        neighbor_uncertainty_inflation=(
                            dynamic_inflation_by_modality[
                                observation.modality
                            ]
                        ),
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
                    epoch_relative_updates.append(
                        _relative_update_record(update, (observation,))
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
                neighbor_states[node_id][neighbor_id][index] = (
                    state.neighbor_state_by_id[neighbor_id]
                )
                active_cross_covariances[node_id][neighbor_id][index] = (
                    state.active_cross_covariance(neighbor_id)
                )
            nis_history[node_id].append(epoch_nis)
            integrity_history[node_id].append(epoch_integrity)
            modality_history[node_id].append(epoch_modalities)
            joint_nis_history[node_id].append(tuple(epoch_joint_nis))
            relative_update_history[node_id].append(
                tuple(epoch_relative_updates)
            )
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
        neighbor_state_history_by_node=neighbor_states,
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
        joint_nis_history_by_node=joint_nis_history,
        relative_update_history_by_node=relative_update_history,
    )


def _relative_update_record(update, observations):
    return {
        "observer_id": observations[0].observer_id,
        "target_id": observations[0].target_id,
        "information_ids": tuple(
            item.information_id for item in observations
        ),
        "modalities": tuple(item.modality for item in observations),
        "prior_active_state": update.prior_active_state.copy(),
        "prior_neighbor_state": update.prior_neighbor_state.copy(),
        "active_jacobian": update.active_jacobian.copy(),
        "neighbor_jacobian": update.neighbor_jacobian.copy(),
        "innovation": update.innovation.copy(),
        "innovation_covariance": update.innovation_covariance.copy(),
        "active_correction": update.active_correction.copy(),
        "projected_neighbor_covariance": (
            update.projected_neighbor_covariance.copy()
        ),
        "nominal_measurement_covariance": (
            update.nominal_measurement_covariance.copy()
        ),
        "active_gain": update.active_gain.copy(),
        "prior_active_covariance": update.prior_active_covariance.copy(),
        "neighbor_uncertainty_inflation": float(
            update.neighbor_uncertainty_inflation
        ),
        "skipped": bool(update.skipped),
    }


def _observation_counterpart(node_id, observation):
    return (
        str(observation.target_id)
        if str(observation.observer_id) == str(node_id)
        else str(observation.observer_id)
    )


def _neighbor_inflation(
    *, modality, node_id, neighbor_id, timestamp, fixed_by_modality,
    policy, quality_by_node_and_time,
):
    fixed = float((fixed_by_modality or {}).get(modality, 0.0))
    if policy is None:
        return fixed
    quality = (quality_by_node_and_time or {}).get(
        (node_id, neighbor_id, timestamp), NeighborLinkQuality()
    )
    dynamic = policy.inflation(
        modality=modality,
        age=quality.age,
        consecutive_losses=quality.consecutive_losses,
        resynchronization_required=quality.resynchronization_required,
    )
    return max(fixed, dynamic)


def _relative_observation_batches(observations, *, enabled, start_time):
    if not enabled:
        return tuple((item,) for item in observations)
    groups = []
    current_key = None
    current = []
    for observation in observations:
        use_batch = (
            start_time is None or float(observation.timestamp) > float(start_time)
        )
        key = (
            float(observation.timestamp),
            str(observation.observer_id),
            str(observation.target_id),
        ) if use_batch else (observation.information_id,)
        if current and key != current_key:
            groups.append(tuple(current))
            current = []
        current_key = key
        current.append(observation)
    if current:
        groups.append(tuple(current))
    return tuple(groups)
