from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.link_lifecycle import LinkLifecycleState
from cooperative.message_transport import MessageChannel
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt
from cooperative.network_schmidt_session import (
    NetworkSchmidtSession,
    NetworkSchmidtStepResult,
)
from cooperative.topology import NetworkTopology
from interfaces.data_objects import AbsolutePositionObservation, ObservationMessage
from orbital_core.measurement_integrity import MeasurementIntegrityPolicy

Array = np.ndarray


@dataclass(frozen=True)
class TransportSourceUpdate:
    state: Array
    error_transition: Array
    independent_process_noise: Array
    information_ids: tuple[str, ...] = ()
    event_error_transition: Array | None = None
    event_process_noise: Array | None = None


@dataclass(frozen=True)
class NetworkOrchestratorStepResult:
    timestamp: float
    result_by_node: dict[str, NetworkSchmidtStepResult]
    accepted_message_count: int
    rejected_message_count: int
    rejection_counts_by_reason: dict[str, int]
    resynchronized_links: tuple[tuple[str, str, str], ...]
    transmitted_message_count: int
    dropped_message_count: int
    stale_topology_message_count: int
    protocol_rejected_message_count: int


class NetworkSchmidtOrchestrator:
    """Online multi-node orchestration for exact-transport Schmidt sessions."""

    def __init__(
        self, *, initial_state_by_node: Mapping[str, Array],
        initial_covariance_by_node: Mapping[str, Array],
        topology: NetworkTopology, initial_timestamp: float = 0.0,
        process_noise_acceleration: float = 1e-4,
        history_window: float | None = None,
        max_pinned_age: float | None = None,
        max_retained_events: int | None = None,
        packet_loss_rate: float = 0.0,
        communication_delay: float = 0.0,
        random_seed: int = 0,
        stop_and_wait: bool = True,
        integrity_policy_by_modality: Mapping[
            str, MeasurementIntegrityPolicy
        ] | None = None,
    ) -> None:
        self.topology = topology
        self.topology_version = 0
        self.active_neighbors_by_node = {
            node: tuple(topology.neighbors(node)) for node in topology.node_ids
        }
        self.sessions = {}
        self.accumulators = {}
        self.channels = {}
        self.pending_deliveries = {
            node: [] for node in topology.node_ids
        }
        self.stop_and_wait = bool(stop_and_wait)
        for receiver in topology.node_ids:
            neighbor_states = {
                neighbor: np.asarray(
                    initial_state_by_node[neighbor], dtype=float
                ).reshape(6)
                for neighbor in topology.neighbors(receiver)
            }
            neighbor_covariances = {
                neighbor: np.asarray(
                    initial_covariance_by_node[neighbor], dtype=float
                ).reshape(6, 6)
                for neighbor in topology.neighbors(receiver)
            }
            lineage_by_neighbor = {
                neighbor: f"{neighbor}->{receiver}:0"
                for neighbor in topology.neighbors(receiver)
            }
            local = initialize_multi_neighbor_schmidt(
                timestamp=float(initial_timestamp), active_node_id=receiver,
                active_state=initial_state_by_node[receiver],
                active_covariance=initial_covariance_by_node[receiver],
                neighbor_state_by_id=neighbor_states,
                neighbor_covariance_by_id=neighbor_covariances,
            )
            self.sessions[receiver] = NetworkSchmidtSession(
                local, lineage_by_neighbor=lineage_by_neighbor,
                process_noise_acceleration=process_noise_acceleration,
                history_window=history_window,
                max_pinned_age=max_pinned_age,
                max_retained_events=max_retained_events,
                integrity_policy_by_modality=integrity_policy_by_modality,
            )
            for source, lineage in lineage_by_neighbor.items():
                self.accumulators[(receiver, source)] = (
                    ExactTransportAccumulator(
                        source_node_id=source, lineage_id=lineage,
                        reference_timestamp=float(initial_timestamp),
                        reference_state=neighbor_states[source],
                        reference_covariance=neighbor_covariances[source],
                    )
                )
                edge = (receiver, source)
                self.channels[edge] = MessageChannel(
                    packet_loss_rate={source: float(packet_loss_rate)},
                    delay_by_source={source: float(communication_delay)},
                    random_seed=int(random_seed) + len(self.channels),
                )

    def step(
        self, timestamp: float, *, topology_version: int,
        active_neighbors_by_node: Mapping[str, tuple[str, ...]],
        source_update_by_node: Mapping[str, TransportSourceUpdate],
        observations: Iterable[ObservationMessage] = (),
        absolute_observations: Iterable[AbsolutePositionObservation] = (),
    ) -> NetworkOrchestratorStepResult:
        timestamp = float(timestamp)
        for session in self.sessions.values():
            session.step(timestamp)
        resynchronized = self._apply_topology(
            topology_version=int(topology_version),
            active_neighbors_by_node=active_neighbors_by_node,
        )
        resynchronized_edges = {
            (receiver, source)
            for receiver, source, _ in resynchronized
        }
        for (receiver, source), accumulator in self.accumulators.items():
            update = source_update_by_node[source]
            just_resynchronized = (receiver, source) in resynchronized_edges
            transition = (
                update.event_error_transition
                if just_resynchronized
                and update.event_error_transition is not None
                else update.error_transition
            )
            noise = (
                update.event_process_noise
                if just_resynchronized
                and update.event_process_noise is not None
                else update.independent_process_noise
            )
            accumulator.append(
                timestamp=timestamp, updated_state=update.state,
                error_transition=transition,
                independent_process_noise=noise,
                information_ids=update.information_ids,
                event_error_transition=update.event_error_transition,
                event_process_noise=update.event_process_noise,
            )

        transmitted_count = dropped_count = 0
        for (receiver, source), accumulator in self.accumulators.items():
            lifecycle = self.sessions[receiver].link_by_neighbor[source]
            if lifecycle.state != LinkLifecycleState.ACTIVE:
                continue
            if self.stop_and_wait and any(
                str(message.source_node_id) == source
                for message, _ in self.pending_deliveries[receiver]
            ):
                continue
            message = accumulator.build_message()
            message.metadata = {
                "topology_version": lifecycle.topology_version
            }
            transmitted = self.channels[(receiver, source)].transmit(message)
            if transmitted is None:
                dropped_count += 1
                continue
            transmitted_count += 1
            self.pending_deliveries[receiver].append((
                transmitted, accumulator
            ))

        deliveries_by_receiver = {
            node: [] for node in self.topology.node_ids
        }
        for receiver, pending in self.pending_deliveries.items():
            remaining = []
            for message, accumulator in pending:
                arrival = (
                    message.timestamp if message.arrival_timestamp is None
                    else message.arrival_timestamp
                )
                if float(arrival) <= timestamp:
                    deliveries_by_receiver[receiver].append((
                        message, accumulator
                    ))
                else:
                    remaining.append((message, accumulator))
            self.pending_deliveries[receiver] = remaining

        observations_by_node = {node: [] for node in self.topology.node_ids}
        for observation in observations:
            observer = str(observation.observer_id)
            target = str(observation.target_id)
            if target in set(active_neighbors_by_node[observer]):
                observations_by_node[observer].append(observation)
        absolute_by_node = {node: [] for node in self.topology.node_ids}
        for observation in absolute_observations:
            absolute_by_node[str(observation.satellite_id)].append(observation)

        results = {}
        accepted = rejected = 0
        stale_topology = protocol_rejected = 0
        rejection_counts = {}
        for receiver, session in self.sessions.items():
            deliveries = deliveries_by_receiver[receiver]
            messages = tuple(message for message, _ in deliveries)
            result = session.step(
                timestamp, state_messages=messages,
                observations=observations_by_node[receiver],
                absolute_observations=absolute_by_node[receiver],
            )
            results[receiver] = result
            for (message, accumulator), outcome in zip(
                deliveries, result.message_results
            ):
                if outcome.accepted:
                    accepted += 1
                    accumulator.acknowledge(message)
                else:
                    rejected += 1
                    if outcome.reason == "inactive_topology_link":
                        stale_topology += 1
                    else:
                        protocol_rejected += 1
                    rejection_counts[outcome.reason] = (
                        rejection_counts.get(outcome.reason, 0) + 1
                    )
        self.topology_version = int(topology_version)
        self.active_neighbors_by_node = {
            node: tuple(values)
            for node, values in active_neighbors_by_node.items()
        }
        return NetworkOrchestratorStepResult(
            timestamp=timestamp, result_by_node=results,
            accepted_message_count=accepted,
            rejected_message_count=rejected,
            rejection_counts_by_reason=rejection_counts,
            resynchronized_links=tuple(resynchronized),
            transmitted_message_count=transmitted_count,
            dropped_message_count=dropped_count,
            stale_topology_message_count=stale_topology,
            protocol_rejected_message_count=protocol_rejected,
        )

    def _apply_topology(
        self, *, topology_version: int,
        active_neighbors_by_node: Mapping[str, tuple[str, ...]],
    ) -> list[tuple[str, str, str]]:
        if topology_version < self.topology_version:
            raise ValueError("Topology versions must be monotonic.")
        resynchronized = []
        for receiver, session in self.sessions.items():
            previous = set(self.active_neighbors_by_node[receiver])
            current = set(active_neighbors_by_node[receiver])
            for source in self.topology.neighbors(receiver):
                lifecycle = session.link_by_neighbor[source]
                if source in previous and source not in current:
                    session.suspend_link(
                        source, topology_version=topology_version
                    )
                elif source not in previous and source in current:
                    requirement = any(
                        neighbor == source
                        for neighbor, _ in session.coordinator.
                        resynchronization_requirements
                    )
                    if lifecycle.state == LinkLifecycleState.RESYNC_REQUIRED:
                        session.link_by_neighbor[source] = (
                            lifecycle.observe_topology_version(
                                topology_version
                            )
                        )
                    else:
                        session.resume_link(
                            source, topology_version=topology_version,
                            history_available=not requirement,
                        )
                    if session.link_by_neighbor[
                        source
                    ].state == LinkLifecycleState.RESYNC_REQUIRED:
                        new_lineage = (
                            f"{source}->{receiver}:resync:"
                            f"{session.link_by_neighbor[source].resynchronization_count + 1}"
                        )
                        baseline = session.establish_resynchronized_link(
                            source, lineage_id=new_lineage
                        )
                        self.accumulators[(receiver, source)] = (
                            ExactTransportAccumulator(
                                source_node_id=source,
                                lineage_id=baseline.lineage_id,
                                reference_timestamp=baseline.timestamp,
                                reference_state=baseline.state_estimate,
                                reference_covariance=baseline.covariance,
                            )
                        )
                        resynchronized.append((
                            receiver, source, new_lineage
                        ))
        return resynchronized
