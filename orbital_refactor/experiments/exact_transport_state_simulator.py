from __future__ import annotations

from dataclasses import replace

import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.message_transport import MessageChannel
from experiments.scenario_controls import link_is_in_outage, topology_edge_is_inactive
from interfaces.data_objects import AbsolutePositionObservation, StateMessage
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)


class ExactTransportStateSimulator:
    """Stateful sender-side navigation and covariance-transport simulation."""

    def __init__(
        self,
        *,
        initial_state_by_node,
        initial_covariance_by_node,
        topology,
        edges,
        packet_loss,
        delay,
        random_seed,
        acknowledge_messages,
        dt,
        process_noise_acceleration,
        absolute_sigma,
        absolute_navigation_dropout_windows,
        node_dropout_windows,
        communication_outages,
        topology_inactive_windows,
        topology_versions,
    ):
        self.topology = topology
        self.edges = tuple(edges)
        self.acknowledge_messages = bool(acknowledge_messages)
        self.dt = float(dt)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.absolute_sigma = float(absolute_sigma)
        self.absolute_navigation_dropout_windows = tuple(
            absolute_navigation_dropout_windows
        )
        self.node_dropout_windows = node_dropout_windows
        self.communication_outages = communication_outages
        self.topology_inactive_windows = topology_inactive_windows
        self.topology_versions = topology_versions
        self.sender_state = {
            node: value.copy() for node, value in initial_state_by_node.items()
        }
        self.sender_covariance = {
            node: value.copy()
            for node, value in initial_covariance_by_node.items()
        }
        self.accumulators = {
            edge: ExactTransportAccumulator(
                source_node_id=edge[1],
                lineage_id=f"{edge[1]}->{edge[0]}:0",
                reference_timestamp=0.0,
                reference_state=self.sender_state[edge[1]],
                reference_covariance=self.sender_covariance[edge[1]],
            )
            for edge in self.edges
        }
        self.channels = {
            edge: MessageChannel(
                packet_loss_rate={edge[1]: packet_loss},
                delay_by_source={edge[1]: delay},
                random_seed=random_seed * 101 + index,
            )
            for index, edge in enumerate(self.edges)
        }
        self.state_messages = {node: [] for node in topology.node_ids}
        self.transmitted_messages: list[StateMessage] = []
        self.pending_acks = []
        self.consecutive_losses = {edge: 0 for edge in self.edges}
        self.link_sequence = {edge: 0 for edge in self.edges}
        self.position_jacobian = np.zeros((3, 6))
        self.position_jacobian[:, :3] = np.eye(3)
        self.absolute_covariance = np.eye(3) * self.absolute_sigma**2

    @property
    def lineages(self):
        return {
            (receiver, source): f"{source}->{receiver}:0"
            for receiver, source in self.edges
        }

    def advance_epoch(self, *, index, timestamp, truth, rng):
        """Advance sender filters and route state messages for one epoch."""

        timestamp = float(timestamp)
        self._process_acknowledgements(timestamp)
        prediction_transition, prediction_noise = self._predict(index)
        absolute_observations = []
        for node in self.topology.node_ids:
            update_transition, update_noise, information_ids = (
                self._absolute_navigation_update(
                    node=node,
                    index=index,
                    timestamp=timestamp,
                    truth=truth,
                    rng=rng,
                    observations=absolute_observations,
                )
            )
            self._transport_source_update(
                node=node,
                timestamp=timestamp,
                prediction_transition=prediction_transition[node],
                prediction_noise=prediction_noise[node],
                update_transition=update_transition,
                update_noise=update_noise,
                information_ids=information_ids,
            )
        return absolute_observations

    def _process_acknowledgements(self, timestamp):
        for arrival, edge, message in sorted(
            self.pending_acks, key=lambda item: item[0]
        ):
            if arrival <= timestamp and self.acknowledge_messages:
                self.accumulators[edge].acknowledge(message)
        self.pending_acks = [
            item for item in self.pending_acks
            if item[0] > timestamp or not self.acknowledge_messages
        ]

    def _predict(self, index):
        transitions = {
            node: np.eye(6) for node in self.topology.node_ids
        }
        noises = {
            node: np.zeros((6, 6)) for node in self.topology.node_ids
        }
        if index == 0:
            return transitions, noises
        for node in self.topology.node_ids:
            transition = numerical_jacobian_discrete(
                lambda value: rk4_step_absolute(value, self.dt),
                self.sender_state[node],
            )
            self.sender_state[node] = rk4_step_absolute(
                self.sender_state[node], self.dt
            )
            noise = make_process_noise(
                self.dt, self.process_noise_acceleration
            )
            self.sender_covariance[node] = (
                transition @ self.sender_covariance[node] @ transition.T + noise
            )
            transitions[node] = transition
            noises[node] = noise
        return transitions, noises

    def _absolute_navigation_update(
        self, *, node, index, timestamp, truth, rng, observations,
    ):
        navigation_available = not any(
            start <= timestamp <= end
            for start, end in (
                self.absolute_navigation_dropout_windows
                + self.node_dropout_windows.get(node, ())
            )
        )
        update_transition = np.eye(6)
        update_noise = np.zeros((6, 6))
        information_ids = ()
        if not navigation_available:
            return update_transition, update_noise, information_ids
        innovation_covariance = (
            self.position_jacobian
            @ self.sender_covariance[node]
            @ self.position_jacobian.T
            + self.absolute_covariance
        )
        gain = (
            self.sender_covariance[node]
            @ self.position_jacobian.T
            @ np.linalg.inv(innovation_covariance)
        )
        update_transition = np.eye(6) - gain @ self.position_jacobian
        update_noise = gain @ self.absolute_covariance @ gain.T
        measurement = (
            truth[node][index, :3]
            + rng.normal(0.0, self.absolute_sigma, 3)
        )
        information_id = f"{node}:absolute:{index}"
        information_ids = (information_id,)
        observations.append(AbsolutePositionObservation(
            timestamp=timestamp,
            satellite_id=node,
            measurement_eci=measurement.copy(),
            covariance=self.absolute_covariance.copy(),
            confidence=1.0,
            valid_flag=True,
            observation_id=information_id,
        ))
        self.sender_state[node] += gain @ (
            measurement - self.position_jacobian @ self.sender_state[node]
        )
        self.sender_covariance[node] = (
            update_transition
            @ self.sender_covariance[node]
            @ update_transition.T
            + update_noise
        )
        return update_transition, update_noise, information_ids

    def _transport_source_update(
        self,
        *,
        node,
        timestamp,
        prediction_transition,
        prediction_noise,
        update_transition,
        update_noise,
        information_ids,
    ):
        for receiver, source in self.edges:
            if source != node:
                continue
            edge = (receiver, source)
            combined_transition = update_transition @ prediction_transition
            combined_noise = (
                update_transition @ prediction_noise @ update_transition.T
                + update_noise
            )
            accumulator = self.accumulators[edge]
            accumulator.append(
                timestamp=timestamp,
                updated_state=self.sender_state[node],
                error_transition=combined_transition,
                independent_process_noise=combined_noise,
                information_ids=information_ids,
                event_error_transition=update_transition,
                event_process_noise=update_noise,
            )
            message = accumulator.build_message()
            self.link_sequence[edge] += 1
            transmitted = self._transmit(edge, timestamp, message)
            if transmitted is None:
                self.consecutive_losses[edge] += 1
                continue
            ack_eligible = self.acknowledge_messages
            transmitted = replace(
                transmitted,
                metadata={
                    "link_sequence": self.link_sequence[edge],
                    "consecutive_losses_before_delivery": (
                        self.consecutive_losses[edge]
                    ),
                    "reference_event_count": len(message.transport_events),
                    "ack_eligible": ack_eligible,
                    "topology_version": self.topology_versions[timestamp],
                },
            )
            self.consecutive_losses[edge] = 0
            self.state_messages[receiver].append(transmitted)
            self.transmitted_messages.append(transmitted)
            if ack_eligible:
                self.pending_acks.append((
                    float(transmitted.arrival_timestamp), edge, message
                ))

    def _transmit(self, edge, timestamp, message):
        receiver, source = edge
        if topology_edge_is_inactive(
            self.topology_inactive_windows,
            first=receiver,
            second=source,
            timestamp=timestamp,
        ) or link_is_in_outage(
            self.communication_outages,
            receiver=receiver,
            source=source,
            timestamp=timestamp,
        ):
            return None
        return self.channels[edge].transmit(message)
