from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from cooperative.link_lifecycle import LinkLifecycle, LinkLifecycleState
from cooperative.multi_neighbor_replay_coordinator import (
    CoordinatorMessageResult,
    MultiNeighborReplayCoordinator,
    ResynchronizationBaseline,
)
from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState
from interfaces.data_objects import (
    AbsolutePositionObservation,
    ObservationMessage,
    StateMessage,
)
from orbital_core.measurement_integrity import MeasurementIntegrityPolicy


@dataclass(frozen=True)
class NetworkSchmidtStepResult:
    timestamp: float
    state: MultiNeighborSchmidtState
    message_results: tuple[CoordinatorMessageResult, ...]
    nis_by_information_id: dict[str, float]


class NetworkSchmidtSession:
    """Persistent one-receiver Schmidt/replay session for online orchestration."""

    def __init__(
        self, initial_state: MultiNeighborSchmidtState, *,
        lineage_by_neighbor: Mapping[str, str],
        process_noise_acceleration: float = 1e-4,
        history_window: float | None = None,
        max_pinned_age: float | None = None,
        max_retained_events: int | None = None,
        integrity_policy_by_modality: Mapping[
            str, MeasurementIntegrityPolicy
        ] | None = None,
    ) -> None:
        if set(lineage_by_neighbor) != set(initial_state.neighbor_ids):
            raise ValueError("Lineage keys must match consider neighbors.")
        self.node_id = str(initial_state.active_node_id)
        self.coordinator = MultiNeighborReplayCoordinator(
            initial_state,
            process_noise_acceleration=process_noise_acceleration,
            history_window=history_window,
            max_pinned_age=max_pinned_age,
            max_retained_events=max_retained_events,
            integrity_policy_by_modality=integrity_policy_by_modality,
        )
        self.link_by_neighbor = {
            str(neighbor): LinkLifecycle(
                receiver_id=self.node_id, source_id=str(neighbor),
                lineage_id=str(lineage),
            )
            for neighbor, lineage in lineage_by_neighbor.items()
        }

    @property
    def state(self) -> MultiNeighborSchmidtState:
        return self.coordinator.state

    def suspend_link(self, neighbor_id: str, *, topology_version: int) -> None:
        neighbor_id = str(neighbor_id)
        self.link_by_neighbor[neighbor_id] = self.link_by_neighbor[
            neighbor_id
        ].suspend(topology_version=topology_version)

    def resume_link(
        self, neighbor_id: str, *, topology_version: int,
        history_available: bool,
    ) -> None:
        neighbor_id = str(neighbor_id)
        self.link_by_neighbor[neighbor_id] = self.link_by_neighbor[
            neighbor_id
        ].resume(
            topology_version=topology_version,
            history_available=history_available,
        )

    def establish_resynchronized_link(
        self, neighbor_id: str, *, lineage_id: str,
    ) -> ResynchronizationBaseline:
        neighbor_id = str(neighbor_id)
        lifecycle = self.link_by_neighbor[neighbor_id]
        baseline = self.coordinator.establish_resynchronized_link(
            neighbor_id=neighbor_id, lineage_id=lineage_id,
        )
        self.link_by_neighbor[neighbor_id] = (
            lifecycle.establish_resynchronized_lineage(
                lineage_id=lineage_id
            )
        )
        return baseline

    def step(
        self, timestamp: float, *,
        state_messages: Iterable[StateMessage] = (),
        observations: Iterable[ObservationMessage] = (),
        absolute_observations: Iterable[AbsolutePositionObservation] = (),
    ) -> NetworkSchmidtStepResult:
        timestamp = float(timestamp)
        if timestamp < float(self.state.timestamp):
            raise ValueError("Session steps must be chronological.")
        if timestamp > float(self.state.timestamp):
            self.coordinator.advance(timestamp)
        self._synchronize_resource_requirements()

        message_results = []
        for message in state_messages:
            source = str(message.source_node_id)
            lifecycle = self.link_by_neighbor[source]
            version = int(message.metadata.get(
                "topology_version", lifecycle.topology_version
            ))
            if not lifecycle.accepts(
                lineage_id=str(message.lineage_id),
                topology_version=version,
            ):
                reason = (
                    "resync_required"
                    if lifecycle.state == LinkLifecycleState.RESYNC_REQUIRED
                    else "inactive_topology_link"
                    if lifecycle.state == LinkLifecycleState.SUSPENDED
                    else "topology_or_lineage_mismatch"
                )
                message_results.append(
                    CoordinatorMessageResult(False, reason)
                )
                continue
            message_results.append(self.coordinator.apply_state_message(
                message, expected_lineage_id=lifecycle.lineage_id,
            ))
            self._synchronize_resource_requirements()

        nis_by_information_id = {}
        for observation in absolute_observations:
            value = self.coordinator.apply_delayed_absolute_observation(
                observation
            )
            if value is not None:
                nis_by_information_id[observation.information_id] = value
        for observation in observations:
            value = self.coordinator.apply_delayed_observation(observation)
            if value is not None:
                nis_by_information_id[observation.information_id] = value
        self._synchronize_resource_requirements()
        return NetworkSchmidtStepResult(
            timestamp=timestamp, state=self.state,
            message_results=tuple(message_results),
            nis_by_information_id=nis_by_information_id,
        )

    def _synchronize_resource_requirements(self) -> None:
        for (neighbor, lineage), reason in (
            self.coordinator.resynchronization_requirements.items()
        ):
            lifecycle = self.link_by_neighbor.get(neighbor)
            if lifecycle is None or lifecycle.lineage_id != lineage:
                continue
            self.link_by_neighbor[neighbor] = (
                lifecycle.require_resynchronization(reason=reason)
            )
