from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.exact_transport_protocol import build_exact_transport_state_message
from interfaces.data_objects import CovarianceTransportEvent, StateMessage

Array = np.ndarray


@dataclass
class ExactTransportAccumulator:
    """Accumulate error transforms from the last receiver-acknowledged baseline."""

    source_node_id: str
    lineage_id: str
    reference_timestamp: float
    reference_state: Array
    reference_covariance: Array
    current_timestamp: float | None = None
    current_state: Array | None = None
    information_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.reference_state = np.asarray(self.reference_state, dtype=float).reshape(6).copy()
        self.reference_covariance = np.asarray(self.reference_covariance, dtype=float).reshape(6, 6).copy()
        self.current_timestamp = float(self.reference_timestamp)
        self.current_state = self.reference_state.copy()
        self._transition = np.eye(6)
        self._noise = np.zeros((6, 6))
        self._events: list[CovarianceTransportEvent] = []
        self._steps: list[tuple[float, Array, Array, Array, tuple[str, ...]]] = []

    def append(
        self, *, timestamp: float, updated_state: Array,
        error_transition: Array, independent_process_noise: Array,
        information_ids: tuple[str, ...] = (),
        event_error_transition: Array | None = None,
        event_process_noise: Array | None = None,
    ) -> None:
        transition = np.asarray(error_transition, dtype=float).reshape(6, 6)
        noise = np.asarray(independent_process_noise, dtype=float).reshape(6, 6)
        self._noise = transition @ self._noise @ transition.T + noise
        self._transition = transition @ self._transition
        self.current_timestamp = float(timestamp)
        self.current_state = np.asarray(updated_state, dtype=float).reshape(6).copy()
        self.information_ids = (*self.information_ids, *(str(value) for value in information_ids))
        self._events.append(CovarianceTransportEvent(
            timestamp=float(timestamp), state_estimate=self.current_state.copy(),
            error_transition=(transition.copy() if event_error_transition is None else
                              np.asarray(event_error_transition, dtype=float).reshape(6, 6).copy()),
            independent_process_noise=(noise.copy() if event_process_noise is None else
                                       np.asarray(event_process_noise, dtype=float).reshape(6, 6).copy()),
            information_ids=tuple(str(value) for value in information_ids),
            event_id=f"{self.lineage_id}:transport:{float(timestamp):.12g}",
            source_error_transition=transition.copy(),
            source_process_noise=noise.copy(),
        ))
        self._steps.append((
            float(timestamp), transition.copy(), noise.copy(),
            self.current_state.copy(), tuple(str(value) for value in information_ids),
        ))

    def build_message(self) -> StateMessage:
        return build_exact_transport_state_message(
            source_node_id=self.source_node_id,
            timestamp=float(self.current_timestamp),
            reference_timestamp=float(self.reference_timestamp),
            reference_state=self.reference_state,
            reference_covariance=self.reference_covariance,
            updated_state=self.current_state,
            error_transition=self._transition,
            independent_process_noise=self._noise,
            lineage_id=self.lineage_id,
            information_ids=self.information_ids,
            transport_events=tuple(self._events),
        )

    def acknowledge(self, message: StateMessage) -> None:
        if message.lineage_id != self.lineage_id:
            raise ValueError("Cannot acknowledge a different lineage.")
        matching = [index for index, step in enumerate(self._steps)
                    if np.isclose(step[0], float(message.timestamp))]
        if not matching:
            raise ValueError("Acknowledgement does not match an unconfirmed message version.")
        acknowledged_index = matching[-1]
        self.reference_timestamp = float(message.timestamp)
        self.reference_state = np.asarray(message.state_estimate, dtype=float).reshape(6).copy()
        self.reference_covariance = np.asarray(message.covariance, dtype=float).reshape(6, 6).copy()
        self._steps = self._steps[acknowledged_index + 1:]
        self._events = self._events[acknowledged_index + 1:]
        self.information_ids = tuple(
            information_id for step in self._steps for information_id in step[4]
        )
        self._transition = np.eye(6)
        self._noise = np.zeros((6, 6))
        for _, transition, noise, _, _ in self._steps:
            self._noise = transition @ self._noise @ transition.T + noise
            self._transition = transition @ self._transition

    @property
    def accumulated_transition(self) -> Array:
        return self._transition.copy()

    @property
    def accumulated_process_noise(self) -> Array:
        return self._noise.copy()
