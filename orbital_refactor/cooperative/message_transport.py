from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

import numpy as np

from interfaces.data_objects import ObservationMessage, StateMessage

Message = StateMessage | ObservationMessage
MessageT = TypeVar("MessageT", StateMessage, ObservationMessage)


@dataclass
class MessageChannel:
    """Packet-loss and delay channel for one V14 message class."""

    packet_loss_rate: dict[str, float] = field(default_factory=dict)
    delay_by_source: dict[str, float] = field(default_factory=dict)
    random_seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        for source, loss in self.packet_loss_rate.items():
            if not 0.0 <= float(loss) <= 1.0:
                raise ValueError(f"packet loss for {source} must be in [0, 1].")
        for source, delay in self.delay_by_source.items():
            if float(delay) < 0.0:
                raise ValueError(f"delay for {source} cannot be negative.")

    def transmit(self, message: MessageT) -> MessageT | None:
        source = message_source_id(message)
        loss = float(self.packet_loss_rate.get(source, 0.0))
        if self._rng.random() < loss:
            return None
        delay = float(self.delay_by_source.get(source, 0.0))
        source_timestamp = (
            float(message.timestamp)
            if message.source_timestamp is None
            else float(message.source_timestamp)
        )
        return replace(
            message,
            source_timestamp=source_timestamp,
            arrival_timestamp=float(message.timestamp) + delay,
        )


@dataclass
class TypedMessageBuffer(Generic[MessageT]):
    """Arrival-time buffer with stable ordering and duplicate suppression."""

    messages: list[MessageT] = field(default_factory=list)
    _keys: set[tuple[object, ...]] = field(default_factory=set)

    def push(self, message: MessageT) -> bool:
        key = message_identity(message)
        if key in self._keys:
            return False
        self.messages.append(message)
        self._keys.add(key)
        return True

    def pop_available(self, current_time: float) -> list[MessageT]:
        available: list[MessageT] = []
        remaining: list[MessageT] = []
        for message in self.messages:
            arrival = message.arrival_timestamp
            if arrival is None or float(arrival) <= float(current_time):
                available.append(message)
                self._keys.discard(message_identity(message))
            else:
                remaining.append(message)
        self.messages = remaining
        return sorted(
            available,
            key=lambda message: (
                float(
                    message.timestamp
                    if message.arrival_timestamp is None
                    else message.arrival_timestamp
                ),
                float(
                    message.timestamp
                    if message.source_timestamp is None
                    else message.source_timestamp
                ),
                message_identity(message),
            ),
        )

    def __len__(self) -> int:
        return len(self.messages)


def message_source_id(message: Message) -> str:
    if isinstance(message, StateMessage):
        return str(message.source_node_id)
    return str(message.observer_id)


def message_identity(message: Message) -> tuple[object, ...]:
    if isinstance(message, ObservationMessage):
        return ("observation", str(message.message_id))
    return (
        "state",
        str(message.source_node_id),
        str(message.target_node_id),
        float(message.timestamp),
    )
