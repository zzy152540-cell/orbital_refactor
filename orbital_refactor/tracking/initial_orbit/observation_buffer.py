from __future__ import annotations

from dataclasses import dataclass

from .data_contracts import IODObservation


@dataclass(frozen=True)
class IODBufferConfig:
    minimum_epochs: int = 3
    minimum_observers: int = 2
    minimum_time_span_seconds: float = 2.0
    maximum_time_span_seconds: float = 60.0

    def __post_init__(self):
        if self.minimum_epochs < 2:
            raise ValueError("minimum_epochs must be at least 2.")
        if self.minimum_observers < 1:
            raise ValueError("minimum_observers must be positive.")
        if self.minimum_time_span_seconds <= 0.0:
            raise ValueError("minimum_time_span_seconds must be positive.")
        if self.maximum_time_span_seconds < self.minimum_time_span_seconds:
            raise ValueError("maximum_time_span_seconds must cover the minimum span.")


class TargetIODObservationBuffer:
    """Bounded, target-specific short-arc observation buffer."""

    def __init__(self, target_id: str, config: IODBufferConfig | None = None):
        self.target_id = str(target_id)
        self.config = config or IODBufferConfig()
        self._observations: list[IODObservation] = []
        self._keys = set()

    @property
    def observations(self) -> tuple[IODObservation, ...]:
        return tuple(self._observations)

    def add(self, observation: IODObservation) -> bool:
        if observation.target_id != self.target_id:
            raise ValueError("Observation target_id does not match this IOD buffer.")
        key = _observation_key(observation)
        if key in self._keys:
            return False
        self._keys.add(key)
        self._observations.append(observation)
        self._observations.sort(key=lambda item: item.timestamp)
        self._trim()
        return True

    def extend(self, observations) -> int:
        return sum(self.add(item) for item in observations)

    def is_ready(self) -> bool:
        valid = [item for item in self._observations if item.valid_flag]
        if not valid:
            return False
        epochs = sorted({item.timestamp for item in valid})
        observers = {item.observer_id for item in valid}
        modalities = {item.modality for item in valid}
        return bool(
            len(epochs) >= self.config.minimum_epochs
            and len(observers) >= self.config.minimum_observers
            and epochs[-1] - epochs[0] >= self.config.minimum_time_span_seconds
            and {"LOS", "RADAR"} <= modalities
        )

    def clear(self) -> None:
        self._observations.clear()
        self._keys.clear()

    def _trim(self):
        if not self._observations:
            return
        newest = self._observations[-1].timestamp
        cutoff = newest - self.config.maximum_time_span_seconds
        retained = [item for item in self._observations if item.timestamp >= cutoff]
        if len(retained) == len(self._observations):
            return
        self._observations = retained
        self._keys = {_observation_key(item) for item in retained}


def _observation_key(observation):
    source = observation.metadata.get(
        "sourceMessageId", observation.metadata.get("sourceModality"),
    )
    if source is None:
        source = tuple(observation.measurement.tolist())
    return (
        observation.observer_id,
        float(observation.timestamp),
        observation.modality,
        source,
    )
