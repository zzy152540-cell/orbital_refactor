from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from interfaces.data_objects import Observation


class MeasurementPreprocessor(Protocol):
    """Optional measurement-side boundary shared by single and multi-node runs.

    Implementations may keep temporal state, so callers must create one
    instance per observer/target stream.  The returned objects remain standard
    ``Observation`` instances; estimator state and covariance are inaccessible
    through this interface.
    """

    def process(
        self, observations: Sequence[Observation], timestamps: np.ndarray,
    ) -> list[Observation]: ...


def validate_preprocessed_stream(
    source: Sequence[Observation], processed: Sequence[Observation], *,
    observer_id: str,
) -> list[Observation]:
    """Fail closed when a preprocessor changes stream identity or chronology."""

    result = list(processed)
    if len(result) != len(source):
        raise ValueError("A measurement preprocessor cannot change stream length.")
    for original, candidate in zip(source, result):
        if candidate.observer_id != observer_id:
            raise ValueError("A measurement preprocessor cannot move data between observers.")
        if candidate.target_id != original.target_id:
            raise ValueError("A measurement preprocessor cannot change the physical target.")
        if candidate.modality.lower() != original.modality.lower():
            raise ValueError("A measurement preprocessor cannot change modality.")
        if not np.isclose(candidate.timestamp, original.timestamp, rtol=0.0, atol=1e-12):
            raise ValueError("A measurement preprocessor cannot change timestamps.")
    return result
