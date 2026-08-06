from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from interfaces.data_objects import StateMessage

Array = np.ndarray


def validated_initial_values(
    values: Mapping[str, Array],
    node_ids: tuple[str, ...],
    shape: tuple[int, ...],
    name: str,
) -> dict[str, Array]:
    """Validate and copy node-indexed initial arrays."""

    if set(values) != set(node_ids):
        raise ValueError(f"Initial {name} keys must match topology node IDs.")
    result = {}
    for node_id, value in values.items():
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            raise ValueError(f"Initial {name} for {node_id} must have shape {shape}.")
        result[node_id] = array.copy()
    return result


def state_source_timestamp(message: StateMessage) -> float:
    """Return the estimator epoch represented by a transported state."""

    return float(
        message.timestamp
        if message.source_timestamp is None
        else message.source_timestamp
    )


def state_at_or_before(
    messages: Sequence[StateMessage],
    timestamp: float,
) -> StateMessage | None:
    """Return the newest message whose source epoch is not in the future."""

    candidates = [
        message
        for message in messages
        if state_source_timestamp(message) <= float(timestamp) + 1e-12
    ]
    return max(candidates, key=state_source_timestamp) if candidates else None
