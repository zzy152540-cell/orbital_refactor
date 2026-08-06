from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CommunicationConfig:
    """Basic communication-layer configuration.

    The first version supports deterministic node dropout.
    Packet loss and delay are reserved for later extensions.
    """

    node_dropout_windows: dict[str, list[tuple[float, float]]]


def node_available(node_id: str, timestamp: float,
                   config: CommunicationConfig) -> bool:
    for start, end in config.node_dropout_windows.get(node_id, []):
        if start <= timestamp < end:
            return False
    return True


def build_validity_history(
    node_ids: list[str],
    timestamps: np.ndarray,
    config: CommunicationConfig,
) -> dict[str, np.ndarray]:
    result = {}
    for node_id in node_ids:
        result[node_id] = np.array(
            [node_available(node_id, float(t), config)
             for t in timestamps],
            dtype=bool,
        )
    return result
