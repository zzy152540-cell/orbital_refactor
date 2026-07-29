from dataclasses import dataclass


@dataclass
class NodeAvailabilityConfig:
    """
    Node communication availability configuration.
    """

    node_id: str

    dropout_windows: list[tuple[float, float]]


def is_node_available(
    timestamp: float,
    config: NodeAvailabilityConfig,
) -> bool:
    """
    Determine whether node is available.
    """

    for start, end in config.dropout_windows:
        if start <= timestamp < end:
            return False

    return True