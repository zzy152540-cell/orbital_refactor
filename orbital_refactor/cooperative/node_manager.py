import numpy as np

from configs.communication_config import (
    NodeAvailabilityConfig,
    is_node_available,
)


def apply_node_availability(
    local_history_by_node,
    timestamps,
    configs,
):

    output = {}

    for node_id, history in local_history_by_node.items():

        config = configs[node_id]

        valid_mask = np.array(
            [
                is_node_available(
                    t,
                    config,
                )
                for t in timestamps
            ]
        )

        history = history.copy()

        history[~valid_mask] = np.nan

        output[node_id] = history

    return output