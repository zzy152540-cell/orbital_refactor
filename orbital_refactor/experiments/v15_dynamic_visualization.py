"""Build topology and event overlays for a phased Walker replay."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from visualization.data_contract import VisualDiagnosticEvent, VisualEdge


def phased_visualization_overlays(*, case, dropout_nodes, inactive_edge,
                                  dropout_window, topology_window):
    timestamps = np.asarray(case["timestamps"], dtype=float)
    configured_pairs = tuple(sorted({
        tuple(sorted((node, neighbor)))
        for node in case["topology"].node_ids
        for neighbor in case["topology"].neighbors(node)
    }))
    delivered = defaultdict(set)
    for receiver, messages in case["state_messages"].items():
        for message in messages:
            arrival = (
                message.timestamp if message.arrival_timestamp is None
                else message.arrival_timestamp
            )
            matches = np.flatnonzero(np.isclose(timestamps, float(arrival)))
            if matches.size:
                delivered[int(matches[0])].add(
                    (str(message.source_node_id), str(receiver))
                )

    edges_by_epoch = []
    for index, timestamp in enumerate(timestamps):
        active = case["active_neighbors_by_timestamp"][float(timestamp)]
        active_pairs = {
            tuple(sorted((node, neighbor)))
            for node, neighbors in active.items() for neighbor in neighbors
        }
        configured = tuple(
            VisualEdge(left, right, "CONFIGURED_TOPOLOGY",
                       status=("ACTIVE" if (left, right) in active_pairs else "INACTIVE"))
            for left, right in configured_pairs
        )
        current = tuple(
            VisualEdge(left, right, "ACTIVE_TOPOLOGY")
            for left, right in sorted(active_pairs)
        )
        flow = tuple(
            VisualEdge(source, target, "ACTUAL_INFORMATION_FLOW", directed=True)
            for source, target in sorted(delivered[index])
        )
        edges_by_epoch.append(configured + current + flow)

    events = defaultdict(list)
    d0, d1 = map(float, dropout_window)
    t0, t1 = map(float, topology_window)
    labels = (
        (d0, "ABS_NAV_DROPOUT_START", "WARNING",
         f"Absolute navigation lost on {', '.join(dropout_nodes)}"),
        (d1 + _step(timestamps), "ABS_NAV_RECOVERED", "INFO",
         f"Absolute navigation restored on {', '.join(dropout_nodes)}"),
        (t0, "TOPOLOGY_LINK_SUSPENDED", "WARNING",
         f"Link {inactive_edge[0]}--{inactive_edge[1]} suspended"),
        (t1 + _step(timestamps), "TOPOLOGY_LINK_RECOVERED", "INFO",
         f"Link {inactive_edge[0]}--{inactive_edge[1]} restored"),
    )
    for timestamp, event_type, severity, description in labels:
        matches = np.flatnonzero(np.isclose(timestamps, timestamp))
        if matches.size:
            events[int(matches[0])].append(VisualDiagnosticEvent(
                timestamp=timestamp, event_type=event_type, severity=severity,
                description=description,
            ))
    return tuple(edges_by_epoch), tuple(
        tuple(events[index]) for index in range(timestamps.size)
    )


def _step(timestamps):
    return 0.0 if len(timestamps) < 2 else float(np.median(np.diff(timestamps)))
