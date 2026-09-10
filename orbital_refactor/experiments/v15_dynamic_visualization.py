"""Build topology and event overlays for a phased Walker replay."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from visualization.data_contract import VisualDiagnosticEvent, VisualEdge


COMMUNICATION_DEGRADATION_PROFILES = {
    "mild": ((60.0, 74.0, 0.10, 2.0), (76.0, 88.0, 0.25, 2.0)),
    "moderate": ((60.0, 74.0, 0.30, 2.0), (76.0, 88.0, 0.60, 4.0)),
    "aggressive": ((60.0, 74.0, 0.50, 4.0), (76.0, 88.0, 0.80, 6.0)),
}


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


def online_visualization_overlays(*, case, steps, dropout_nodes,
                                  dropout_window):
    """Project actual online transport and resynchronization diagnostics."""

    timestamps = np.asarray(case["timestamps"], dtype=float)
    configured_pairs = tuple(sorted({
        tuple(sorted((node, neighbor)))
        for node in case["topology"].node_ids
        for neighbor in case["topology"].neighbors(node)
    }))
    edges_by_epoch, events_by_epoch, metadata_by_epoch = [], [], []
    previous_active_pairs = None
    for timestamp, step in zip(timestamps, steps):
        active = case["active_neighbors_by_timestamp"][float(timestamp)]
        active_pairs = {
            tuple(sorted((node, neighbor)))
            for node, neighbors in active.items() for neighbor in neighbors
        }
        edges = [
            VisualEdge(left, right, "CONFIGURED_TOPOLOGY", status=(
                "ACTIVE" if (left, right) in active_pairs else "INACTIVE"
            ))
            for left, right in configured_pairs
        ]
        edges.extend(
            VisualEdge(left, right, "ACTIVE_TOPOLOGY")
            for left, right in sorted(active_pairs)
        )
        delivered = {
            (str(record["source_id"]), str(record["receiver_id"]))
            for record in step.message_diagnostic_records
            if bool(record.get("accepted"))
        }
        delayed = {
            (str(record["source_id"]), str(record["receiver_id"]))
            for record in step.transport_diagnostic_records
            if record["status"] == "DELAYED"
        }
        edges.extend(
            VisualEdge(source, receiver, "ACTUAL_INFORMATION_FLOW", directed=True)
            for source, receiver in sorted(delivered)
        )
        edges.extend(
            VisualEdge(source, receiver, "PENDING_INFORMATION_FLOW",
                       status="DELAYED", directed=True)
            for source, receiver in sorted(delayed - delivered)
        )

        events = []
        if previous_active_pairs is not None:
            removed = previous_active_pairs - active_pairs
            restored = active_pairs - previous_active_pairs
            for edge in sorted(removed):
                events.append(_event(timestamp, "TOPOLOGY_LINK_SUSPENDED",
                                     "WARNING", f"Link {edge[0]}--{edge[1]} suspended"))
            for edge in sorted(restored):
                events.append(_event(timestamp, "TOPOLOGY_LINK_RECOVERED", "INFO",
                                     f"Link {edge[0]}--{edge[1]} restored"))
        dropped = step.dropped_message_count
        delayed_count = sum(
            record["status"] == "DELAYED"
            for record in step.transport_diagnostic_records
        )
        if dropped:
            events.append(_event(timestamp, "PACKET_LOSS", "WARNING",
                                 f"{dropped} state messages dropped"))
        if delayed_count:
            events.append(_event(timestamp, "MESSAGE_DELAYED", "INFO",
                                 f"{delayed_count} state messages delayed"))
        if step.protocol_rejected_message_count:
            reasons = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(step.rejection_counts_by_reason.items())
            )
            events.append(_event(
                timestamp, "RESYNC_REQUIRED", "WARNING",
                f"{step.protocol_rejected_message_count} protocol gaps: {reasons}",
            ))
        if step.resynchronized_links:
            links = ", ".join(
                f"{source}->{receiver}"
                for receiver, source, _ in step.resynchronized_links[:4]
            )
            remainder = len(step.resynchronized_links) - 4
            suffix = "" if remainder <= 0 else f" and {remainder} more"
            events.append(_event(
                timestamp, "RESYNC_COMPLETED", "INFO",
                f"{len(step.resynchronized_links)} links resynchronized: "
                f"{links}{suffix}",
            ))
        d0, d1 = map(float, dropout_window)
        if np.isclose(timestamp, d0):
            events.append(_event(timestamp, "ABS_NAV_DROPOUT_START", "WARNING",
                                 f"Absolute navigation lost on {', '.join(dropout_nodes)}"))
        if np.isclose(timestamp, d1 + _step(timestamps)):
            events.append(_event(timestamp, "ABS_NAV_RECOVERED", "INFO",
                                 f"Absolute navigation restored on {', '.join(dropout_nodes)}"))
        edges_by_epoch.append(tuple(edges))
        events_by_epoch.append(tuple(events))
        metadata_by_epoch.append({
            "topology_version": int(case["topology_version_by_timestamp"][float(timestamp)]),
            "active_edge_count": len(active_pairs),
            "transmitted_message_count": step.transmitted_message_count,
            "delivered_message_count": step.accepted_message_count,
            "dropped_message_count": dropped,
            "delayed_message_count": delayed_count,
            "resynchronization_count": len(step.resynchronized_links),
        })
        previous_active_pairs = active_pairs
    return tuple(edges_by_epoch), tuple(events_by_epoch), tuple(metadata_by_epoch)


def _event(timestamp, event_type, severity, description):
    return VisualDiagnosticEvent(
        timestamp=float(timestamp), event_type=event_type,
        severity=severity, description=description,
    )


def _step(timestamps):
    return 0.0 if len(timestamps) < 2 else float(np.median(np.diff(timestamps)))
