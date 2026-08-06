from __future__ import annotations

import numpy as np

from cooperative.topology import NetworkTopology


def validate_communication_outages(outages, *, edges):
    """Normalize directed communication-outage windows."""

    if outages is None:
        return {}
    valid_edges = set(edges)
    normalized = {}
    for raw_edge, raw_windows in outages.items():
        edge = (str(raw_edge[0]), str(raw_edge[1]))
        if edge not in valid_edges:
            raise ValueError(f"Communication outage references unknown link: {edge}")
        windows = []
        for start, end in raw_windows:
            start = float(start)
            end = float(end)
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                raise ValueError(
                    "Communication outage windows require finite start <= end."
                )
            windows.append((start, end))
        normalized[edge] = tuple(windows)
    return normalized


def link_is_in_outage(outages, *, receiver, source, timestamp):
    """Return whether a directed link is unavailable at an epoch."""

    return any(
        start <= timestamp <= end
        for start, end in outages.get((str(receiver), str(source)), ())
    )


def validate_topology_inactive_windows(raw, *, topology: NetworkTopology):
    """Normalize inactive windows for configured undirected topology edges."""

    configured_edges = {
        tuple(sorted((node, neighbor)))
        for node in topology.node_ids
        for neighbor in topology.neighbors(node)
    }
    normalized = {}
    for edge, raw_windows in (raw or {}).items():
        if len(edge) != 2:
            raise ValueError("Topology schedule keys must contain two nodes.")
        key = tuple(sorted((str(edge[0]), str(edge[1]))))
        if key not in configured_edges:
            raise ValueError("Topology schedule references an unknown edge.")
        windows = tuple((float(start), float(end)) for start, end in raw_windows)
        if any(
            not np.isfinite(start) or not np.isfinite(end) or end < start
            for start, end in windows
        ):
            raise ValueError(
                "Topology inactive windows require finite start <= end."
            )
        normalized[key] = windows
    return normalized


def topology_edge_is_inactive(
    inactive_windows, *, first, second, timestamp,
):
    """Return whether an undirected topology edge is inactive at an epoch."""

    key = tuple(sorted((str(first), str(second))))
    return any(
        start <= timestamp <= end
        for start, end in inactive_windows.get(key, ())
    )


def topology_runtime_schedule(timestamps, *, topology, inactive_windows):
    """Build topology versions and active-neighbor maps for every epoch."""

    versions = {}
    active_neighbors = {}
    previous_signature = None
    version = 0
    for timestamp in timestamps:
        timestamp = float(timestamp)
        current = {
            node: tuple(
                neighbor for neighbor in topology.neighbors(node)
                if not topology_edge_is_inactive(
                    inactive_windows, first=node, second=neighbor,
                    timestamp=timestamp,
                )
            )
            for node in topology.node_ids
        }
        signature = tuple((node, current[node]) for node in topology.node_ids)
        if previous_signature is not None and signature != previous_signature:
            version += 1
        versions[timestamp] = version
        active_neighbors[timestamp] = current
        previous_signature = signature
    return versions, active_neighbors


def validate_measurement_periods(periods, *, modalities):
    """Normalize per-modality periods and reject disabled modality keys."""

    if periods is None:
        return {}
    unknown = set(periods) - set(modalities)
    if unknown:
        raise ValueError(
            f"Measurement periods reference disabled modalities: {sorted(unknown)}"
        )
    normalized = {str(key): float(value) for key, value in periods.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("Measurement periods must be finite and positive.")
    return normalized


def measurement_is_due(modality, timestamp, periods):
    """Return whether a periodic modality is scheduled at an epoch."""

    period = periods.get(str(modality))
    if period is None:
        return True
    quotient = float(timestamp) / period
    return bool(np.isclose(quotient, round(quotient), rtol=0.0, atol=1e-9))


def validate_absolute_navigation_dropouts(
    global_windows, windows_by_node, *, node_ids,
):
    """Normalize global and per-node absolute-navigation dropout windows."""

    global_windows = tuple(
        (float(start), float(end)) for start, end in global_windows
    )
    if any(
        not np.isfinite(start) or not np.isfinite(end) or end < start
        for start, end in global_windows
    ):
        raise ValueError(
            "Absolute-navigation dropout windows require finite start <= end."
        )
    node_windows = {
        str(node): tuple((float(start), float(end)) for start, end in windows)
        for node, windows in (windows_by_node or {}).items()
    }
    if set(node_windows) - set(node_ids):
        raise ValueError(
            "Absolute-navigation dropout windows reference unknown nodes."
        )
    if any(
        not np.isfinite(start) or not np.isfinite(end) or end < start
        for windows in node_windows.values()
        for start, end in windows
    ):
        raise ValueError(
            "Absolute-navigation dropout windows require finite start <= end."
        )
    return global_windows, node_windows
