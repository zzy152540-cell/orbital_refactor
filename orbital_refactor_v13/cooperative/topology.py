from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class NetworkTopology:
    """Undirected communication-neighbor map for satellite nodes."""

    adjacency: dict[str, tuple[str, ...]]

    def __init__(self, adjacency: Mapping[str, list[str] | tuple[str, ...]]) -> None:
        normalized = {
            str(node_id): tuple(str(neighbor) for neighbor in neighbors)
            for node_id, neighbors in adjacency.items()
        }
        _validate_adjacency(normalized)
        object.__setattr__(self, "adjacency", normalized)

    @property
    def node_ids(self) -> list[str]:
        return list(self.adjacency)

    def neighbors(self, node_id: str) -> tuple[str, ...]:
        try:
            return self.adjacency[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown node_id: {node_id}") from exc

    def includes_edge(self, source_id: str, target_id: str) -> bool:
        return target_id in self.neighbors(source_id)


def chain_topology(node_ids: list[str] | tuple[str, ...]) -> NetworkTopology:
    """Build the v13.1 first-step chain topology, e.g. sat_01--sat_02--sat_03."""

    nodes = [str(node_id) for node_id in node_ids]
    if len(nodes) < 1:
        raise ValueError("At least one node is required.")
    adjacency: dict[str, list[str]] = {}
    for index, node_id in enumerate(nodes):
        neighbors: list[str] = []
        if index > 0:
            neighbors.append(nodes[index - 1])
        if index + 1 < len(nodes):
            neighbors.append(nodes[index + 1])
        adjacency[node_id] = neighbors
    return NetworkTopology(adjacency)


def fully_connected_topology(
    node_ids: list[str] | tuple[str, ...],
) -> NetworkTopology:
    """Build the symmetric all-to-all topology used by the three-satellite baseline."""

    nodes = [str(node_id) for node_id in node_ids]
    if not nodes:
        raise ValueError("At least one node is required.")
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique.")
    return NetworkTopology(
        {node_id: [other for other in nodes if other != node_id] for node_id in nodes}
    )


def ring_topology(node_ids: list[str] | tuple[str, ...]) -> NetworkTopology:
    nodes = [str(node_id) for node_id in node_ids]
    if len(nodes) < 3:
        raise ValueError("Ring topology requires at least three nodes.")
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique.")
    return NetworkTopology({
        node_id: [nodes[(index - 1) % len(nodes)], nodes[(index + 1) % len(nodes)]]
        for index, node_id in enumerate(nodes)
    })


def star_topology(node_ids: list[str] | tuple[str, ...]) -> NetworkTopology:
    nodes = [str(node_id) for node_id in node_ids]
    if len(nodes) < 2:
        raise ValueError("Star topology requires at least two nodes.")
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique.")
    center = nodes[0]
    return NetworkTopology({
        node_id: ([other for other in nodes if other != center] if node_id == center else [center])
        for node_id in nodes
    })


def _validate_adjacency(adjacency: Mapping[str, tuple[str, ...]]) -> None:
    if not adjacency:
        raise ValueError("Topology cannot be empty.")
    node_set = set(adjacency)
    for node_id, neighbors in adjacency.items():
        if node_id in neighbors:
            raise ValueError(f"Node {node_id!r} cannot be its own neighbor.")
        missing = [neighbor for neighbor in neighbors if neighbor not in node_set]
        if missing:
            raise ValueError(f"Topology for {node_id!r} references unknown nodes: {missing}")
        for neighbor in neighbors:
            if node_id not in adjacency[neighbor]:
                raise ValueError(f"Topology edge {node_id!r}->{neighbor!r} must be symmetric.")
