from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

UndirectedEdge = tuple[str, str]


def normalized_undirected_edge(left: str, right: str) -> UndirectedEdge:
    if left == right:
        raise ValueError("An undirected edge requires two different nodes.")
    return tuple(sorted((str(left), str(right))))


@dataclass(frozen=True)
class GraphNodeFeature:
    node_id: str
    state: tuple[float, ...]
    covariance_diagonal: tuple[float, ...] | None = None
    estimator_metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class GraphMeasurementFeature:
    observer_id: str
    target_id: str
    modality: str
    frame: str
    covariance: tuple[tuple[float, ...], ...]
    quaternion_i2b_wxyz: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.observer_id == self.target_id:
            raise ValueError("A relative measurement requires two nodes.")
        covariance = tuple(tuple(row) for row in self.covariance)
        if not covariance or any(len(row) != len(covariance) for row in covariance):
            raise ValueError("Measurement covariance must be square.")
        if self.quaternion_i2b_wxyz is not None and (
            len(self.quaternion_i2b_wxyz) != 4
        ):
            raise ValueError("Measurement attitude must be a quaternion.")


@dataclass(frozen=True)
class GraphEdgeFeature:
    nodes: UndirectedEdge
    distance: float
    geometrically_visible: bool = True
    measurement_modalities: tuple[str, ...] = ()
    communication_available: bool = True
    delay: float = 0.0
    packet_loss_rate: float = 0.0
    nis_by_modality: tuple[tuple[str, float], ...] = ()
    nis_sample_count_by_modality: tuple[tuple[str, int], ...] = ()
    consecutive_anomaly_count_by_modality: tuple[tuple[str, int], ...] = ()
    observation_age: float | None = None

    def __post_init__(self) -> None:
        if self.nodes != normalized_undirected_edge(*self.nodes):
            raise ValueError("GraphEdgeFeature nodes must be sorted.")
        if self.distance < 0.0 or self.delay < 0.0:
            raise ValueError("Edge distance and delay cannot be negative.")
        if self.observation_age is not None and self.observation_age < 0.0:
            raise ValueError("observation_age cannot be negative.")
        if not 0.0 <= self.packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be in [0, 1].")
        if any(value < 0 for _, value in self.nis_sample_count_by_modality):
            raise ValueError("NIS sample counts cannot be negative.")
        if any(
            value < 0 for _, value in self.consecutive_anomaly_count_by_modality
        ):
            raise ValueError("Consecutive anomaly counts cannot be negative.")


@dataclass(frozen=True)
class GraphObservationProvenance:
    """Declare feature sources so policy inputs cannot hide truth leakage."""

    schema_version: str = "v14-unspecified"
    state_source: str = "unspecified"
    geometry_source: str = "unspecified"
    online_decision_safe: bool = False

    def __post_init__(self) -> None:
        allowed = {"estimator", "measurement", "configured", "truth", "unspecified"}
        if not self.schema_version:
            raise ValueError("Graph observation schema version cannot be empty.")
        if self.state_source not in allowed:
            raise ValueError("Unsupported graph state source.")
        if self.geometry_source not in allowed:
            raise ValueError("Unsupported graph geometry source.")
        if self.online_decision_safe and (
            self.state_source != "estimator"
            or self.geometry_source not in {"estimator", "measurement"}
        ):
            raise ValueError(
                "Online-safe observations require estimator states and "
                "estimated or measured geometry."
            )


@dataclass(frozen=True)
class GraphObservation:
    """One topology-decision snapshot, independent of policy implementation."""

    timestamp: float
    nodes: tuple[GraphNodeFeature, ...]
    candidate_edges: tuple[GraphEdgeFeature, ...]
    previous_active_edges: tuple[UndirectedEdge, ...] = ()
    estimation_dependency_edges: tuple[UndirectedEdge, ...] = ()
    graph_metrics: tuple[tuple[str, float], ...] = ()
    measurements: tuple[GraphMeasurementFeature, ...] = ()
    provenance: GraphObservationProvenance = GraphObservationProvenance()

    def __post_init__(self) -> None:
        node_ids = tuple(node.node_id for node in self.nodes)
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("GraphObservation requires unique nodes.")
        node_set = set(node_ids)
        edges = tuple(edge.nodes for edge in self.candidate_edges)
        if len(set(edges)) != len(edges):
            raise ValueError("GraphObservation candidate edges must be unique.")
        for edge in (
            *edges, *self.previous_active_edges,
            *self.estimation_dependency_edges,
        ):
            if set(edge) - node_set:
                raise ValueError("GraphObservation edge references an unknown node.")
        for measurement in self.measurements:
            if {measurement.observer_id, measurement.target_id} - node_set:
                raise ValueError("Graph measurement references an unknown node.")


def validate_deployment_graph_observation(
    observation: GraphObservation,
) -> None:
    """Reject observations whose sources are unsafe or unspecified for policy use."""

    provenance = observation.provenance
    if not provenance.online_decision_safe:
        raise ValueError("Graph observation is not marked online-decision safe.")
    if "truth" in {provenance.state_source, provenance.geometry_source}:
        raise ValueError("Deployment graph observation cannot use truth features.")


@dataclass(frozen=True)
class TopologyAction:
    """Topology selected from a decision-time graph observation."""

    policy_name: str
    active_edges: tuple[UndirectedEdge, ...]

    def __post_init__(self) -> None:
        if len(set(self.active_edges)) != len(self.active_edges):
            raise ValueError("TopologyAction active edges must be unique.")
        if any(edge != normalized_undirected_edge(*edge)
               for edge in self.active_edges):
            raise ValueError("TopologyAction active edges must be sorted.")


TopologyDecision = TopologyAction


class TopologyPolicy(Protocol):
    def select(self, observation: GraphObservation) -> TopologyAction: ...


def build_graph_observation(
    *, timestamp, state_by_node, candidate_distance_by_edge,
    previous_active_edges=(), covariance_by_node=None,
    estimator_metrics_by_node=None, measurement_modalities_by_edge=None,
    geometrically_visible_by_edge=None,
    communication_available_by_edge=None, delay_by_edge=None,
    packet_loss_rate_by_edge=None, nis_by_modality_by_edge=None,
    nis_sample_count_by_modality_by_edge=None,
    consecutive_anomaly_count_by_modality_by_edge=None,
    observation_age_by_edge=None, estimation_dependency_edges=(),
    graph_metrics=None,
    measurements=(),
    provenance=None,
) -> GraphObservation:
    """Normalize estimator, visibility, and communication data for a policy."""

    covariance_by_node = covariance_by_node or {}
    estimator_metrics_by_node = estimator_metrics_by_node or {}
    measurement_modalities_by_edge = measurement_modalities_by_edge or {}
    geometrically_visible_by_edge = geometrically_visible_by_edge or {}
    communication_available_by_edge = communication_available_by_edge or {}
    delay_by_edge = delay_by_edge or {}
    packet_loss_rate_by_edge = packet_loss_rate_by_edge or {}
    nis_by_modality_by_edge = nis_by_modality_by_edge or {}
    nis_sample_count_by_modality_by_edge = (
        nis_sample_count_by_modality_by_edge or {}
    )
    consecutive_anomaly_count_by_modality_by_edge = (
        consecutive_anomaly_count_by_modality_by_edge or {}
    )
    observation_age_by_edge = observation_age_by_edge or {}
    graph_metrics = graph_metrics or {}

    def edge_value(mapping, edge, default):
        return mapping.get(edge, mapping.get((edge[1], edge[0]), default))

    nodes = []
    for node_id, state in state_by_node.items():
        covariance = covariance_by_node.get(node_id)
        diagonal = (
            tuple(float(covariance[index][index])
                  for index in range(len(covariance)))
            if covariance is not None else None
        )
        nodes.append(GraphNodeFeature(
            node_id=str(node_id),
            state=tuple(float(value) for value in state),
            covariance_diagonal=diagonal,
            estimator_metrics=tuple(sorted(
                (str(key), float(value))
                for key, value in estimator_metrics_by_node.get(node_id, {}).items()
            )),
        ))
    edges = []
    for raw_edge, distance in candidate_distance_by_edge.items():
        edge = normalized_undirected_edge(*raw_edge)
        nis = edge_value(nis_by_modality_by_edge, edge, {})
        sample_count = edge_value(
            nis_sample_count_by_modality_by_edge, edge, {}
        )
        anomaly_count = edge_value(
            consecutive_anomaly_count_by_modality_by_edge, edge, {}
        )
        edges.append(GraphEdgeFeature(
            nodes=edge,
            distance=float(distance),
            geometrically_visible=bool(edge_value(
                geometrically_visible_by_edge, edge, True
            )),
            measurement_modalities=tuple(sorted(
                str(value) for value in edge_value(
                    measurement_modalities_by_edge, edge, ()
                )
            )),
            communication_available=bool(edge_value(
                communication_available_by_edge, edge, True
            )),
            delay=float(edge_value(delay_by_edge, edge, 0.0)),
            packet_loss_rate=float(edge_value(
                packet_loss_rate_by_edge, edge, 0.0
            )),
            nis_by_modality=tuple(sorted(
                (str(key), float(value)) for key, value in nis.items()
            )),
            nis_sample_count_by_modality=tuple(sorted(
                (str(key), int(value)) for key, value in sample_count.items()
            )),
            consecutive_anomaly_count_by_modality=tuple(sorted(
                (str(key), int(value)) for key, value in anomaly_count.items()
            )),
            observation_age=(
                None
                if edge_value(observation_age_by_edge, edge, None) is None
                else float(edge_value(
                    observation_age_by_edge, edge, None
                ))
            ),
        ))
    return GraphObservation(
        timestamp=float(timestamp),
        nodes=tuple(nodes),
        candidate_edges=tuple(sorted(edges, key=lambda value: value.nodes)),
        previous_active_edges=tuple(sorted(
            normalized_undirected_edge(*edge) for edge in previous_active_edges
        )),
        estimation_dependency_edges=tuple(sorted(
            normalized_undirected_edge(*edge)
            for edge in estimation_dependency_edges
        )),
        graph_metrics=tuple(sorted(
            (str(key), float(value)) for key, value in graph_metrics.items()
        )),
        measurements=tuple(measurements),
        provenance=(
            GraphObservationProvenance()
            if provenance is None else provenance
        ),
    )


@dataclass(frozen=True)
class LowChurnConnectedTreePolicy:
    """Retain short old edges, then form the existing degree-bounded tree."""

    maximum_degree: int = 3

    def __post_init__(self) -> None:
        if self.maximum_degree < 2:
            raise ValueError("maximum_degree must be at least two.")

    def select(self, observation: GraphObservation) -> TopologyAction:
        nodes = tuple(node.node_id for node in observation.nodes)
        candidates = {
            edge.nodes: edge.distance for edge in observation.candidate_edges
            if edge.geometrically_visible and edge.communication_available
        }
        previous = set(observation.previous_active_edges)
        if _component_sizes(nodes, candidates) != (len(nodes),):
            raise ValueError("Candidate visibility graph is disconnected.")
        parent = {node: node for node in nodes}
        degree = {node: 0 for node in nodes}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        selected = set()
        for left, right in sorted(
            candidates,
            key=lambda edge: (edge not in previous, candidates[edge], edge),
        ):
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                continue
            if degree[left] >= self.maximum_degree or degree[right] >= self.maximum_degree:
                continue
            selected.add((left, right))
            degree[left] += 1
            degree[right] += 1
            parent[right_root] = left_root
            if len(selected) == len(nodes) - 1:
                break
        if len(selected) != len(nodes) - 1:
            raise RuntimeError("Greedy degree-constrained selector could not form a spanning tree.")
        return TopologyAction(type(self).__name__, tuple(sorted(selected)))


def _component_sizes(node_ids, edges) -> tuple[int, ...]:
    adjacency = {node: [] for node in node_ids}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    unseen, sizes = set(node_ids), []
    while unseen:
        pending, size = [unseen.pop()], 0
        while pending:
            node, size = pending.pop(), size + 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))
