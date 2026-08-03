from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from orbital_core.constants import R_EARTH

Array = np.ndarray


@dataclass(frozen=True)
class VisibilityConfig:
    """Physical geometry limits for one inter-satellite observation."""

    earth_occultation: bool = True
    earth_radius: float = R_EARTH
    earth_clearance: float = 0.0
    maximum_range: float | None = None
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.earth_radius <= 0.0:
            raise ValueError("earth_radius must be positive.")
        if self.earth_clearance < 0.0:
            raise ValueError("earth_clearance cannot be negative.")
        if self.maximum_range is not None and self.maximum_range <= 0.0:
            raise ValueError("maximum_range must be positive when provided.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance cannot be negative.")


@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    reason: str
    range: float
    earth_clearance: float


@dataclass(frozen=True)
class MeasurementOpportunity:
    timestamp: float
    observer_id: str
    target_id: str
    modality: str
    visibility: VisibilityResult


@dataclass(frozen=True)
class VisibilityCountSummary:
    opportunity_count: int
    visible_count: int
    visibility_rate: float
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class VisibilityOpportunitySummary:
    overall: VisibilityCountSummary
    by_modality: dict[str, VisibilityCountSummary]
    by_directed_edge: dict[tuple[str, str], VisibilityCountSummary]
    visible_directed_edge_count_by_timestamp: dict[float, int]
    longest_unavailable_epochs_by_edge_and_modality: dict[tuple[str, str, str], int]
    longest_unavailable_span_by_edge_and_modality: dict[tuple[str, str, str], float]


class CandidateTopology(Protocol):
    @property
    def node_ids(self) -> tuple[str, ...]: ...

    def neighbors(self, node_id: str) -> tuple[str, ...]: ...


def evaluate_inter_satellite_visibility(
    observer_state: Array,
    target_state: Array,
    config: VisibilityConfig | None = None,
) -> VisibilityResult:
    """Evaluate range and finite-segment Earth occultation in ECI geometry."""

    limits = VisibilityConfig() if config is None else config
    observer = _position(observer_state, "observer_state")
    target = _position(target_state, "target_state")
    relative = target - observer
    distance = float(np.linalg.norm(relative))
    if distance <= limits.tolerance:
        return VisibilityResult(False, "invalid_geometry", distance, float("nan"))

    effective_radius = limits.earth_radius + limits.earth_clearance
    observer_radius = float(np.linalg.norm(observer))
    target_radius = float(np.linalg.norm(target))
    if min(observer_radius, target_radius) <= effective_radius + limits.tolerance:
        clearance = min(observer_radius, target_radius) - effective_radius
        return VisibilityResult(False, "invalid_geometry", distance, clearance)

    fraction = float(-np.dot(observer, relative) / np.dot(relative, relative))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    closest = observer + fraction * relative
    clearance = float(np.linalg.norm(closest) - effective_radius)
    if limits.earth_occultation and clearance <= limits.tolerance:
        return VisibilityResult(False, "earth_occulted", distance, clearance)
    if (
        limits.maximum_range is not None
        and distance > limits.maximum_range + limits.tolerance
    ):
        return VisibilityResult(False, "range_exceeded", distance, clearance)
    return VisibilityResult(True, "visible", distance, clearance)


def generate_inter_satellite_observation_opportunities(
    *,
    timestamps: Array,
    truth_state_history_by_node: Mapping[str, Array],
    candidate_topology: CandidateTopology,
    visibility_by_modality: Mapping[str, VisibilityConfig],
) -> tuple[MeasurementOpportunity, ...]:
    """Generate directed physical observation opportunities for every epoch."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    node_ids = tuple(str(value) for value in candidate_topology.node_ids)
    if set(truth_state_history_by_node) != set(node_ids):
        raise ValueError("Truth-history keys must match candidate topology nodes.")
    if not visibility_by_modality:
        raise ValueError("At least one visibility modality is required.")
    normalized_visibility = {
        str(modality): config
        for modality, config in visibility_by_modality.items()
    }
    if len(normalized_visibility) != len(visibility_by_modality):
        raise ValueError("Visibility modality names must remain unique as strings.")
    if not all(
        isinstance(config, VisibilityConfig)
        for config in normalized_visibility.values()
    ):
        raise TypeError("Every modality visibility value must be VisibilityConfig.")
    modalities = tuple(sorted(normalized_visibility))
    histories = {
        node_id: _state_history(
            truth_state_history_by_node[node_id], times.size, node_id
        )
        for node_id in node_ids
    }
    opportunities = []
    for index, timestamp in enumerate(times):
        for observer_id in node_ids:
            for target_id in candidate_topology.neighbors(observer_id):
                target_id = str(target_id)
                if target_id not in histories or target_id == observer_id:
                    raise ValueError("Candidate neighbors must be distinct topology nodes.")
                for modality in modalities:
                    visibility = evaluate_inter_satellite_visibility(
                        histories[observer_id][index], histories[target_id][index],
                        normalized_visibility[modality],
                    )
                    opportunities.append(MeasurementOpportunity(
                        timestamp=float(timestamp), observer_id=observer_id,
                        target_id=target_id, modality=modality,
                        visibility=visibility,
                    ))
    return tuple(opportunities)


def summarize_observation_opportunities(
    opportunities: tuple[MeasurementOpportunity, ...],
) -> VisibilityOpportunitySummary:
    """Summarize visibility without altering or discarding opportunity records.

    Unavailable span is the elapsed time from the first to last sample in the
    longest consecutive unavailable run. A one-sample run therefore has zero
    span while still reporting one unavailable epoch.
    """

    if not opportunities:
        raise ValueError("At least one measurement opportunity is required.")
    ordered = sorted(
        opportunities,
        key=lambda item: (
            item.timestamp, item.observer_id, item.target_id, item.modality,
        ),
    )
    overall = _count_summary(ordered)
    modality_groups: dict[str, list[MeasurementOpportunity]] = {}
    edge_groups: dict[tuple[str, str], list[MeasurementOpportunity]] = {}
    timeline_groups: dict[tuple[str, str, str], list[MeasurementOpportunity]] = {}
    visible_edges: dict[float, set[tuple[str, str]]] = {}
    all_timestamps = set()
    for item in ordered:
        modality_groups.setdefault(item.modality, []).append(item)
        edge = (item.observer_id, item.target_id)
        edge_groups.setdefault(edge, []).append(item)
        timeline_groups.setdefault((*edge, item.modality), []).append(item)
        timestamp = float(item.timestamp)
        all_timestamps.add(timestamp)
        if item.visibility.visible:
            visible_edges.setdefault(timestamp, set()).add(edge)
    longest_epochs = {}
    longest_spans = {}
    for key, values in timeline_groups.items():
        epochs, span = _longest_unavailable_run(values)
        longest_epochs[key] = epochs
        longest_spans[key] = span
    return VisibilityOpportunitySummary(
        overall=overall,
        by_modality={
            key: _count_summary(values)
            for key, values in sorted(modality_groups.items())
        },
        by_directed_edge={
            key: _count_summary(values)
            for key, values in sorted(edge_groups.items())
        },
        visible_directed_edge_count_by_timestamp={
            timestamp: len(visible_edges.get(timestamp, set()))
            for timestamp in sorted(all_timestamps)
        },
        longest_unavailable_epochs_by_edge_and_modality=longest_epochs,
        longest_unavailable_span_by_edge_and_modality=longest_spans,
    )


def _position(state: Array, name: str) -> Array:
    value = np.asarray(state, dtype=float).reshape(-1)
    if value.size not in {3, 6}:
        raise ValueError(f"{name} must contain either 3-position or 6-state values.")
    position = value[:3]
    if not np.all(np.isfinite(position)):
        raise ValueError(f"{name} position must contain only finite values.")
    return position


def _state_history(history: Array, count: int, node_id: str) -> Array:
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] not in {3, 6}:
        raise ValueError(
            f"Truth history for {node_id} must have shape ({count}, 3) or ({count}, 6)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Truth history for {node_id} must contain finite values.")
    return values


def _count_summary(
    opportunities: list[MeasurementOpportunity],
) -> VisibilityCountSummary:
    visible = sum(item.visibility.visible for item in opportunities)
    rejections: dict[str, int] = {}
    for item in opportunities:
        if not item.visibility.visible:
            reason = item.visibility.reason
            rejections[reason] = rejections.get(reason, 0) + 1
    total = len(opportunities)
    return VisibilityCountSummary(
        opportunity_count=total,
        visible_count=visible,
        visibility_rate=visible / total,
        rejection_counts=rejections,
    )


def _longest_unavailable_run(
    opportunities: list[MeasurementOpportunity],
) -> tuple[int, float]:
    ordered = sorted(opportunities, key=lambda item: item.timestamp)
    best_epochs = 0
    best_span = 0.0
    run_start = 0.0
    run_epochs = 0
    previous_timestamp = 0.0
    for item in ordered:
        timestamp = float(item.timestamp)
        if item.visibility.visible:
            run_epochs = 0
            continue
        if run_epochs == 0:
            run_start = timestamp
        run_epochs += 1
        previous_timestamp = timestamp
        span = previous_timestamp - run_start
        if run_epochs > best_epochs or (run_epochs == best_epochs and span > best_span):
            best_epochs = run_epochs
            best_span = span
    return best_epochs, best_span
