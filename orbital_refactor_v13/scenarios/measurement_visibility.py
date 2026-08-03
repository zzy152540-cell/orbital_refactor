from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol

import numpy as np

from orbital_core.attitude import quat_to_dcm_i2b
from orbital_core.constants import R_EARTH

Array = np.ndarray


@dataclass(frozen=True)
class VisibilityConfig:
    """Physical geometry limits for one inter-satellite observation."""

    earth_occultation: bool = True
    earth_radius: float = R_EARTH
    earth_clearance: float = 0.0
    maximum_range: float | None = None
    field_of_view_half_angle: float | None = None
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.earth_radius <= 0.0:
            raise ValueError("earth_radius must be positive.")
        if self.earth_clearance < 0.0:
            raise ValueError("earth_clearance cannot be negative.")
        if self.maximum_range is not None and self.maximum_range <= 0.0:
            raise ValueError("maximum_range must be positive when provided.")
        if self.field_of_view_half_angle is not None and not (
            0.0 < self.field_of_view_half_angle <= np.pi
        ):
            raise ValueError(
                "field_of_view_half_angle must be in (0, pi] when provided."
            )
        if self.tolerance < 0.0:
            raise ValueError("tolerance cannot be negative.")


@dataclass(frozen=True)
class VisibilityTemporalFilterConfig:
    """Debounce and hysteresis applied after instantaneous geometry checks."""

    acquisition_epochs: int = 1
    loss_epochs: int = 1
    fov_hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if self.acquisition_epochs < 1 or self.loss_epochs < 1:
            raise ValueError("Temporal confirmation epochs must be at least one.")
        if self.fov_hysteresis < 0.0:
            raise ValueError("fov_hysteresis cannot be negative.")


@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    reason: str
    range: float
    earth_clearance: float
    off_boresight_angle: float | None = None


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
    visible_directed_edge_count_by_timestamp_and_modality: dict[
        tuple[float, str], int
    ]
    longest_unavailable_epochs_by_edge_and_modality: dict[tuple[str, str, str], int]
    longest_unavailable_span_by_edge_and_modality: dict[tuple[str, str, str], float]
    availability_switch_count_by_edge_and_modality: dict[tuple[str, str, str], int]


class CandidateTopology(Protocol):
    @property
    def node_ids(self) -> tuple[str, ...]: ...

    def neighbors(self, node_id: str) -> tuple[str, ...]: ...


def evaluate_inter_satellite_visibility(
    observer_state: Array,
    target_state: Array,
    config: VisibilityConfig | None = None,
    *,
    quaternion_i2b_wxyz: Array | None = None,
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
    off_boresight_angle = None
    if limits.field_of_view_half_angle is not None:
        if quaternion_i2b_wxyz is None:
            raise ValueError(
                "FOV visibility requires observer quaternion_i2b_wxyz."
            )
        relative_body = quat_to_dcm_i2b(quaternion_i2b_wxyz) @ relative
        cosine = float(np.clip(relative_body[0] / distance, -1.0, 1.0))
        off_boresight_angle = float(np.arccos(cosine))
        if off_boresight_angle > (
            limits.field_of_view_half_angle + limits.tolerance
        ):
            return VisibilityResult(
                False, "outside_fov", distance, clearance, off_boresight_angle
            )
    return VisibilityResult(
        True, "visible", distance, clearance, off_boresight_angle
    )


def generate_inter_satellite_observation_opportunities(
    *,
    timestamps: Array,
    truth_state_history_by_node: Mapping[str, Array],
    candidate_topology: CandidateTopology,
    visibility_by_modality: Mapping[str, VisibilityConfig],
    attitude_history_by_node: Mapping[str, Array] | None = None,
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
    attitude_histories = _attitude_histories(
        attitude_history_by_node, node_ids=node_ids, count=times.size,
    )
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
                        quaternion_i2b_wxyz=(
                            None if attitude_histories is None
                            else attitude_histories[observer_id][index]
                        ),
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
    visible_edges_by_time_and_modality: dict[
        tuple[float, str], set[tuple[str, str]]
    ] = {}
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
            visible_edges_by_time_and_modality.setdefault(
                (timestamp, item.modality), set()
            ).add(edge)
    longest_epochs = {}
    longest_spans = {}
    switch_counts = {}
    for key, values in timeline_groups.items():
        epochs, span = _longest_unavailable_run(values)
        longest_epochs[key] = epochs
        longest_spans[key] = span
        ordered_values = sorted(values, key=lambda item: item.timestamp)
        switch_counts[key] = sum(
            left.visibility.visible != right.visibility.visible
            for left, right in zip(ordered_values, ordered_values[1:])
        )
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
        visible_directed_edge_count_by_timestamp_and_modality={
            (timestamp, modality): len(
                visible_edges_by_time_and_modality.get((timestamp, modality), set())
            )
            for timestamp in sorted(all_timestamps)
            for modality in sorted(modality_groups)
        },
        longest_unavailable_epochs_by_edge_and_modality=longest_epochs,
        longest_unavailable_span_by_edge_and_modality=longest_spans,
        availability_switch_count_by_edge_and_modality=switch_counts,
    )


def stabilize_observation_opportunities(
    opportunities: tuple[MeasurementOpportunity, ...],
    *,
    visibility_by_modality: Mapping[str, VisibilityConfig],
    temporal_filter_by_modality: Mapping[str, VisibilityTemporalFilterConfig],
) -> tuple[MeasurementOpportunity, ...]:
    """Apply per-link FOV hysteresis and consecutive-epoch confirmation."""

    if not opportunities:
        raise ValueError("At least one measurement opportunity is required.")
    unknown = set(temporal_filter_by_modality) - set(visibility_by_modality)
    if unknown:
        raise ValueError(f"Temporal filters have unknown modalities: {sorted(unknown)}")
    groups: dict[tuple[str, str, str], list[MeasurementOpportunity]] = {}
    for item in opportunities:
        groups.setdefault(
            (item.observer_id, item.target_id, item.modality), []
        ).append(item)
    stabilized = []
    for (_, _, modality), values in sorted(groups.items()):
        limits = visibility_by_modality[modality]
        settings = temporal_filter_by_modality.get(
            modality, VisibilityTemporalFilterConfig()
        )
        if (
            settings.fov_hysteresis > 0.0
            and limits.field_of_view_half_angle is None
        ):
            raise ValueError("FOV hysteresis requires a modality FOV limit.")
        if (
            limits.field_of_view_half_angle is not None
            and settings.fov_hysteresis >= limits.field_of_view_half_angle
        ):
            raise ValueError("fov_hysteresis must be smaller than the FOV half-angle.")
        available = False
        acquisition_count = 0
        loss_count = 0
        for item in sorted(values, key=lambda value: value.timestamp):
            raw = item.visibility
            fov_eligible = raw.reason in {"visible", "outside_fov"}
            if not fov_eligible:
                available = False
                acquisition_count = 0
                loss_count = 0
                stabilized.append(item)
                continue
            candidate = raw.visible
            if limits.field_of_view_half_angle is not None:
                angle = raw.off_boresight_angle
                if angle is None:
                    raise ValueError("FOV temporal filtering requires off-boresight angle.")
                threshold = limits.field_of_view_half_angle + (
                    settings.fov_hysteresis if available
                    else -settings.fov_hysteresis
                )
                candidate = angle <= threshold + limits.tolerance
            if candidate:
                loss_count = 0
                acquisition_count += 1
                if not available and acquisition_count >= settings.acquisition_epochs:
                    available = True
                visibility = replace(
                    raw,
                    visible=available,
                    reason=("visible" if available else "acquisition_pending"),
                )
            else:
                acquisition_count = 0
                loss_count += 1
                if available and loss_count >= settings.loss_epochs:
                    available = False
                visibility = replace(
                    raw,
                    visible=available,
                    reason=("visible_temporal_hold" if available else "outside_fov"),
                )
            stabilized.append(replace(item, visibility=visibility))
    return tuple(sorted(
        stabilized,
        key=lambda item: (
            item.timestamp, item.observer_id, item.target_id, item.modality,
        ),
    ))


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


def _attitude_histories(histories, *, node_ids, count):
    if histories is None:
        return None
    if set(histories) != set(node_ids):
        raise ValueError("Attitude-history keys must match candidate topology nodes.")
    result = {}
    for node_id in node_ids:
        values = np.asarray(histories[node_id], dtype=float)
        if values.shape != (count, 4) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Attitude history for {node_id} must have shape ({count}, 4) "
                "and contain finite values."
            )
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms < 1e-15):
            raise ValueError("Attitude quaternions must have nonzero norm.")
        result[node_id] = values / norms[:, None]
    return result


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
