"""Persistent local-to-cooperative target identity reconciliation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from cooperative.temporal_alignment import propagate_state_covariance

from .data_contracts import GlobalTargetEstimate, TargetNodeReport
from .target_fusion import fuse_target_reports


@dataclass(frozen=True)
class TargetIdentityAssignment:
    observer_id: str
    local_target_id: str
    local_track_id: str
    cooperative_target_id: str
    cooperative_track_id: str


@dataclass(frozen=True)
class TargetIdentityResolution:
    reports: tuple[TargetNodeReport, ...]
    assignments: tuple[TargetIdentityAssignment, ...]
    created_cooperative_target_ids: tuple[str, ...]
    reacquired_cooperative_target_ids: tuple[str, ...] = ()


@dataclass
class _CanonicalTrack:
    target_id: str
    track_id: str
    timestamp: float
    state_eci: np.ndarray
    covariance_eci: np.ndarray


class CooperativeTrackIdentityResolver:
    """Map independent node-local tracks onto persistent cooperative tracks."""

    def __init__(
        self,
        *,
        position_gate_m: float = 100_000.0,
        velocity_gate_mps: float = 500.0,
        process_noise_acceleration: float = 1e-4,
        minimum_match_margin: float = 0.05,
        max_reacquisition_gap_seconds: float = np.inf,
        target_prefix: str = "target-coop-",
        track_prefix: str = "track-coop-",
    ) -> None:
        if position_gate_m <= 0.0 or velocity_gate_mps <= 0.0:
            raise ValueError("Identity gates must be positive.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration cannot be negative.")
        if minimum_match_margin < 0.0:
            raise ValueError("minimum_match_margin cannot be negative.")
        if max_reacquisition_gap_seconds <= 0.0:
            raise ValueError("max_reacquisition_gap_seconds must be positive.")
        self.position_gate_m = float(position_gate_m)
        self.velocity_gate_mps = float(velocity_gate_mps)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.minimum_match_margin = float(minimum_match_margin)
        self.max_reacquisition_gap_seconds = float(
            max_reacquisition_gap_seconds
        )
        self.target_prefix = str(target_prefix)
        self.track_prefix = str(track_prefix)
        self._binding_by_local_key: dict[
            tuple[str, str, str], tuple[str, str]
        ] = {}
        self._canonical_by_target: dict[str, _CanonicalTrack] = {}
        self._last_seen_by_local_key: dict[tuple[str, str, str], float] = {}
        self._next_index = 1

    @property
    def cooperative_target_ids(self) -> tuple[str, ...]:
        """Return every persistent cooperative identity, including dormant ones."""

        return tuple(sorted(self._canonical_by_target))

    def resolve(
        self,
        reports: Iterable[TargetNodeReport],
    ) -> TargetIdentityResolution:
        values = tuple(sorted(
            reports,
            key=lambda item: (item.timestamp, item.observer_id, item.target_id),
        ))
        if not values:
            return TargetIdentityResolution((), (), ())
        timestamps = {float(item.timestamp) for item in values}
        if len(timestamps) != 1:
            raise ValueError("Identity resolution requires one aligned epoch.")
        timestamp = next(iter(timestamps))
        occupied_by_node: dict[str, set[str]] = defaultdict(set)
        canonical_reports = []
        assignments = []
        created = []
        reacquired = []
        canonical_before_epoch = set(self._canonical_by_target)

        reports_by_node: dict[str, list[TargetNodeReport]] = defaultdict(list)
        for report in values:
            reports_by_node[report.observer_id].append(report)
        for observer_id, node_reports in reports_by_node.items():
            identities: dict[int, tuple[str, str]] = {}
            unresolved = []
            for index, report in enumerate(node_reports):
                local_key = (observer_id, report.target_id, report.track_id)
                identity = self._binding_by_local_key.get(local_key)
                if identity is None:
                    unresolved.append((index, report))
                    continue
                if identity[0] in occupied_by_node[observer_id]:
                    raise ValueError(
                        "One node cannot bind two simultaneous local tracks to one "
                        "cooperative target."
                    )
                identities[index] = identity
                occupied_by_node[observer_id].add(identity[0])

            matches = self._match_jointly(
                [report for _, report in unresolved],
                timestamp=timestamp,
                excluded_target_ids=occupied_by_node[observer_id],
            )
            for (index, report), identity in zip(unresolved, matches):
                if identity is None:
                    identity = self._create_identity()
                identities[index] = identity
                local_key = (observer_id, report.target_id, report.track_id)
                self._binding_by_local_key[local_key] = identity
                if identity[0] not in self._canonical_by_target:
                    self._canonical_by_target[identity[0]] = _CanonicalTrack(
                        target_id=identity[0],
                        track_id=identity[1],
                        timestamp=timestamp,
                        state_eci=report.state_eci.copy(),
                        covariance_eci=report.covariance_eci.copy(),
                    )
                    created.append(identity[0])
                elif identity[0] in canonical_before_epoch:
                    reacquired.append(identity[0])
                occupied_by_node[observer_id].add(identity[0])

            for index, report in enumerate(node_reports):
                identity = identities[index]
                local_key = (observer_id, report.target_id, report.track_id)
                self._last_seen_by_local_key[local_key] = timestamp
                canonical_reports.append(replace(
                    report,
                    target_id=identity[0],
                    track_id=identity[1],
                ))
                assignments.append(TargetIdentityAssignment(
                    observer_id=observer_id,
                    local_target_id=report.target_id,
                    local_track_id=report.track_id,
                    cooperative_target_id=identity[0],
                    cooperative_track_id=identity[1],
                ))

        self._refresh_canonical_tracks(canonical_reports)
        return TargetIdentityResolution(
            reports=tuple(canonical_reports),
            assignments=tuple(assignments),
            created_cooperative_target_ids=tuple(created),
            reacquired_cooperative_target_ids=tuple(dict.fromkeys(reacquired)),
        )

    def retire_local_track(
        self,
        *,
        observer_id: str,
        local_target_id: str,
        local_track_id: str,
    ) -> bool:
        key = (str(observer_id), str(local_target_id), str(local_track_id))
        removed = self._binding_by_local_key.pop(key, None) is not None
        self._last_seen_by_local_key.pop(key, None)
        return removed

    def remove_observer(self, observer_id: str) -> tuple[str, ...]:
        """Release one departed observer without deleting global identities."""

        node_id = str(observer_id)
        keys = [
            key for key in self._binding_by_local_key
            if key[0] == node_id
        ]
        released = []
        for key in keys:
            cooperative_target_id, _ = self._binding_by_local_key.pop(key)
            self._last_seen_by_local_key.pop(key, None)
            released.append(cooperative_target_id)
        return tuple(sorted(set(released)))

    def observer_ids_for_target(self, cooperative_target_id: str) -> tuple[str, ...]:
        """Return nodes currently bound to one persistent global identity."""

        target_id = str(cooperative_target_id)
        return tuple(sorted({
            observer_id
            for (observer_id, _local_target_id, _local_track_id), identity
            in self._binding_by_local_key.items()
            if identity[0] == target_id
        }))

    def localize_estimates(
        self,
        *,
        observer_id: str,
        estimates: Mapping[str, GlobalTargetEstimate],
    ) -> dict[str, GlobalTargetEstimate]:
        candidates: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        for key, (cooperative_target_id, _cooperative_track_id) in (
            self._binding_by_local_key.items()
        ):
            node_id, local_target_id, local_track_id = key
            if node_id != observer_id:
                continue
            candidates[cooperative_target_id].append((
                self._last_seen_by_local_key.get(key, -np.inf),
                local_target_id,
                local_track_id,
            ))
        reverse = {
            cooperative_target_id: (local_target_id, local_track_id)
            for cooperative_target_id, values in candidates.items()
            for _, local_target_id, local_track_id in [max(values)]
        }
        localized = {}
        for cooperative_target_id, estimate in estimates.items():
            local_identity = reverse.get(cooperative_target_id)
            if local_identity is None:
                continue
            local_target_id, local_track_id = local_identity
            localized[local_target_id] = replace(
                estimate,
                target_id=local_target_id,
                track_id=local_track_id,
            )
        return localized

    def _match_jointly(
        self,
        reports: list[TargetNodeReport],
        *,
        timestamp: float,
        excluded_target_ids: set[str],
    ) -> list[tuple[str, str] | None]:
        if not reports:
            return []
        candidates = [
            track for target_id, track in sorted(self._canonical_by_target.items())
            if target_id not in excluded_target_ids
            and timestamp - track.timestamp <= self.max_reacquisition_gap_seconds
        ]
        if not candidates:
            return [None] * len(reports)
        predicted = []
        for track in candidates:
            state = track.state_eci
            if timestamp > track.timestamp:
                state, _ = propagate_state_covariance(
                    track.state_eci,
                    track.covariance_eci,
                    timestamp - track.timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
            predicted.append(state)
        costs = np.full((len(reports), len(candidates)), np.inf)
        for row, report in enumerate(reports):
            for column, state in enumerate(predicted):
                position_error = float(np.linalg.norm(
                    report.state_eci[:3] - state[:3]
                ))
                velocity_error = float(np.linalg.norm(
                    report.state_eci[3:] - state[3:]
                ))
                if (
                    position_error <= self.position_gate_m
                    and velocity_error <= self.velocity_gate_mps
                ):
                    costs[row, column] = (
                        position_error / self.position_gate_m
                        + velocity_error / self.velocity_gate_mps
                    )
        safe = np.where(np.isfinite(costs), costs, 1e12)
        rows, columns = linear_sum_assignment(safe)
        matches: list[tuple[str, str] | None] = [None] * len(reports)
        for row, column in zip(rows, columns):
            score = float(costs[row, column])
            if not np.isfinite(score):
                continue
            alternatives = np.delete(costs[row], column)
            finite_alternatives = alternatives[np.isfinite(alternatives)]
            margin = (
                np.inf
                if finite_alternatives.size == 0
                else float(np.min(finite_alternatives) - score)
            )
            if margin < self.minimum_match_margin:
                continue
            track = candidates[column]
            matches[row] = (track.target_id, track.track_id)
        return matches

    def _create_identity(self) -> tuple[str, str]:
        index = self._next_index
        self._next_index += 1
        return (
            f"{self.target_prefix}{index:04d}",
            f"{self.track_prefix}{index:04d}",
        )

    def _refresh_canonical_tracks(
        self,
        reports: Iterable[TargetNodeReport],
    ) -> None:
        grouped: dict[str, list[TargetNodeReport]] = defaultdict(list)
        for report in reports:
            grouped[report.target_id].append(report)
        for target_id, values in grouped.items():
            estimate = fuse_target_reports(values)
            self._canonical_by_target[target_id] = _CanonicalTrack(
                target_id=target_id,
                track_id=estimate.track_id,
                timestamp=estimate.timestamp,
                state_eci=estimate.state_eci.copy(),
                covariance_eci=estimate.covariance_eci.copy(),
            )
