"""Persistent local-to-cooperative target identity reconciliation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

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
        target_prefix: str = "target-coop-",
        track_prefix: str = "track-coop-",
    ) -> None:
        if position_gate_m <= 0.0 or velocity_gate_mps <= 0.0:
            raise ValueError("Identity gates must be positive.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration cannot be negative.")
        self.position_gate_m = float(position_gate_m)
        self.velocity_gate_mps = float(velocity_gate_mps)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.target_prefix = str(target_prefix)
        self.track_prefix = str(track_prefix)
        self._binding_by_local_key: dict[
            tuple[str, str, str], tuple[str, str]
        ] = {}
        self._canonical_by_target: dict[str, _CanonicalTrack] = {}
        self._last_seen_by_local_key: dict[tuple[str, str, str], float] = {}
        self._next_index = 1

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

        for report in values:
            local_key = (report.observer_id, report.target_id, report.track_id)
            identity = self._binding_by_local_key.get(local_key)
            if identity is None:
                identity = self._match_or_create(
                    report,
                    timestamp=timestamp,
                    excluded_target_ids=occupied_by_node[report.observer_id],
                )
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
            if identity[0] in occupied_by_node[report.observer_id]:
                raise ValueError(
                    "One node cannot bind two simultaneous local tracks to one "
                    "cooperative target."
                )
            occupied_by_node[report.observer_id].add(identity[0])
            self._last_seen_by_local_key[local_key] = timestamp
            canonical_reports.append(replace(
                report,
                target_id=identity[0],
                track_id=identity[1],
            ))
            assignments.append(TargetIdentityAssignment(
                observer_id=report.observer_id,
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

    def _match_or_create(
        self,
        report: TargetNodeReport,
        *,
        timestamp: float,
        excluded_target_ids: set[str],
    ) -> tuple[str, str]:
        candidates = []
        for target_id, track in self._canonical_by_target.items():
            if target_id in excluded_target_ids:
                continue
            state = track.state_eci
            if timestamp > track.timestamp:
                state, _ = propagate_state_covariance(
                    track.state_eci,
                    track.covariance_eci,
                    timestamp - track.timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
            position_error = float(np.linalg.norm(report.state_eci[:3] - state[:3]))
            velocity_error = float(np.linalg.norm(report.state_eci[3:] - state[3:]))
            if (
                position_error <= self.position_gate_m
                and velocity_error <= self.velocity_gate_mps
            ):
                score = (
                    position_error / self.position_gate_m
                    + velocity_error / self.velocity_gate_mps
                )
                candidates.append((score, target_id, track.track_id))
        if candidates:
            _, target_id, track_id = min(candidates)
            return target_id, track_id
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
