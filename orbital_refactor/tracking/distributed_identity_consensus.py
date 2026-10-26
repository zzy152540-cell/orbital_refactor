"""Neighbor-only reconciliation of independently created local track identities.

The implementation is intentionally deployment agnostic.  One object emulates
the private identity registry held by every observer, but information moves
between registries only across edges in ``NetworkTopology``.  This makes it
useful for deterministic single-process validation of a logically distributed
discovery and initialization pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from cooperative.temporal_alignment import propagate_state_covariance
from cooperative.topology import NetworkTopology

from .data_contracts import GlobalTargetEstimate, TargetNodeReport
from .distributed_identity import (
    TargetIdentityAssignment,
    TargetIdentityResolution,
)


@dataclass
class _TrackDescriptor:
    target_id: str
    track_id: str
    timestamp: float
    state_eci: np.ndarray
    covariance_eci: np.ndarray

    @classmethod
    def from_report(cls, report: TargetNodeReport) -> "_TrackDescriptor":
        return cls(
            target_id=report.target_id,
            track_id=report.track_id,
            timestamp=float(report.timestamp),
            state_eci=report.state_eci.copy(),
            covariance_eci=report.covariance_eci.copy(),
        )

    def copy(self) -> "_TrackDescriptor":
        return _TrackDescriptor(
            target_id=self.target_id,
            track_id=self.track_id,
            timestamp=self.timestamp,
            state_eci=self.state_eci.copy(),
            covariance_eci=self.covariance_eci.copy(),
        )


@dataclass
class _NodeRegistry:
    bindings: dict[tuple[str, str], str]
    aliases: dict[str, str]
    tracks: dict[str, _TrackDescriptor]


class NeighborTrackIdentityConsensus:
    """Reconcile local track IDs through repeated neighbor announcements.

    Each observer creates identities from its own local tracks.  Registries
    exchange only target-state summaries with current topology neighbors.  A
    deterministic minimum-ID rule makes independently discovered aliases
    converge without a central identity allocator.
    """

    def __init__(
        self,
        *,
        topology: NetworkTopology,
        position_gate_m: float = 100_000.0,
        velocity_gate_mps: float = 500.0,
        process_noise_acceleration: float = 1e-4,
        minimum_match_margin: float = 0.05,
        exchange_hops: int = 1,
    ) -> None:
        if position_gate_m <= 0.0 or velocity_gate_mps <= 0.0:
            raise ValueError("Identity gates must be positive.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration cannot be negative.")
        if minimum_match_margin < 0.0:
            raise ValueError("minimum_match_margin cannot be negative.")
        if int(exchange_hops) < 1:
            raise ValueError("exchange_hops must be positive.")
        self.topology = topology
        self.position_gate_m = float(position_gate_m)
        self.velocity_gate_mps = float(velocity_gate_mps)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.minimum_match_margin = float(minimum_match_margin)
        self.exchange_hops = int(exchange_hops)
        self._registries = {
            node_id: _NodeRegistry({}, {}, {})
            for node_id in topology.node_ids
        }

    def update_topology(self, topology: NetworkTopology) -> None:
        """Update membership while retaining registries for surviving nodes."""

        retained = {
            node_id: self._registries.get(
                node_id, _NodeRegistry({}, {}, {}),
            )
            for node_id in topology.node_ids
        }
        self._registries = retained
        self.topology = topology

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
        unknown = {
            report.observer_id for report in values
            if report.observer_id not in self._registries
        }
        if unknown:
            raise ValueError(
                f"Reports reference nodes absent from topology: {sorted(unknown)}"
            )

        created = []
        reacquired = []
        reports_by_node: dict[str, list[TargetNodeReport]] = defaultdict(list)
        for report in values:
            reports_by_node[report.observer_id].append(report)
        for node_id, node_reports in reports_by_node.items():
            node_created, node_reacquired = self._bind_local_reports(
                node_id, node_reports,
            )
            created.extend(node_created)
            reacquired.extend(node_reacquired)

        for _ in range(self.exchange_hops):
            outbound = {
                node_id: self._announcements(node_id)
                for node_id in self.topology.node_ids
            }
            for receiver_id in sorted(self.topology.node_ids):
                for source_id in sorted(self.topology.neighbors(receiver_id)):
                    self._receive(
                        receiver_id,
                        outbound[source_id],
                        timestamp=timestamp,
                    )

        canonical_reports = []
        assignments = []
        for report in values:
            registry = self._registries[report.observer_id]
            local_key = (report.target_id, report.track_id)
            cooperative_id = self._root(
                registry, registry.bindings[local_key],
            )
            registry.bindings[local_key] = cooperative_id
            cooperative_track_id = self._track_id(cooperative_id)
            canonical_reports.append(replace(
                report,
                target_id=cooperative_id,
                track_id=cooperative_track_id,
            ))
            assignments.append(TargetIdentityAssignment(
                observer_id=report.observer_id,
                local_target_id=report.target_id,
                local_track_id=report.track_id,
                cooperative_target_id=cooperative_id,
                cooperative_track_id=cooperative_track_id,
            ))
        created_roots = tuple(sorted({
            self._root(self._registries[node_id], identity)
            for node_id, identity in created
        }))
        reacquired_roots = tuple(sorted({
            self._root(self._registries[node_id], identity)
            for node_id, identity in reacquired
        }))
        return TargetIdentityResolution(
            reports=tuple(canonical_reports),
            assignments=tuple(assignments),
            created_cooperative_target_ids=created_roots,
            reacquired_cooperative_target_ids=reacquired_roots,
        )

    def localize_estimates(
        self,
        *,
        observer_id: str,
        estimates: Mapping[str, GlobalTargetEstimate],
    ) -> dict[str, GlobalTargetEstimate]:
        registry = self._registries[str(observer_id)]
        local_by_cooperative = {}
        for (local_target_id, local_track_id), identity in registry.bindings.items():
            root = self._root(registry, identity)
            local_by_cooperative.setdefault(
                root, (local_target_id, local_track_id),
            )
        localized = {}
        for cooperative_id, estimate in estimates.items():
            root = self._root(registry, cooperative_id)
            local_identity = local_by_cooperative.get(root)
            if local_identity is None:
                continue
            local_target_id, local_track_id = local_identity
            localized[local_target_id] = replace(
                estimate,
                target_id=local_target_id,
                track_id=local_track_id,
            )
        return localized

    def retire_local_track(
        self,
        *,
        observer_id: str,
        local_target_id: str,
        local_track_id: str,
    ) -> bool:
        registry = self._registries[str(observer_id)]
        return registry.bindings.pop(
            (str(local_target_id), str(local_track_id)), None,
        ) is not None

    def remove_observer(self, observer_id: str) -> tuple[str, ...]:
        node_id = str(observer_id)
        registry = self._registries.get(node_id)
        if registry is None:
            return ()
        released = tuple(sorted({
            self._root(registry, identity)
            for identity in registry.bindings.values()
        }))
        return released

    def cooperative_target_ids_by_node(self) -> Mapping[str, tuple[str, ...]]:
        return {
            node_id: tuple(sorted({
                self._root(registry, target_id)
                for target_id in registry.tracks
            }))
            for node_id, registry in self._registries.items()
        }

    def _bind_local_reports(
        self,
        node_id: str,
        reports: list[TargetNodeReport],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        registry = self._registries[node_id]
        occupied = {
            self._root(registry, identity)
            for key, identity in registry.bindings.items()
            if key in {(item.target_id, item.track_id) for item in reports}
        }
        created = []
        reacquired = []
        for report in reports:
            local_key = (report.target_id, report.track_id)
            identity = registry.bindings.get(local_key)
            if identity is None:
                identity = self._best_match(
                    registry,
                    _TrackDescriptor.from_report(report),
                    excluded=occupied,
                )
                if identity is None:
                    identity = self._origin_identity(node_id, report)
                    created.append((node_id, identity))
                else:
                    reacquired.append((node_id, identity))
                registry.bindings[local_key] = identity
            root = self._root(registry, identity)
            occupied.add(root)
            canonical = replace(
                report, target_id=root, track_id=self._track_id(root),
            )
            self._store_descriptor(
                registry, _TrackDescriptor.from_report(canonical),
            )
        return created, reacquired

    def _announcements(self, node_id: str) -> tuple[_TrackDescriptor, ...]:
        registry = self._registries[node_id]
        normalized = {}
        for identity, descriptor in registry.tracks.items():
            root = self._root(registry, identity)
            candidate = descriptor.copy()
            candidate.target_id = root
            candidate.track_id = self._track_id(root)
            previous = normalized.get(root)
            if previous is None or self._prefer(candidate, previous):
                normalized[root] = candidate
        return tuple(normalized[key] for key in sorted(normalized))

    def _receive(
        self,
        receiver_id: str,
        announcements: tuple[_TrackDescriptor, ...],
        *,
        timestamp: float,
    ) -> None:
        registry = self._registries[receiver_id]
        for incoming in announcements:
            incoming_id = self._root(registry, incoming.target_id)
            if incoming_id in registry.tracks:
                self._store_descriptor(registry, incoming)
                continue
            match = self._best_match(registry, incoming, excluded=set())
            if match is not None:
                incoming_id = self._union(registry, incoming_id, match)
            descriptor = incoming.copy()
            if descriptor.timestamp < timestamp:
                state, covariance = propagate_state_covariance(
                    descriptor.state_eci,
                    descriptor.covariance_eci,
                    timestamp - descriptor.timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
                descriptor.timestamp = timestamp
                descriptor.state_eci = state
                descriptor.covariance_eci = covariance
            descriptor.target_id = incoming_id
            descriptor.track_id = self._track_id(incoming_id)
            self._store_descriptor(registry, descriptor)

    def _best_match(
        self,
        registry: _NodeRegistry,
        incoming: _TrackDescriptor,
        *,
        excluded: set[str],
    ) -> str | None:
        candidates = []
        seen = set()
        for identity, descriptor in registry.tracks.items():
            root = self._root(registry, identity)
            if root in excluded or root in seen:
                continue
            seen.add(root)
            state = descriptor.state_eci
            if incoming.timestamp > descriptor.timestamp:
                state, _ = propagate_state_covariance(
                    descriptor.state_eci,
                    descriptor.covariance_eci,
                    incoming.timestamp - descriptor.timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
            elif descriptor.timestamp > incoming.timestamp:
                incoming_state, _ = propagate_state_covariance(
                    incoming.state_eci,
                    incoming.covariance_eci,
                    descriptor.timestamp - incoming.timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
                state = descriptor.state_eci
                position_error = float(np.linalg.norm(incoming_state[:3] - state[:3]))
                velocity_error = float(np.linalg.norm(incoming_state[3:] - state[3:]))
                if position_error <= self.position_gate_m and velocity_error <= self.velocity_gate_mps:
                    candidates.append((
                        position_error / self.position_gate_m
                        + velocity_error / self.velocity_gate_mps,
                        root,
                    ))
                continue
            position_error = float(np.linalg.norm(incoming.state_eci[:3] - state[:3]))
            velocity_error = float(np.linalg.norm(incoming.state_eci[3:] - state[3:]))
            if position_error <= self.position_gate_m and velocity_error <= self.velocity_gate_mps:
                candidates.append((
                    position_error / self.position_gate_m
                    + velocity_error / self.velocity_gate_mps,
                    root,
                ))
        if not candidates:
            return None
        candidates.sort()
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] < self.minimum_match_margin
        ):
            return None
        return candidates[0][1]

    def _store_descriptor(
        self,
        registry: _NodeRegistry,
        descriptor: _TrackDescriptor,
    ) -> None:
        root = self._root(registry, descriptor.target_id)
        descriptor = descriptor.copy()
        descriptor.target_id = root
        descriptor.track_id = self._track_id(root)
        previous = registry.tracks.get(root)
        if previous is None or self._prefer(descriptor, previous):
            registry.tracks[root] = descriptor

    @staticmethod
    def _prefer(candidate: _TrackDescriptor, previous: _TrackDescriptor) -> bool:
        if candidate.timestamp != previous.timestamp:
            return candidate.timestamp > previous.timestamp
        return float(np.trace(candidate.covariance_eci)) < float(
            np.trace(previous.covariance_eci)
        )

    def _union(self, registry: _NodeRegistry, first: str, second: str) -> str:
        first_root = self._root(registry, first)
        second_root = self._root(registry, second)
        winner = min(first_root, second_root)
        loser = max(first_root, second_root)
        if winner == loser:
            return winner
        registry.aliases[loser] = winner
        for key, identity in tuple(registry.bindings.items()):
            if self._root(registry, identity) == winner:
                registry.bindings[key] = winner
        winner_descriptor = registry.tracks.get(winner)
        loser_descriptor = registry.tracks.pop(loser, None)
        if loser_descriptor is not None and (
            winner_descriptor is None
            or self._prefer(loser_descriptor, winner_descriptor)
        ):
            loser_descriptor.target_id = winner
            loser_descriptor.track_id = self._track_id(winner)
            registry.tracks[winner] = loser_descriptor
        return winner

    @staticmethod
    def _root(registry: _NodeRegistry, identity: str) -> str:
        current = str(identity)
        trail = []
        while current in registry.aliases:
            trail.append(current)
            current = registry.aliases[current]
        for value in trail:
            registry.aliases[value] = current
        return current

    @staticmethod
    def _origin_identity(node_id: str, report: TargetNodeReport) -> str:
        return f"target-coop-{node_id}-{report.target_id}-{report.track_id}"

    @staticmethod
    def _track_id(target_id: str) -> str:
        return f"track-{target_id}"
