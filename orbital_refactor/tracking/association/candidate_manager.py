from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.initial_orbit.data_contracts import IODObservation
from tracking.initial_orbit.multi_target_initializer import MultiTargetIODManager, MultiTargetIODUpdate

from .data_contracts import UnlabeledIODObservation


@dataclass
class _Candidate:
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float


class AutonomousIODManager:
    """Build internal target identities from locally correlated LOS/radar detections."""

    def __init__(
        self,
        *,
        spatial_gate_m: float = 100_000.0,
        candidate_timeout_seconds: float = 120.0,
        iod_manager=None,
    ):
        if spatial_gate_m <= 0.0:
            raise ValueError("spatial_gate_m must be positive.")
        if candidate_timeout_seconds <= 0.0:
            raise ValueError("candidate_timeout_seconds must be positive.")
        self.spatial_gate_m = float(spatial_gate_m)
        self.candidate_timeout_seconds = float(candidate_timeout_seconds)
        self.iod_manager = iod_manager or MultiTargetIODManager()
        self._candidates: dict[str, _Candidate] = {}
        self._next_identifier = 1

    @property
    def candidate_ids(self):
        return tuple(sorted(self._candidates))

    def ingest(self, observations) -> MultiTargetIODUpdate:
        values = tuple(observations)
        if any(not isinstance(item, UnlabeledIODObservation) for item in values):
            raise TypeError("AutonomousIODManager requires unlabeled IOD observations.")
        labeled = []
        by_epoch = defaultdict(list)
        for item in values:
            by_epoch[item.timestamp].append(item)
        for timestamp in sorted(by_epoch):
            self._expire_candidates(timestamp)
            labeled.extend(self._label_epoch(tuple(by_epoch[timestamp])))
        return self.iod_manager.ingest(labeled)

    def retire_target(self, target_id: str) -> None:
        key = str(target_id)
        self._candidates.pop(key, None)
        self.iod_manager.retire(key)

    def _expire_candidates(self, timestamp):
        expired = [
            target_id for target_id, candidate in self._candidates.items()
            if timestamp - candidate.timestamp > self.candidate_timeout_seconds
        ]
        for target_id in expired:
            self.retire_target(target_id)

    def _label_epoch(self, observations):
        bundles = _make_bundles(observations, self.spatial_gate_m)
        if not bundles:
            return ()
        target_ids = tuple(sorted(self._candidates))
        assignments = {}
        if target_ids:
            costs = np.empty((len(bundles), len(target_ids)))
            for row, bundle in enumerate(bundles):
                for column, target_id in enumerate(target_ids):
                    candidate = self._candidates[target_id]
                    dt = bundle["timestamp"] - candidate.timestamp
                    prediction = candidate.position + max(dt, 0.0) * candidate.velocity
                    costs[row, column] = np.linalg.norm(bundle["position"] - prediction)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] <= self.spatial_gate_m:
                    assignments[row] = target_ids[column]
        for row in range(len(bundles)):
            if row not in assignments:
                assignments[row] = f"target-auto-{self._next_identifier:04d}"
                self._next_identifier += 1

        labeled = []
        for row, bundle in enumerate(bundles):
            target_id = assignments[row]
            previous = self._candidates.get(target_id)
            velocity = np.zeros(3)
            if previous is not None and bundle["timestamp"] > previous.timestamp:
                dt = bundle["timestamp"] - previous.timestamp
                velocity = (bundle["position"] - previous.position) / dt
            self._candidates[target_id] = _Candidate(
                target_id, bundle["position"], velocity, bundle["timestamp"],
            )
            for item in bundle["observations"]:
                labeled.append(IODObservation(
                    timestamp=item.timestamp, observer_id=item.observer_id,
                    target_id=target_id, modality=item.modality,
                    observer_state_eci=item.observer_state_eci,
                    measurement=item.measurement, covariance=item.covariance,
                    valid_flag=item.valid_flag,
                    metadata={**item.metadata, "associationMethod": "AUTONOMOUS_SHORT_ARC"},
                ))
        return tuple(labeled)


def _make_bundles(observations, clustering_gate):
    local = defaultdict(list)
    for index, item in enumerate(observations):
        if not item.valid_flag or item.detection_group_id is None:
            continue
        local[(item.observer_id, item.detection_group_id)].append((index, item))
    seeds = []
    for members in local.values():
        radar = next((item for _, item in members if item.modality == "RADAR"), None)
        los = next((item for _, item in members if item.modality == "LOS"), None)
        if radar is None or los is None:
            continue
        position = radar.observer_state_eci[:3] + radar.measurement[0] * los.measurement
        seeds.append({
            "position": position,
            "timestamp": radar.timestamp,
            "observations": [item for _, item in members],
        })
    clusters = []
    for seed in seeds:
        match = next((
            value for value in clusters
            if np.linalg.norm(seed["position"] - value["position"]) <= clustering_gate
        ), None)
        if match is None:
            clusters.append(seed)
        else:
            count = len(match["observations"])
            added = len(seed["observations"])
            match["position"] = (
                count * match["position"] + added * seed["position"]
            ) / (count + added)
            match["observations"].extend(seed["observations"])
    return clusters

