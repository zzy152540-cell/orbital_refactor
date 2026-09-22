"""Load the supplied 6 functional + 25 background J2000 trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from orbital_core.coordinates import build_rtn_quaternion, state_eci_to_spri


@dataclass(frozen=True)
class ProjectTrajectoryDataset:
    scene_id: str
    start_time_ms: int
    timestamps: np.ndarray
    state_history_by_node: Mapping[str, np.ndarray]
    numeric_id_by_node: Mapping[str, int]
    functional_nodes: tuple[str, ...]
    background_nodes: tuple[str, ...]

    def nearest_background_observers(self) -> dict[str, str]:
        result = {}
        for target in self.functional_nodes:
            target_position = self.state_history_by_node[target][0, :3]
            result[target] = min(
                self.background_nodes,
                key=lambda node: float(np.linalg.norm(
                    self.state_history_by_node[node][0, :3] - target_position
                )),
            )
        return result

    def maximum_dynamics_velocity_residuals(self) -> dict[str, float]:
        """Return maximum one-second two-body+J2 velocity residual per node."""

        residuals = {}
        for node, history in self.state_history_by_node.items():
            if len(history) < 2:
                residuals[node] = 0.0
                continue
            differences = np.vstack([
                rk4_step_absolute(history[index], 1.0)[3:] - history[index + 1, 3:]
                for index in range(len(history) - 1)
            ])
            residuals[node] = float(np.max(np.linalg.norm(differences, axis=1)))
        return residuals

    def nonmaneuver_nodes(self, *, maximum_velocity_residual_mps=0.1) -> tuple[str, ...]:
        residuals = self.maximum_dynamics_velocity_residuals()
        return tuple(
            node for node in self.state_history_by_node
            if residuals[node] <= float(maximum_velocity_residual_mps)
        )

    def nearest_observers(
        self, targets, *, candidates=None, minimum_cross_track_fraction=0.0,
    ) -> dict[str, str]:
        candidate_nodes = tuple(
            self.state_history_by_node if candidates is None else candidates
        )
        result = {}
        for target in targets:
            available = tuple(node for node in candidate_nodes if node != target)
            if minimum_cross_track_fraction > 0.0:
                conditioned = []
                for node in available:
                    observer_history = self.state_history_by_node[node]
                    relative_history = self.state_history_by_node[target] - observer_history
                    relative_spri = np.vstack([
                        state_eci_to_spri(relative, build_rtn_quaternion(observer_state))
                        for relative, observer_state in zip(relative_history, observer_history)
                    ])
                    distance = np.linalg.norm(relative_spri[:, :3], axis=1)
                    cross_track_fraction = np.abs(relative_spri[:, 2]) / np.maximum(
                        distance, 1e-12,
                    )
                    positive_depth_fraction = float(np.mean(relative_spri[:, 2] > 0.0))
                    minimum_fraction = float(np.min(cross_track_fraction))
                    if minimum_fraction >= float(minimum_cross_track_fraction):
                        conditioned.append((node, positive_depth_fraction, minimum_fraction))
                if conditioned:
                    best_positive_fraction = max(item[1] for item in conditioned)
                    conditioned = [item for item in conditioned if item[1] == best_positive_fraction]
                    best_minimum_fraction = max(item[2] for item in conditioned)
                    available = tuple(
                        item[0] for item in conditioned if item[2] == best_minimum_fraction
                    )
                else:
                    available = ()
            if not available:
                raise ValueError(f"No observer candidate is available for {target}.")
            target_position = self.state_history_by_node[target][0, :3]
            result[target] = min(
                available,
                key=lambda node: float(np.linalg.norm(
                    self.state_history_by_node[node][0, :3] - target_position
                )),
            )
        return result


def load_project_trajectory_dataset(
    background_path: str | Path,
    functional_path: str | Path,
) -> ProjectTrajectoryDataset:
    background = _read_json(background_path)
    functional = _read_json(functional_path)
    if not isinstance(functional, list) or not functional:
        raise ValueError("Functional trajectory root must be a nonempty array.")
    frame_times = np.array([_time_ms(frame["time"]) for frame in functional], dtype=np.int64)
    if np.any(np.diff(frame_times) != 1000):
        raise ValueError("Functional trajectory must use contiguous one-second frames.")
    start_ms = int(frame_times[0])
    functional_nodes = tuple(item["satObj"] for item in functional[0]["satelliteList"])
    functional_histories = {
        node: np.vstack([
            _state(next(item for item in frame["satelliteList"] if item["satObj"] == node)["position"])
            for frame in functional
        ])
        for node in functional_nodes
    }
    numeric_ids = {
        item["satObj"]: int(item["satId"])
        for item in functional[0]["satelliteList"]
    }

    background_entries = background.get("satellites", [])
    background_nodes = tuple(str(item["name"]) for item in background_entries)
    background_histories = {}
    for item in background_entries:
        points = item.get("points", [])
        times = np.array([_time_ms(point["time"]) for point in points], dtype=np.int64)
        if not np.array_equal(times, frame_times):
            raise ValueError(f"Background trajectory time mismatch for {item.get('name')}.")
        node = str(item["name"])
        background_histories[node] = np.vstack([_state(point) for point in points])
        numeric_ids[node] = int(item["norad"])
    histories = {**functional_histories, **background_histories}
    if len(functional_nodes) != 6 or len(background_nodes) != 25 or len(histories) != 31:
        raise ValueError("Project data must contain 6 functional and 25 background nodes.")
    timestamps = (frame_times - start_ms).astype(float) / 1000.0
    timestamps.setflags(write=False)
    for history in histories.values():
        history.setflags(write=False)
    return ProjectTrajectoryDataset(
        scene_id=str(background.get("sceneId", functional[0].get("run_id", ""))),
        start_time_ms=start_ms,
        timestamps=timestamps,
        state_history_by_node=histories,
        numeric_id_by_node=numeric_ids,
        functional_nodes=functional_nodes,
        background_nodes=background_nodes,
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _time_ms(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def _state(value):
    state = np.array([value[name] for name in ("x", "y", "z", "vx", "vy", "vz")], dtype=float)
    if np.any(~np.isfinite(state)):
        raise ValueError("Trajectory state contains non-finite values.")
    return state
