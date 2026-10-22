"""Toolkit-independent timeline model used by the replay GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualization.recording import VisualizationRecordingReader


class VisualizationReplayModel:
    def __init__(self, source: str | Path) -> None:
        self.reader = VisualizationRecordingReader(source)
        self._frames = tuple(self.reader)
        self.timestamps = np.asarray(
            [frame.timestamp for frame in self._frames], dtype=float,
        )
        self.node_ids = tuple(node.node_id for node in self._frames[0].nodes)
        expected = set(self.node_ids)
        node_maps = []
        for frame in self._frames:
            node_map = {node.node_id: node for node in frame.nodes}
            if set(node_map) != expected:
                raise ValueError("Replay frames must cover an identical node set.")
            node_maps.append(node_map)
        self._node_maps = tuple(node_maps)
        self._state_history_by_node = {
            node_id: self._build_node_state_history(node_id)
            for node_id in self.node_ids
        }
        self._error_history_cache = {
            (node_id, quantity): self._build_node_error_history(
                node_id, quantity,
            )
            for node_id in self.node_ids
            for quantity in ("position", "velocity", "acceleration")
        }
        self._fleet_error_cache = {}
        self._cann_history_cache = {}
        self._events = tuple(
            event for frame in self._frames for event in frame.events
        )

    def __len__(self) -> int:
        return len(self._frames)

    def frame(self, index: int):
        return self._frames[index]

    @property
    def events(self):
        return self._events

    def nearest_index(self, timestamp: float) -> int:
        return int(np.argmin(np.abs(self.timestamps - float(timestamp))))

    def node_position_history(self, node_id: str) -> tuple[np.ndarray, np.ndarray]:
        if node_id not in self.node_ids:
            raise KeyError(f"Unknown replay node: {node_id}")
        truth, estimate = self._state_history_by_node[node_id]
        return truth[:, :3], estimate[:, :3]

    def node_position_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "position")

    def node_velocity_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "velocity")

    def node_acceleration_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "acceleration")

    def node_error_history(self, node_id: str, quantity: str) -> np.ndarray:
        normalized = str(quantity).strip().lower()
        key = (node_id, normalized)
        if key not in self._error_history_cache:
            if node_id not in self.node_ids:
                raise KeyError(f"Unknown replay node: {node_id}")
            raise ValueError(f"Unsupported replay error quantity: {quantity!r}")
        return self._error_history_cache[key]

    def fleet_error_rmse_history(
        self, quantity: str, node_ids=None,
    ) -> np.ndarray:
        selected_ids = self.node_ids if node_ids is None else tuple(node_ids)
        cache_key = (tuple(selected_ids), str(quantity).strip().lower())
        if cache_key in self._fleet_error_cache:
            return self._fleet_error_cache[cache_key]
        errors = np.stack([
            self.node_error_history(node_id, quantity)
            for node_id in selected_ids
        ])
        valid = np.isfinite(errors)
        counts = np.sum(valid, axis=0)
        squared_sum = np.nansum(np.square(errors), axis=0)
        result = np.full(errors.shape[1], np.nan)
        available = counts > 0
        result[available] = np.sqrt(squared_sum[available] / counts[available])
        result.setflags(write=False)
        self._fleet_error_cache[cache_key] = result
        return result

    @property
    def fleet_position_rmse_history(self) -> np.ndarray:
        recorded = np.asarray([
            (
                np.nan
                if frame.metadata.get("fleet_position_rmse_m") is None
                else float(frame.metadata["fleet_position_rmse_m"])
            )
            for frame in self._frames
        ])
        computed = self.fleet_error_rmse_history("position")
        return np.where(np.isfinite(recorded), recorded, computed)

    @property
    def fleet_velocity_rmse_history(self) -> np.ndarray:
        return self.fleet_error_rmse_history("velocity")

    @property
    def fleet_acceleration_rmse_history(self) -> np.ndarray:
        return self.fleet_error_rmse_history("acceleration")

    def _node_state_history(self, node_id: str) -> tuple[np.ndarray, np.ndarray]:
        if node_id not in self.node_ids:
            raise KeyError(f"Unknown replay node: {node_id}")
        return self._state_history_by_node[node_id]

    def _build_node_state_history(
        self, node_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        truth, estimate = [], []
        for node_map in self._node_maps:
            node = node_map[node_id]
            truth.append(
                np.full(node.estimate_state.shape, np.nan)
                if node.truth_state is None else node.truth_state
            )
            estimate.append(
                node.estimate_state if node.valid
                else np.full(node.estimate_state.shape, np.nan)
            )
        result = (np.asarray(truth), np.asarray(estimate))
        for value in result:
            value.setflags(write=False)
        return result

    def _build_node_error_history(
        self, node_id: str, quantity: str,
    ) -> np.ndarray:
        truth, estimate = self._state_history_by_node[node_id]
        if quantity == "position":
            truth_values, estimate_values = truth[:, :3], estimate[:, :3]
        elif quantity == "velocity":
            truth_values, estimate_values = truth[:, 3:6], estimate[:, 3:6]
        elif quantity == "acceleration":
            truth_values = self._finite_difference(truth[:, 3:6])
            estimate_values = self._finite_difference(estimate[:, 3:6])
        else:  # pragma: no cover - internal construction is fixed above
            raise ValueError(quantity)
        result = np.linalg.norm(estimate_values - truth_values, axis=1)
        result.setflags(write=False)
        return result

    def _finite_difference(self, values: np.ndarray) -> np.ndarray:
        result = np.full_like(values, np.nan, dtype=float)
        if values.shape[0] < 2:
            return result
        intervals = np.diff(self.timestamps)
        valid = np.isfinite(intervals) & (intervals > 0.0)
        differences = values[1:] - values[:-1]
        result[1:][valid] = differences[valid] / intervals[valid, None]
        return result

    def cann_activity_history(
        self, node_id: str, representation: str,
    ) -> np.ndarray | None:
        return self._cann_history(node_id, representation, "activity")

    def cann_decoded_history(
        self, node_id: str, representation: str,
    ) -> np.ndarray | None:
        return self._cann_history(node_id, representation, "decoded_value")

    def _cann_history(
        self, node_id: str, representation: str, field: str,
    ) -> np.ndarray | None:
        cache_key = (node_id, representation, field)
        if cache_key in self._cann_history_cache:
            return self._cann_history_cache[cache_key]
        values = []
        for frame in self._frames:
            item = next((
                candidate for candidate in frame.cann
                if candidate.node_id == node_id
                and candidate.representation == representation
                and candidate.available
                and getattr(candidate, field) is not None
            ), None)
            values.append(None if item is None else getattr(item, field))
        shapes = {value.shape for value in values if value is not None}
        if len(shapes) != 1:
            self._cann_history_cache[cache_key] = None
            return None
        shape = shapes.pop()
        result = np.stack([
            np.full(shape, np.nan) if value is None else value
            for value in values
        ])
        result.setflags(write=False)
        self._cann_history_cache[cache_key] = result
        return result
