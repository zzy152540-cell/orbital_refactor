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
        for frame in self._frames:
            if {node.node_id for node in frame.nodes} != expected:
                raise ValueError("Replay frames must cover an identical node set.")

    def __len__(self) -> int:
        return len(self._frames)

    def frame(self, index: int):
        return self._frames[index]

    @property
    def events(self):
        return tuple(
            event for frame in self._frames for event in frame.events
        )

    def nearest_index(self, timestamp: float) -> int:
        return int(np.argmin(np.abs(self.timestamps - float(timestamp))))

    def node_position_history(self, node_id: str) -> tuple[np.ndarray, np.ndarray]:
        if node_id not in self.node_ids:
            raise KeyError(f"Unknown replay node: {node_id}")
        truth, estimate = [], []
        for frame in self._frames:
            node = next(item for item in frame.nodes if item.node_id == node_id)
            truth.append(
                np.full(3, np.nan) if node.truth_state is None
                else node.truth_state[:3]
            )
            estimate.append(
                node.estimate_state[:3] if node.valid
                else np.full(3, np.nan)
            )
        return np.asarray(truth), np.asarray(estimate)

    def node_position_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "position")

    def node_velocity_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "velocity")

    def node_acceleration_error_history(self, node_id: str) -> np.ndarray:
        return self.node_error_history(node_id, "acceleration")

    def node_error_history(self, node_id: str, quantity: str) -> np.ndarray:
        truth, estimate = self._node_state_history(node_id)
        normalized = str(quantity).strip().lower()
        if normalized == "position":
            truth_values, estimate_values = truth[:, :3], estimate[:, :3]
        elif normalized == "velocity":
            truth_values, estimate_values = truth[:, 3:6], estimate[:, 3:6]
        elif normalized == "acceleration":
            truth_values = self._finite_difference(truth[:, 3:6])
            estimate_values = self._finite_difference(estimate[:, 3:6])
        else:
            raise ValueError(f"Unsupported replay error quantity: {quantity!r}")
        return np.linalg.norm(estimate_values - truth_values, axis=1)

    def fleet_error_rmse_history(self, quantity: str) -> np.ndarray:
        errors = np.stack([
            self.node_error_history(node_id, quantity)
            for node_id in self.node_ids
        ])
        valid = np.isfinite(errors)
        counts = np.sum(valid, axis=0)
        squared_sum = np.nansum(np.square(errors), axis=0)
        result = np.full(errors.shape[1], np.nan)
        available = counts > 0
        result[available] = np.sqrt(squared_sum[available] / counts[available])
        return result

    @property
    def fleet_position_rmse_history(self) -> np.ndarray:
        recorded = np.asarray([
            float(frame.metadata.get("fleet_position_rmse_m", np.nan))
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
        truth, estimate = [], []
        for frame in self._frames:
            node = next(item for item in frame.nodes if item.node_id == node_id)
            truth.append(
                np.full(node.estimate_state.shape, np.nan)
                if node.truth_state is None else node.truth_state
            )
            estimate.append(
                node.estimate_state if node.valid
                else np.full(node.estimate_state.shape, np.nan)
            )
        return np.asarray(truth), np.asarray(estimate)

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
            return None
        shape = shapes.pop()
        return np.stack([
            np.full(shape, np.nan) if value is None else value
            for value in values
        ])
