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
            estimate.append(node.estimate_state[:3])
        return np.asarray(truth), np.asarray(estimate)

    def node_position_error_history(self, node_id: str) -> np.ndarray:
        truth, estimate = self.node_position_history(node_id)
        return np.linalg.norm(estimate - truth, axis=1)

    @property
    def fleet_position_rmse_history(self) -> np.ndarray:
        return np.asarray([
            float(frame.metadata.get("fleet_position_rmse_m", np.nan))
            for frame in self._frames
        ])

    def cann_activity_history(
        self, node_id: str, representation: str,
    ) -> np.ndarray | None:
        values = []
        for frame in self._frames:
            item = next((
                candidate for candidate in frame.cann
                if candidate.node_id == node_id
                and candidate.representation == representation
                and candidate.available and candidate.activity is not None
            ), None)
            if item is None:
                return None
            values.append(item.activity)
        shapes = {value.shape for value in values}
        return None if len(shapes) != 1 else np.stack(values)
