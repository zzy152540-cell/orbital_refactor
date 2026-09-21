import json
from pathlib import Path

import numpy as np
import pytest

from exporters.trajectory_time_series import export_estimate_trajectory_frames
from visualization.data_contract import VisualNodeState, VisualizationFrame


def _frame(index, timestamp=None):
    return VisualizationFrame(
        scenario_id="scene-a", run_id="run-a",
        timestamp=float(index if timestamp is None else timestamp),
        epoch_index=index,
        nodes=(VisualNodeState(
            node_id="sat_p01_s01",
            estimate_state=np.array([1, 2, 3, 4, 5, 6], dtype=float) + index,
            covariance_diagonal=np.ones(6),
        ),),
    )


def test_estimate_frames_export_as_trajectory_2(tmp_path):
    target = tmp_path / "trajectory.json"
    export_estimate_trajectory_frames(
        (_frame(0), _frame(1)), target,
        scene_id="scene-a", time_base_id="TB-A", dataset_id="EST-A",
        start_time_ms=1789704000000,
        satellite_id_by_node={"sat_p01_s01": 90001},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == "TRAJECTORY-2.0"
    assert payload["sampleIntervalMs"] == 1000
    assert payload["frameCount"] == 2
    assert payload["durationMs"] == 1000
    assert payload["frames"][1]["frameIndex"] == 1
    assert payload["frames"][1]["timeMs"] == 1789704001000
    satellite = payload["frames"][0]["satelliteList"][0]
    assert satellite["satObj"] == "sat_p01_s01"
    assert satellite["satId"] == 90001
    assert satellite["position"] == {
        "x": 1.0, "y": 2.0, "z": 3.0,
        "vx": 4.0, "vy": 5.0, "vz": 6.0,
    }


def test_trajectory_export_rejects_non_one_second_frames(tmp_path):
    with pytest.raises(ValueError, match="one-second"):
        export_estimate_trajectory_frames(
            (_frame(0), _frame(1, timestamp=2.0)),
            tmp_path / "trajectory.json",
            scene_id="scene-a", time_base_id="TB-A", dataset_id="EST-A",
            start_time_ms=1789704000000,
        )
