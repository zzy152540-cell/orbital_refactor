import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exporters.trajectory_time_series import (
    export_absolute_state_histories,
    export_estimate_trajectory_frames,
    export_single_filter_history,
)
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


def test_absolute_multi_satellite_histories_export_as_estimates(tmp_path):
    states = {
        "sat-b": np.array([[20, 21, 22, 2, 2.1, 2.2], [23, 24, 25, 2.3, 2.4, 2.5]]),
        "sat-a": np.array([[10, 11, 12, 1, 1.1, 1.2], [13, 14, 15, 1.3, 1.4, 1.5]]),
    }
    target = export_absolute_state_histories(
        [0.0, 1.0], states, tmp_path / "absolute.json",
        scene_id="scene-a", time_base_id="TB-A", dataset_id="EST-A",
        start_time_ms=1_000,
        satellite_id_by_node={"sat-a": 101, "sat-b": 102},
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["dataKind"] == "ESTIMATED"
    assert [item["satObj"] for item in payload["frames"][0]["satelliteList"]] == [
        "sat-a", "sat-b",
    ]
    assert payload["frames"][1]["satelliteList"][1]["position"]["x"] == 23.0


def test_single_relative_filter_history_is_converted_to_absolute_j2000(tmp_path):
    history = SimpleNamespace(
        timestamps=np.array([0.0, 1.0]),
        fused_state_history=np.array([
            [1, 2, 3, 0.1, 0.2, 0.3],
            [4, 5, 6, 0.4, 0.5, 0.6],
        ], dtype=float),
    )
    observer = np.array([
        [100, 200, 300, 10, 20, 30],
        [110, 210, 310, 11, 21, 31],
    ], dtype=float)
    target = export_single_filter_history(
        history, observer, tmp_path / "single.json",
        target_id="target-a", satellite_id=90001,
        scene_id="scene-a", time_base_id="TB-A", dataset_id="EST-A",
        start_time_ms=1_000,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    first = payload["frames"][0]["satelliteList"][0]
    assert first["satId"] == 90001
    assert first["position"] == {
        "x": 101.0, "y": 202.0, "z": 303.0,
        "vx": 10.1, "vy": 20.2, "vz": 30.3,
    }
