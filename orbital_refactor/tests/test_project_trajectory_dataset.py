import json

import numpy as np

from adapters.project_trajectory_dataset import load_project_trajectory_dataset


def test_project_dataset_loads_and_pairs_functional_targets(tmp_path):
    times = ["2026-09-18 04:00:00.000", "2026-09-18 04:00:01.000"]
    functional = []
    for frame_index, time in enumerate(times):
        functional.append({
            "run_id": "scene-a", "time": time,
            "satelliteList": [
                {"satObj": f"F{i}", "satId": 90000 + i,
                 "position": {"x": 7.0e6 + i * 100.0, "y": frame_index * 10.0, "z": 0.0,
                              "vx": 0.0, "vy": 10.0, "vz": 0.0}}
                for i in range(6)
            ],
        })
    background = {"sceneId": "scene-a", "satellites": []}
    for i in range(25):
        background["satellites"].append({
            "name": f"V{i}", "norad": str(91000 + i),
            "points": [
                {"time": time, "x": 7.0e6 + i * 100.0 + 25.0, "y": index * 10.0,
                 "z": 0.0, "vx": 0.0, "vy": 10.0, "vz": 0.0}
                for index, time in enumerate(times)
            ],
        })
    background_path = tmp_path / "background.json"
    functional_path = tmp_path / "functional.json"
    background_path.write_text(json.dumps(background), encoding="utf-8")
    functional_path.write_text(json.dumps(functional), encoding="utf-8")
    dataset = load_project_trajectory_dataset(background_path, functional_path)
    assert len(dataset.state_history_by_node) == 31
    assert dataset.timestamps.tolist() == [0.0, 1.0]
    assert dataset.nearest_background_observers()["F0"] == "V0"
    assert dataset.nearest_observers(("F0",), candidates=dataset.background_nodes)["F0"] == "V0"
    assert set(dataset.nonmaneuver_nodes(maximum_velocity_residual_mps=20.0)) == set(dataset.state_history_by_node)
    assert np.all(np.isfinite(dataset.state_history_by_node["F0"]))
