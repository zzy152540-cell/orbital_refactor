import numpy as np

from adapters.external_scene_input import adapt_external_scene_input
from adapters.project_scene_builder import build_project_scene_input
from orbital_core.orbit_elements import eci_to_keplerian, keplerian_to_eci


def test_eci_keplerian_round_trip():
    state = keplerian_to_eci(
        6_854_000.0, 0.002, np.deg2rad(36), np.deg2rad(12),
        np.deg2rad(101), np.deg2rad(258),
    )
    np.testing.assert_allclose(keplerian_to_eci(*eci_to_keplerian(state)), state, atol=1e-7)


def test_project_builder_produces_external_31_satellite_scene(tmp_path):
    start = "2026-09-18 04:00:00.000"
    end = "2026-09-18 04:00:01.000"
    functional_states = []
    initialization = []
    for index in range(6):
        name = f"F{index}"
        state = keplerian_to_eci(6_854_000 + index, 0.001, 0.6, 0.1, 0.2, 0.3 + index * 0.01)
        functional_states.append({
            "satObj": name, "satId": 90001 + index,
            "position": dict(zip(("x", "y", "z", "vx", "vy", "vz"), state)),
        })
        initialization.append({"name": name, "tle": {"line1": "1 test", "line2": "2 test"}})
    background = {"sceneId": "scene-a", "satellites": []}
    for index in range(25):
        name = f"V{index}"
        state = keplerian_to_eci(6_860_000 + index, 0.001, 0.6, 0.1, 0.2, 0.4 + index * 0.01)
        background["satellites"].append({
            "name": name, "norad": str(91000 + index), "tleLine1": "", "tleLine2": "",
            "points": [{"time": start, **dict(zip(("x", "y", "z", "vx", "vy", "vz"), state))}],
        })
    functional = [
        {"run_id": "scene-a", "time": start, "satelliteList": functional_states},
        {"run_id": "scene-a", "time": end, "satelliteList": functional_states},
    ]
    paths = [tmp_path / name for name in ("init.json", "background.json", "functional.json")]
    import json
    for path, value in zip(paths, (initialization, background, functional)):
        path.write_text(json.dumps(value), encoding="utf-8")
    payload = build_project_scene_input(*paths)
    assert payload["satelliteCount"] == 31
    parsed = adapt_external_scene_input(payload)
    assert len(parsed.satellites) == 31
    assert len(parsed.timestamps) == 2
