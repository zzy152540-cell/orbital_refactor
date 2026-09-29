import json
from pathlib import Path

from experiments.run_external_multitarget_estimation import (
    run_external_multitarget_estimation,
)


def test_external_multitarget_example_runs_to_j2000_output(tmp_path):
    project = Path(__file__).resolve().parents[2]
    source = project / "configs" / "external_multitarget_3x2_example.json"

    summary = run_external_multitarget_estimation(
        source, tmp_path, random_seed=5, image_size=17,
    )

    assert summary["observerCount"] == 3
    assert summary["targetCount"] == 2
    assert summary["localFilterTaskCount"] == 6
    payload = json.loads(
        (tmp_path / "estimated_targets.json").read_text(encoding="utf-8")
    )
    assert payload["coordinateFrame"] == "J2000_ECI"
    assert payload["initialOrbitDeterminationPerformed"] is False
    assert payload["frameCount"] == 11
    first = payload["frames"][0]
    assert first["time"] == "2026-09-18 04:00:00.000"
    assert len(first["targetList"]) == 2
    required_state = {"x", "y", "z", "vx", "vy", "vz"}
    assert required_state <= set(first["targetList"][0])
