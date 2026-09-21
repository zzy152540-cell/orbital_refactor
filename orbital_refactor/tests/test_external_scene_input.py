import numpy as np
import pytest

from adapters.external_scene_input import adapt_external_scene_input


def _payload():
    return {
        "source": "space-perception", "sceneId": "scene-1", "name": "demo",
        "startTime": "2026-09-18 04:00:00",
        "endTime": "2026-09-18 04:00:02",
        "satelliteCount": 1,
        "satellites": [{
            "assetId": "asset-1", "assetName": "B01-O",
            "satelliteId": "satellite-1", "norad": "90001",
            "semiMajorAxis": "6854.252339", "eccentricity": "0.002091951",
            "inclination": "35.796685", "raan": "359.845475",
            "argPerigee": "101.923841", "meanAnomaly": "258.9007",
            "epoch": "26261.16666667",
            "tleLine1": "1 90001U          26261.16666667  .00000000",
            "tleLine2": (
                "2 90001  35.7801 359.9792 0020633 137.8247 "
                "223.2562 15.32054775    07"
            ),
        }],
    }


def test_external_scene_uses_structured_elements_and_utc_one_second_axis():
    result = adapt_external_scene_input(_payload())

    assert result.scene_id == "scene-1"
    assert result.step_seconds == 1.0
    np.testing.assert_array_equal(result.timestamps, [0.0, 1.0, 2.0])
    state = result.initial_state_by_node["B01-O"]
    assert state.shape == (6,)
    assert np.all(np.isfinite(state))
    assert 6.7e6 < np.linalg.norm(state[:3]) < 7.0e6
    assert result.satellites[0].tle_line2.startswith("2 90001")
    diagnostic = result.tle_diagnostics[0]
    assert diagnostic.available
    assert abs(diagnostic.argument_of_perigee_difference_deg) > 30.0


def test_external_scene_rejects_count_mismatch():
    payload = _payload()
    payload["satelliteCount"] = 2
    with pytest.raises(ValueError, match="satelliteCount"):
        adapt_external_scene_input(payload)
