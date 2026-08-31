import numpy as np
import pytest

from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import run_orbital_phase_sidecar
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    generate_walker_delta_scenario,
)


def _walker_history():
    timestamps = np.arange(0.0, 20.0 + 2.0, 2.0)
    config = WalkerDeltaConfig(
        total_satellites=20, plane_count=5, phasing=3,
        semi_major_axis=6_978_137.0, eccentricity=0.001,
        inclination=np.deg2rad(53.0), raan_origin=np.deg2rad(15.0),
        argument_of_perigee=np.deg2rad(8.0),
        base_true_anomaly=np.deg2rad(25.0),
    )
    scenario = generate_walker_delta_scenario(
        timestamps=timestamps, config=config,
    )
    node = scenario.node_ids[0]
    frame = OrbitalPlaneFrame.from_raan_inclination(
        raan=scenario.elements_by_node[node].raan,
        inclination=config.inclination,
    )
    return timestamps, scenario.truth_state_history_by_node[node], frame, node


def test_orbital_phase_sidecar_tracks_walker_history_without_feedback():
    timestamps, states, frame, node = _walker_history()
    result = run_orbital_phase_sidecar(
        timestamps=timestamps, state_history_eci=states, frame=frame,
        source_id=node,
    )
    assert result.source_id == node
    assert result.timestamps.shape == timestamps.shape
    assert np.all(result.valid)
    assert not np.any(result.cue_applied)
    assert np.max(np.abs(result.phase_residual)) < np.deg2rad(0.1)


def test_orbital_phase_sidecar_marks_configured_cue_cadence():
    timestamps, states, frame, node = _walker_history()
    result = run_orbital_phase_sidecar(
        timestamps=timestamps, state_history_eci=states, frame=frame,
        cue_interval_samples=3, source_id=node,
    )
    expected = np.zeros(timestamps.size, dtype=bool)
    expected[3::3] = True
    np.testing.assert_array_equal(result.cue_applied, expected)


def test_orbital_phase_sidecar_rejects_misaligned_history():
    timestamps, states, frame, _ = _walker_history()
    with pytest.raises(ValueError, match="shape"):
        run_orbital_phase_sidecar(
            timestamps=timestamps, state_history_eci=states[:-1], frame=frame,
        )
