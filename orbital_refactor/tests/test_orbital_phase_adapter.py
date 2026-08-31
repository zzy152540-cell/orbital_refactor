import numpy as np
import pytest

from brain_inspired.orbital_phase_adapter import (
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from brain_inspired.passive_phase_observer import PassiveRingCANNObserver
from orbital_core.constants import MU_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    generate_walker_delta_scenario,
)


def _circular_error(actual, expected):
    return (actual - expected + np.pi) % (2.0 * np.pi) - np.pi


def test_orbital_phase_extraction_matches_keplerian_argument_of_latitude():
    semi_major_axis = 7_000_000.0
    eccentricity = 0.01
    inclination = np.deg2rad(53.0)
    raan = np.deg2rad(42.0)
    argument_of_perigee = np.deg2rad(17.0)
    true_anomaly = np.deg2rad(123.0)
    state = keplerian_to_eci(
        semi_major_axis, eccentricity, inclination, raan,
        argument_of_perigee, true_anomaly,
    )
    frame = OrbitalPlaneFrame.from_raan_inclination(
        raan=raan, inclination=inclination,
    )
    phase = extract_orbital_phase_state(
        timestamp=5.0, state_eci=state, frame=frame, source_id="sat-01",
    )
    semilatus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    radius = semilatus_rectum / (1.0 + eccentricity * np.cos(true_anomaly))
    expected_rate = np.sqrt(MU_EARTH * semilatus_rectum) / radius**2
    assert abs(_circular_error(
        phase.argument_of_latitude, argument_of_perigee + true_anomaly,
    )) < 1.0e-12
    assert phase.argument_of_latitude_rate == pytest.approx(expected_rate)
    assert phase.cross_track_position == pytest.approx(0.0, abs=1.0e-8)
    assert phase.source_id == "sat-01"


def test_walker_truth_can_drive_passive_cann_without_phase_feedback():
    timestamps = np.arange(0.0, 10.0 + 2.0, 2.0)
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
    node_id = scenario.node_ids[0]
    elements = scenario.elements_by_node[node_id]
    frame = OrbitalPlaneFrame.from_raan_inclination(
        raan=elements.raan, inclination=config.inclination,
    )
    extracted = [
        extract_orbital_phase_state(
            timestamp=timestamp,
            state_eci=scenario.truth_state_history_by_node[node_id][index],
            frame=frame, source_id=node_id,
        )
        for index, timestamp in enumerate(timestamps)
    ]
    observer = PassiveRingCANNObserver()
    observer.initialize(
        phase=extracted[0].argument_of_latitude,
        timestamp=extracted[0].timestamp, source_id=node_id,
    )
    output = None
    for phase_state in extracted[1:]:
        output = observer.update(
            phase_state.as_periodic_input(use_phase_hint=False)
        )
    assert output is not None
    assert output.cue_applied is False
    assert abs(_circular_error(
        output.decoded_phase, extracted[-1].argument_of_latitude,
    )) < np.deg2rad(0.1)


def test_orbital_plane_frame_rejects_nonorthogonal_axes():
    with pytest.raises(ValueError, match="orthonormal"):
        OrbitalPlaneFrame(np.ones(3), np.ones(3), np.ones(3))
