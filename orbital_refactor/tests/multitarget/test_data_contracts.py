import numpy as np
import pytest

from tracking import (
    ECI_FRAME,
    MAX_INFORMATION_IDS,
    MultiTargetInput,
    ObserverState,
    TargetInitialState,
    TargetNodeReport,
    TargetTrackKey,
)


def _state(offset=0.0):
    return np.array([7.0e6 + offset, 0.0, 0.0, 0.0, 7500.0, 0.0])


def test_observer_and_target_identifiers_are_distinct_roles():
    observer = ObserverState("observer_a", 0.0, _state(), np.eye(6))
    target = TargetInitialState("target_01", "track_01", 0.0, _state(1000.0), np.eye(6))

    value = MultiTargetInput(
        scene_id="scene",
        observer_states={"observer_a": observer},
        target_initial_states={"target_01": target},
    )

    assert value.observer_states["observer_a"].frame == ECI_FRAME
    assert value.target_initial_states["target_01"].track_id == "track_01"


def test_observer_and_target_identifier_sets_cannot_overlap():
    observer = ObserverState("shared", 0.0, _state(), np.eye(6))
    target = TargetInitialState("shared", "track", 0.0, _state(1000.0), np.eye(6))

    with pytest.raises(ValueError, match="disjoint"):
        MultiTargetInput("scene", {"shared": observer}, {"shared": target})


def test_public_target_state_rejects_ambiguous_reference_frame():
    with pytest.raises(ValueError, match="J2000_ECI"):
        ObserverState("observer", 0.0, _state(), np.eye(6), frame="SPRI")


def test_track_key_keeps_scene_and_track_identity_separate():
    assert TargetTrackKey("scene-a", "track-1") != TargetTrackKey("scene-b", "track-1")


def test_information_lineage_is_deduplicated_and_bounded():
    identifiers = tuple(
        f"information-{index}" for index in range(MAX_INFORMATION_IDS + 5)
    )
    report = TargetNodeReport(
        observer_id="observer",
        target_id="target",
        track_id="track",
        timestamp=0.0,
        state_eci=_state(),
        covariance_eci=np.eye(6),
        quality_score=0.9,
        used_measurement_ids=(identifiers[0], *identifiers, identifiers[-1]),
    )

    assert len(report.used_measurement_ids) == MAX_INFORMATION_IDS
    assert report.used_measurement_ids[0] == "information-5"
    assert report.used_measurement_ids[-1] == identifiers[-1]
