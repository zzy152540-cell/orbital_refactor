import numpy as np
import pytest

from cooperative.third_party_observation import (
    apply_third_party_observation,
    classify_observation_receiver,
    run_third_party_target_track_filter,
)
from interfaces.data_objects import ObservationMessage, TargetEstimate
from orbital_core.measurements import measure_relative_range


def _track(estimator, target, state, *, information_ids=()):
    return TargetEstimate(
        estimator_node_id=estimator, target_node_id=target, timestamp=0.0,
        state_estimate=np.asarray(state, dtype=float), covariance=np.eye(6) * 4.0,
        quality_score=1.0, information_ids=information_ids,
    )


def _case():
    state_a = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    state_b = state_a + np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    state_c = state_a + np.array([-800.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    tracks = {
        "sat_a": _track("sat_c", "sat_a", state_a),
        "sat_b": _track(
            "sat_c", "sat_b", state_b + np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ),
        "sat_c": _track("sat_c", "sat_c", state_c),
    }
    observation = ObservationMessage(
        message_id="copy-c", physical_observation_id="physical-a-b-0",
        observer_id="sat_a", target_id="sat_b", timestamp=0.0,
        modality="RANGE",
        measurement=np.array([measure_relative_range(state_a, state_b)]),
        covariance=np.array([[1.0]]),
    )
    return tracks, observation


def test_receiver_classification_preserves_directed_target_semantics():
    _, observation = _case()

    assert classify_observation_receiver("sat_a", observation).role == (
        "observer_self_state"
    )
    assert classify_observation_receiver("sat_b", observation).role == (
        "target_self_state"
    )
    third_party = classify_observation_receiver("sat_c", observation)
    assert third_party.role == "third_party_target_track"
    assert third_party.update_target_id == "sat_b"
    assert third_party.nuisance_target_id == "sat_a"


def test_third_party_observation_updates_only_receiver_target_track():
    tracks, observation = _case()
    before = {
        key: (value.state_estimate.copy(), value.covariance.copy())
        for key, value in tracks.items()
    }

    result = apply_third_party_observation(
        receiver_id="sat_c", target_estimate_by_id=tracks,
        observation=observation,
    )

    assert result.update.estimate.target_node_id == "sat_b"
    assert result.update.estimate.estimator_node_id == "sat_c"
    assert result.update.estimate.information_ids == ("physical-a-b-0",)
    assert not np.allclose(
        result.target_estimate_by_id["sat_b"].state_estimate,
        before["sat_b"][0],
    )
    for unchanged in ("sat_a", "sat_c"):
        np.testing.assert_allclose(
            result.target_estimate_by_id[unchanged].state_estimate,
            before[unchanged][0],
        )
        np.testing.assert_allclose(
            result.target_estimate_by_id[unchanged].covariance,
            before[unchanged][1],
        )


def test_third_party_target_track_rejects_duplicate_physical_observation():
    tracks, observation = _case()
    first = apply_third_party_observation(
        receiver_id="sat_c", target_estimate_by_id=tracks,
        observation=observation,
    )

    with pytest.raises(ValueError, match="already been used"):
        apply_third_party_observation(
            receiver_id="sat_c",
            target_estimate_by_id=first.target_estimate_by_id,
            observation=observation,
        )


def test_third_party_target_track_filter_propagates_and_updates_remote_target():
    tracks, first = _case()
    timestamps = np.array([0.0, 2.0])
    state_a_2 = tracks["sat_a"].state_estimate.copy()
    state_b_2 = tracks["sat_b"].state_estimate.copy()
    second = ObservationMessage(
        message_id="copy-c-2", physical_observation_id="physical-a-b-2",
        observer_id="sat_a", target_id="sat_b", timestamp=2.0,
        modality="RANGE",
        measurement=np.array([measure_relative_range(state_a_2, state_b_2)]),
        covariance=np.array([[1.0]]),
    )

    history = run_third_party_target_track_filter(
        receiver_id="sat_c", timestamps=timestamps,
        initial_target_estimate_by_id=tracks,
        observation_messages=[first, second],
        process_noise_acceleration=0.0,
    )

    assert history.state_history_by_target["sat_b"].shape == (2, 6)
    assert history.covariance_history_by_target["sat_b"].shape == (2, 6, 6)
    assert set(history.nis_history[0]) == {"physical-a-b-0"}
    assert set(history.nis_history[1]) == {"physical-a-b-2"}
    assert history.used_observation_ids == (
        "physical-a-b-0", "physical-a-b-2",
    )
    for covariance in history.covariance_history_by_target["sat_b"]:
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
