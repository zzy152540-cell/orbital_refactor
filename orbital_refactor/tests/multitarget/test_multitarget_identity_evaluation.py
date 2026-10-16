import numpy as np

from experiments.multitarget_identity_evaluation import (
    IdentityAwareTrackEvaluator,
)


def _state(x, vx=0.0):
    return np.array([x, 0.0, 0.0, vx, 0.0, 0.0])


def test_fixed_identity_scoring_does_not_hide_a_track_swap():
    evaluator = IdentityAwareTrackEvaluator(("truth-a", "truth-b"))
    truth = {"truth-a": _state(0.0), "truth-b": _state(100.0)}

    first = evaluator.update(
        timestamp=0.0,
        state_by_track_id={"track-1": _state(0.0), "track-2": _state(100.0)},
        truth_state_by_id=truth,
    )
    swapped = evaluator.update(
        timestamp=1.0,
        state_by_track_id={"track-1": _state(100.0), "track-2": _state(0.0)},
        truth_state_by_id=truth,
    )

    assert first.samples_by_truth["truth-a"].position_error_m == 0.0
    assert swapped.samples_by_truth["truth-a"].position_error_m == 100.0
    assert swapped.samples_by_truth["truth-b"].position_error_m == 100.0
    assert evaluator.identity_switch_count == 2
    assert evaluator.nearest_truth_disagreement_count == 2


def test_missing_truth_and_replacement_track_are_audited():
    evaluator = IdentityAwareTrackEvaluator(("truth-a", "truth-b"))
    truth = {"truth-a": _state(0.0), "truth-b": _state(100.0)}
    evaluator.update(
        timestamp=0.0,
        state_by_track_id={"track-1": _state(0.0), "track-2": _state(100.0)},
        truth_state_by_id=truth,
    )
    missing = evaluator.update(
        timestamp=1.0,
        state_by_track_id={"track-1": _state(1.0)},
        truth_state_by_id=truth,
    )
    recovered = evaluator.update(
        timestamp=2.0,
        state_by_track_id={"track-1": _state(2.0), "track-3": _state(102.0)},
        truth_state_by_id=truth,
    )

    assert missing.missing_truth_ids == ("truth-b",)
    assert not recovered.missing_truth_ids
    assert evaluator.missing_truth_sample_count == 1
    assert evaluator.track_replacement_count == 1
    assert evaluator.assignment_by_track_id["track-3"] == "truth-b"
