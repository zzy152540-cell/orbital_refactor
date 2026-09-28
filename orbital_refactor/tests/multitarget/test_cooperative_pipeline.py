import numpy as np
import pytest

from tracking import TargetNodeReport, run_cooperative_target_fusion


def _report(observer, target, *, timestamp=10.0, offset=0.0, valid=True):
    return TargetNodeReport(
        observer_id=observer,
        target_id=target,
        track_id=f"track-{target}",
        timestamp=timestamp,
        state_eci=np.array([
            7.0e6 + offset, 100.0, -50.0, 0.0, 7500.0, 1.0,
        ]),
        covariance_eci=np.diag([4.0, 4.0, 4.0, 0.04, 0.04, 0.04]),
        quality_score=0.9,
        valid_flag=valid,
    )


def test_three_observers_fuse_one_known_target_end_to_end():
    output = run_cooperative_target_fusion(
        scene_id="scene-3-to-1",
        timestamp=10.0,
        expected_target_ids=("target_01",),
        reports=[
            _report("observer_a", "target_01", offset=-2.0),
            _report("observer_b", "target_01", offset=1.0),
            _report("observer_c", "target_01", offset=3.0),
        ],
    )

    assert output.scene_id == "scene-3-to-1"
    assert set(output.estimates_by_target) == {"target_01"}
    estimate = output.estimates_by_target["target_01"]
    assert estimate.contributing_observer_ids == (
        "observer_a", "observer_b", "observer_c",
    )
    assert estimate.state_eci.shape == (6,)
    assert estimate.covariance_eci.shape == (6, 6)


def test_three_observers_two_targets_remain_independent():
    output = run_cooperative_target_fusion(
        scene_id="scene-3-to-2",
        timestamp=10.0,
        expected_target_ids=("target_01", "target_02"),
        reports=[
            _report("observer_a", "target_01", offset=0.0),
            _report("observer_b", "target_01", offset=2.0),
            _report("observer_a", "target_02", offset=1000.0),
            _report("observer_c", "target_02", offset=1002.0),
        ],
    )

    assert set(output.estimates_by_target) == {"target_01", "target_02"}
    first = output.estimates_by_target["target_01"]
    second = output.estimates_by_target["target_02"]
    assert first.contributing_observer_ids == ("observer_a", "observer_b")
    assert second.contributing_observer_ids == ("observer_a", "observer_c")
    assert second.state_eci[0] - first.state_eci[0] > 900.0


def test_invalid_report_is_excluded_without_hiding_other_targets():
    output = run_cooperative_target_fusion(
        scene_id="scene",
        timestamp=10.0,
        reports=[
            _report("observer_a", "target_01", valid=False),
            _report("observer_b", "target_01", valid=True),
            _report("observer_c", "target_02", valid=True),
        ],
    )

    assert output.estimates_by_target["target_01"].contributing_observer_ids == (
        "observer_b",
    )
    assert output.estimates_by_target["target_02"].contributing_observer_ids == (
        "observer_c",
    )


def test_pipeline_rejects_wrong_epoch_and_unexpected_target():
    with pytest.raises(ValueError, match="fusion timestamp"):
        run_cooperative_target_fusion(
            scene_id="scene", timestamp=10.0,
            reports=[_report("observer_a", "target_01", timestamp=11.0)],
        )
    with pytest.raises(ValueError, match="unexpected target"):
        run_cooperative_target_fusion(
            scene_id="scene", timestamp=10.0,
            expected_target_ids=("target_01",),
            reports=[_report("observer_a", "target_02")],
        )
