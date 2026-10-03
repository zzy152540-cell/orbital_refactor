import numpy as np
import pytest

from tracking import TargetNodeReport, fuse_target_reports, group_reports_by_target


def _report(observer, target, *, track=None, timestamp=0.0, offset=0.0):
    return TargetNodeReport(
        observer_id=observer,
        target_id=target,
        track_id=track or f"track-{target}",
        timestamp=timestamp,
        state_eci=np.array([7.0e6 + offset, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        covariance_eci=np.eye(6) * 4.0,
        quality_score=0.8,
    )


def test_multiple_observers_for_same_target_are_preserved():
    grouped = group_reports_by_target([
        _report("observer_b", "target_01", offset=2.0),
        _report("observer_a", "target_01", offset=1.0),
    ])

    assert list(grouped) == ["target_01"]
    assert [item.observer_id for item in grouped["target_01"]] == [
        "observer_a", "observer_b",
    ]


def test_one_observer_can_report_multiple_targets_without_cross_fusion():
    grouped = group_reports_by_target([
        _report("observer_a", "target_01"),
        _report("observer_a", "target_02"),
    ])

    assert set(grouped) == {"target_01", "target_02"}


def test_target_ci_produces_target_indexed_global_estimate():
    result = fuse_target_reports([
        _report("observer_a", "target_01", offset=-2.0),
        _report("observer_b", "target_01", offset=2.0),
        _report("observer_c", "target_01", offset=1.0),
    ])

    assert result.target_id == "target_01"
    assert result.track_id == "track-target_01"
    assert result.contributing_observer_ids == (
        "observer_a", "observer_b", "observer_c",
    )
    assert set(result.node_weights) == {
        "observer_a", "observer_b", "observer_c",
    }


def test_target_ci_supports_more_than_three_neighbor_reports():
    reports = [
        _report(f"observer_{index}", "target_01", offset=float(index))
        for index in range(5)
    ]
    result = fuse_target_reports(reports)
    reversed_result = fuse_target_reports(reversed(reports))

    assert result.contributing_observer_ids == tuple(
        f"observer_{index}" for index in range(5)
    )
    assert set(result.node_weights) == set(result.contributing_observer_ids)
    assert np.isclose(sum(result.node_weights.values()), 1.0)
    assert np.all(np.linalg.eigvalsh(result.covariance_eci) >= -1e-10)
    assert np.allclose(result.state_eci, reversed_result.state_eci)
    assert np.allclose(result.covariance_eci, reversed_result.covariance_eci)
    assert result.node_weights == reversed_result.node_weights


def test_target_ci_rejects_cross_target_reports():
    with pytest.raises(ValueError, match="same physical target_id"):
        fuse_target_reports([
            _report("observer_a", "target_01"),
            _report("observer_b", "target_02"),
        ])


def test_target_ci_rejects_unaligned_epochs_and_duplicate_observers():
    with pytest.raises(ValueError, match="same timestamp"):
        fuse_target_reports([
            _report("observer_a", "target_01", timestamp=0.0),
            _report("observer_b", "target_01", timestamp=1.0),
        ])
    with pytest.raises(ValueError, match="one report per observer"):
        fuse_target_reports([
            _report("observer_a", "target_01", offset=0.0),
            _report("observer_a", "target_01", offset=1.0),
        ])
