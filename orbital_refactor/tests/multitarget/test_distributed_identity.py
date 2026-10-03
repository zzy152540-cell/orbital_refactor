import numpy as np

from tracking import CooperativeTrackIdentityResolver, TargetNodeReport


def _report(observer, target, track, offset, timestamp=0.0):
    return TargetNodeReport(
        observer_id=observer,
        target_id=target,
        track_id=track,
        timestamp=timestamp,
        state_eci=np.array([7.0e6 + offset, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        covariance_eci=np.diag([25.0, 25.0, 25.0, 0.1, 0.1, 0.1]),
        quality_score=0.9,
    )


def test_different_local_names_map_to_the_same_two_physical_targets():
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
    )
    result = resolver.resolve([
        _report("node-a", "alpha", "track-alpha", 0.0),
        _report("node-a", "beta", "track-beta", 1000.0),
        _report("node-b", "red", "track-red", 5.0),
        _report("node-b", "blue", "track-blue", 1005.0),
    ])

    mapping = {
        (item.observer_id, item.local_target_id): item.cooperative_target_id
        for item in result.assignments
    }
    assert mapping[("node-a", "alpha")] == mapping[("node-b", "red")]
    assert mapping[("node-a", "beta")] == mapping[("node-b", "blue")]
    assert mapping[("node-a", "alpha")] != mapping[("node-a", "beta")]
    assert len(result.created_cooperative_target_ids) == 2


def test_bindings_persist_when_targets_move_and_local_names_stay_different():
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=100.0,
        velocity_gate_mps=10.0,
    )
    first = resolver.resolve([
        _report("node-a", "alpha", "track-alpha", 0.0),
        _report("node-b", "red", "track-red", 5.0),
    ])
    second = resolver.resolve([
        _report("node-a", "alpha", "track-alpha", 7500.0, timestamp=1.0),
        _report("node-b", "red", "track-red", 7505.0, timestamp=1.0),
    ])

    first_ids = {item.cooperative_target_id for item in first.assignments}
    second_ids = {item.cooperative_target_id for item in second.assignments}
    assert first_ids == second_ids
    assert second.created_cooperative_target_ids == ()


def test_cooperative_estimate_localizes_back_to_each_nodes_track_identity():
    resolver = CooperativeTrackIdentityResolver(position_gate_m=100.0)
    result = resolver.resolve([
        _report("node-a", "alpha", "track-alpha", 0.0),
        _report("node-b", "red", "track-red", 5.0),
    ])
    from tracking import fuse_target_reports

    estimate = fuse_target_reports(result.reports)
    localized_a = resolver.localize_estimates(
        observer_id="node-a", estimates={estimate.target_id: estimate},
    )
    localized_b = resolver.localize_estimates(
        observer_id="node-b", estimates={estimate.target_id: estimate},
    )

    assert localized_a["alpha"].track_id == "track-alpha"
    assert localized_b["red"].track_id == "track-red"

    assert resolver.retire_local_track(
        observer_id="node-a",
        local_target_id="alpha",
        local_track_id="track-alpha",
    )
    assert resolver.localize_estimates(
        observer_id="node-a", estimates={estimate.target_id: estimate},
    ) == {}
    assert not resolver.retire_local_track(
        observer_id="node-a",
        local_target_id="alpha",
        local_track_id="track-alpha",
    )
