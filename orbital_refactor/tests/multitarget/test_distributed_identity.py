import numpy as np

from cooperative.temporal_alignment import propagate_state_covariance
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


def _state_report(observer, target, track, state, timestamp):
    return TargetNodeReport(
        observer_id=observer,
        target_id=target,
        track_id=track,
        timestamp=timestamp,
        state_eci=np.asarray(state, dtype=float),
        covariance_eci=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
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


def test_joint_reacquisition_preserves_two_identities_after_long_occlusion():
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=2_000.0,
        velocity_gate_mps=20.0,
        minimum_match_margin=0.01,
    )
    covariance = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    state_alpha = np.array([7.0e6, -20_000.0, 0.0, 20.0, 7540.0, 0.0])
    state_beta = np.array([7.0e6, 20_000.0, 0.0, -20.0, 7460.0, 0.0])
    first = resolver.resolve([
        _state_report("node-a", "alpha", "track-alpha", state_alpha, 0.0),
        _state_report("node-a", "beta", "track-beta", state_beta, 0.0),
    ])
    original = {
        item.local_target_id: item.cooperative_target_id
        for item in first.assignments
    }
    for local_target_id in ("alpha", "beta"):
        assert resolver.retire_local_track(
            observer_id="node-a",
            local_target_id=local_target_id,
            local_track_id=f"track-{local_target_id}",
        )

    timestamp = 900.0
    propagated_alpha, _ = propagate_state_covariance(
        state_alpha, covariance, timestamp,
    )
    propagated_beta, _ = propagate_state_covariance(
        state_beta, covariance, timestamp,
    )
    # Reverse the lexical order of the physical targets to ensure identity
    # assignment is driven by the joint kinematic match, not input order.
    reacquired = resolver.resolve([
        _state_report(
            "node-a", "new-01", "track-new-01",
            propagated_beta + np.array([20.0, -10.0, 5.0, 0.02, 0.0, 0.0]),
            timestamp,
        ),
        _state_report(
            "node-a", "new-02", "track-new-02",
            propagated_alpha + np.array([-15.0, 10.0, -5.0, -0.02, 0.0, 0.0]),
            timestamp,
        ),
    ])
    recovered = {
        item.local_target_id: item.cooperative_target_id
        for item in reacquired.assignments
    }

    assert recovered["new-01"] == original["beta"]
    assert recovered["new-02"] == original["alpha"]
    assert reacquired.created_cooperative_target_ids == ()
    assert set(reacquired.reacquired_cooperative_target_ids) == set(original.values())


def test_ambiguous_reappearance_is_not_forced_onto_an_existing_identity():
    resolver = CooperativeTrackIdentityResolver(
        position_gate_m=1_000.0,
        velocity_gate_mps=100.0,
        minimum_match_margin=0.2,
    )
    resolver.resolve([
        _report("node-a", "alpha", "track-alpha", -100.0),
        _report("node-a", "beta", "track-beta", 100.0),
    ])

    ambiguous = resolver.resolve([
        _report("node-b", "unknown", "track-unknown", 0.0),
    ])

    assert len(ambiguous.created_cooperative_target_ids) == 1
    assert ambiguous.reacquired_cooperative_target_ids == ()
