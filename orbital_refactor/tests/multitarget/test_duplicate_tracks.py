import numpy as np

from tracking import (
    DuplicateTrackConfig,
    DuplicateTrackResolver,
    TargetInitialState,
    TargetTrack,
    TargetTrackKey,
    TrackLifecycle,
)


def _state(target_id, track_id, offset_m=0.0, velocity_offset=0.0):
    return TargetInitialState(
        target_id=target_id,
        track_id=track_id,
        timestamp=0.0,
        state_eci=np.array([
            7.0e6 + offset_m, 0.0, 0.0,
            velocity_offset, 7500.0, 0.0,
        ]),
        covariance_eci=np.diag([400.0] * 3 + [0.25] * 3),
    )


def test_duplicate_birth_is_aliased_to_existing_stable_track():
    existing_state = _state("existing", "track-existing")
    existing = TargetTrack(
        key=TargetTrackKey("scene", "track-existing"),
        target_id="existing",
        lifecycle=TrackLifecycle.TRACKING,
        estimate=existing_state,
    )
    duplicate = _state("new-candidate", "track-new", offset_m=20.0)
    resolver = DuplicateTrackResolver(DuplicateTrackConfig(
        maximum_position_separation_m=100.0,
        maximum_velocity_separation_mps=1.0,
        maximum_squared_mahalanobis=20.0,
    ))

    result = resolver.resolve(
        {duplicate.target_id: duplicate}, {"existing": existing},
    )

    assert result.accepted_births == {}
    assert result.aliases == {"new-candidate": "existing"}


def test_distinct_birth_is_kept_as_an_independent_track():
    existing_state = _state("existing", "track-existing")
    existing = TargetTrack(
        key=TargetTrackKey("scene", "track-existing"),
        target_id="existing", lifecycle=TrackLifecycle.TRACKING,
        estimate=existing_state,
    )
    distinct = _state("distinct", "track-distinct", offset_m=5000.0)

    result = DuplicateTrackResolver().resolve(
        {distinct.target_id: distinct}, {"existing": existing},
    )

    assert result.aliases == {}
    assert result.accepted_births == {"distinct": distinct}
