import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from tracking import (
    GlobalTargetEstimate,
    ManeuverDetectionConfig,
    ManeuverDetector,
    MultiTargetOutput,
    TargetInitialState,
    TrackLifecycle,
    TrackManager,
)


def _initial(timestamp=0.0, state=None):
    return TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=timestamp,
        state_eci=(
            np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
            if state is None else state
        ),
        covariance_eci=np.eye(6),
    )


def _posterior(prior, *, position_jump=0.0):
    timestamp = prior.timestamp + 10.0
    state = rk4_step_absolute(prior.state_eci, 10.0)
    state[0] += position_jump
    return GlobalTargetEstimate(
        target_id=prior.target_id,
        track_id=prior.track_id,
        timestamp=timestamp,
        state_eci=state,
        covariance_eci=np.eye(6),
        contributing_observer_ids=("observer-1",),
        node_weights={"observer-1": 1.0},
    )


def _as_prior(value):
    return TargetInitialState(
        target_id=value.target_id,
        track_id=value.track_id,
        timestamp=value.timestamp,
        state_eci=value.state_eci,
        covariance_eci=value.covariance_eci,
    )


def test_persistent_model_mismatch_sets_and_clears_maneuver_state():
    detector = ManeuverDetector(ManeuverDetectionConfig(
        nis_threshold=20.0,
        confirmation_epochs=2,
        clearing_epochs=2,
        adaptive_process_noise_scale=40.0,
    ))
    first_prior = _initial()
    first = detector.assess(first_prior, _posterior(first_prior, position_jump=1000.0))
    second_prior = _as_prior(_posterior(first_prior, position_jump=1000.0))
    second_posterior = _posterior(second_prior, position_jump=1000.0)
    second = detector.assess(second_prior, second_posterior)

    assert first.threshold_exceeded and not first.suspected
    assert second.suspected
    assert second.process_noise_scale == 40.0

    third_prior = _as_prior(second_posterior)
    third_posterior = _posterior(third_prior)
    third = detector.assess(third_prior, third_posterior)
    fourth_prior = _as_prior(third_posterior)
    fourth = detector.assess(fourth_prior, _posterior(fourth_prior))

    assert third.suspected
    assert not fourth.suspected
    assert fourth.process_noise_scale == 1.0


def test_track_manager_exposes_maneuver_suspected_lifecycle():
    initial = _initial()
    estimate = _posterior(initial, position_jump=1000.0)
    manager = TrackManager(
        scene_id="scene", initial_states={initial.target_id: initial},
    )
    output = MultiTargetOutput(
        scene_id="scene",
        timestamp=estimate.timestamp,
        estimates_by_target={estimate.target_id: estimate},
    )

    tracks = manager.step(
        output, maneuver_suspected_target_ids=("target-1",),
    )

    assert tracks["target-1"].lifecycle is TrackLifecycle.MANEUVER_SUSPECTED

