import numpy as np
import pytest

from orbital_core.dynamics import rk4_step_absolute
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.data_contracts import TargetInitialState
from tracking.inverse_depth_bearing_filter import (
    inverse_depth_multi_observer_bearing_update,
)


def _line_observation(truth, observer, index):
    line = truth[:3] - observer[:3]
    line /= np.linalg.norm(line)
    return UnlabeledIODObservation(
        timestamp=1.0,
        observer_id=f"observer-{index}",
        modality="LOS",
        observer_state_eci=observer,
        measurement=line,
        covariance=np.diag([1e-12, 1e-12]),
        detection_group_id=f"group-{index}",
        metadata={
            "sourceModality": "OPTICAL",
            "sourceMessageId": f"los-{index}",
        },
    )


def test_inverse_depth_update_reduces_a_large_range_biased_prior():
    truth_zero = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    truth = rk4_step_absolute(truth_zero, 1.0)
    radial = truth_zero[:3] / np.linalg.norm(truth_zero[:3])
    prior_state = truth_zero.copy()
    prior_state[:3] += 10_000.0 * radial
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=prior_state,
        covariance_eci=np.diag([1e8, 1e8, 1e8, 100.0, 100.0, 100.0]),
    )
    observers = (
        np.array([6.9e6, -1.5e6, 0.2e6, 20.0, 7300.0, -10.0]),
        np.array([6.7e6, 1.8e6, -0.5e6, -30.0, 7200.0, 20.0]),
        np.array([7.4e6, 0.6e6, 1.4e6, 15.0, 7350.0, -25.0]),
    )
    observations = tuple(
        _line_observation(truth, observer, index)
        for index, observer in enumerate(observers)
    )

    update = inverse_depth_multi_observer_bearing_update(
        prior, observations, timestamp=1.0,
    )
    propagated_prior = rk4_step_absolute(prior.state_eci, 1.0)

    assert np.linalg.norm(update.state_eci[:3] - truth[:3]) < np.linalg.norm(
        propagated_prior[:3] - truth[:3]
    )
    assert update.measurement_count == 3
    assert update.modality_counts == {"OPTICAL": 3}
    assert update.measurement_ids == ("los-0", "los-1", "los-2")
    assert np.all(np.isfinite(update.state_eci))
    assert np.linalg.eigvalsh(update.covariance_eci).min() >= -1e-8


def test_inverse_depth_update_requires_multiple_observers():
    truth = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    observer = np.array([6.9e6, -1.5e6, 0.2e6, 20.0, 7300.0, -10.0])
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth,
        covariance_eci=np.eye(6),
    )

    with pytest.raises(ValueError, match="At least two distinct observers"):
        inverse_depth_multi_observer_bearing_update(
            prior,
            (_line_observation(rk4_step_absolute(truth, 1.0), observer, 0),),
            timestamp=1.0,
        )
