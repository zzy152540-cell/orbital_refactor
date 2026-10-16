import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.data_contracts import TargetInitialState
from tracking.multi_epoch_bearing_filter import MultiEpochBearingWindow
from tracking.multi_observer_bearing_filter import _tangent_basis


def _observations(truth, timestamp):
    result = []
    positions = (
        np.array([6.8e6, -1.1e6, 0.1e6]),
        np.array([6.7e6, 1.3e6, -0.4e6]),
        np.array([7.9e6, 0.4e6, 1.4e6]),
    )
    for index, position in enumerate(positions):
        line = truth[:3] - position
        line /= np.linalg.norm(line)
        tangent = np.vstack(_tangent_basis(line))
        line_covariance = tangent.T @ np.diag([1e-11, 2e-11]) @ tangent
        result.append(UnlabeledIODObservation(
            timestamp=timestamp,
            observer_id=f"observer-{index}",
            modality="LOS",
            observer_state_eci=np.concatenate((position, np.zeros(3))),
            measurement=line,
            covariance=np.diag([1e-11, 2e-11]),
            detection_group_id=f"group-{timestamp}-{index}",
            metadata={
                "sourceModality": "OPTICAL",
                "sourceMessageId": f"message-{timestamp}-{index}",
                "lineOfSightCovarianceEci": line_covariance,
            },
        ))
    return tuple(result)


def test_multi_epoch_window_relinearizes_past_lines_and_returns_current_state():
    truth0 = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth0 + np.array([800.0, -500.0, 300.0, 0.4, -0.3, 0.2]),
        covariance_eci=np.diag([2e6, 2e6, 2e6, 4.0, 4.0, 4.0]),
    )
    truth1 = rk4_step_absolute(truth0, 1.0)
    truth2 = rk4_step_absolute(truth0, 2.0)
    window = MultiEpochBearingWindow(
        prior, maximum_epochs=3, process_noise_acceleration=0.0,
    )
    window.append(_observations(truth1, 1.0), timestamp=1.0)
    window.append(_observations(truth2, 2.0), timestamp=2.0)

    result = window.estimate()
    propagated_prior = rk4_step_absolute(prior.state_eci, 2.0)

    assert result.timestamp == 2.0
    assert result.measurement_count == 6
    assert np.linalg.norm(result.state_eci[:3] - truth2[:3]) < (
        np.linalg.norm(propagated_prior[:3] - truth2[:3])
    )
    assert np.linalg.eigvalsh(result.covariance_eci).min() >= -1e-8
    assert result.modality_counts == {"OPTICAL": 6}
