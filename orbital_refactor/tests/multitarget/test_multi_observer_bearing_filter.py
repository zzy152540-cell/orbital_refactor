import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.data_contracts import TargetInitialState
from tracking.multi_observer_bearing_filter import (
    _tangent_basis,
    batch_multi_observer_bearing_update,
)


def test_multi_observer_bearing_batch_uses_geometry_to_reduce_position_error():
    truth_at_zero = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([1200.0, -800.0, 500.0, 0, 0, 0]),
        covariance_eci=np.diag([2e6, 2e6, 2e6, 100.0, 100.0, 100.0]),
    )
    observer_positions = (
        np.array([6.9e6, -1.5e6, 0.2e6]),
        np.array([6.7e6, 1.8e6, -0.5e6]),
        np.array([7.8e6, 0.4e6, 1.4e6]),
    )
    observations = []
    for index, position in enumerate(observer_positions):
        observer_state = np.concatenate((position, np.zeros(3)))
        line = truth[:3] - position
        line /= np.linalg.norm(line)
        observations.append(UnlabeledIODObservation(
            timestamp=1.0,
            observer_id=f"observer-{index}",
            modality="LOS",
            observer_state_eci=observer_state,
            measurement=line,
            covariance=np.diag([1e-10, 1e-10]),
            detection_group_id=f"group-{index}",
            metadata={
                "sourceModality": "OPTICAL",
                "sourceMessageId": f"message-{index}",
            },
        ))

    result = batch_multi_observer_bearing_update(
        prior, observations, timestamp=1.0,
    )
    propagated_prior = rk4_step_absolute(prior.state_eci, 1.0)

    assert np.linalg.norm(result.state_eci[:3] - truth[:3]) < (
        np.linalg.norm(propagated_prior[:3] - truth[:3])
    )
    assert np.trace(result.covariance_eci[:3, :3]) < 6e6
    assert result.measurement_count == 3
    assert result.observer_ids == (
        "observer-0", "observer-1", "observer-2",
    )
    assert result.modality_counts == {"OPTICAL": 3}


def test_j2000_line_covariance_projects_consistently_between_tangent_bases():
    line = np.array([0.4, -0.3, 0.866025403784])
    line /= np.linalg.norm(line)
    first_basis = np.vstack(_tangent_basis(line))
    angle = 0.63
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    second_basis = rotation @ first_basis
    source_covariance = np.diag([2e-8, 7e-8])
    line_covariance = first_basis.T @ source_covariance @ first_basis

    first_projection = first_basis @ line_covariance @ first_basis.T
    second_projection = second_basis @ line_covariance @ second_basis.T

    assert np.allclose(
        second_projection,
        rotation @ first_projection @ rotation.T,
        atol=1e-20,
    )
    assert np.linalg.eigvalsh(line_covariance).min() >= -1e-20


def test_multi_observer_update_uses_j2000_line_covariance_metadata():
    truth_at_zero = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([500.0, -300.0, 200.0, 0, 0, 0]),
        covariance_eci=np.diag([2e6, 2e6, 2e6, 100.0, 100.0, 100.0]),
    )
    observations = []
    for index, position in enumerate((
        np.array([6.9e6, -1.5e6, 0.2e6]),
        np.array([6.7e6, 1.8e6, -0.5e6]),
    )):
        line = truth[:3] - position
        line /= np.linalg.norm(line)
        tangent = np.vstack(_tangent_basis(line))
        line_covariance = tangent.T @ np.diag([1e-10, 3e-10]) @ tangent
        observations.append(UnlabeledIODObservation(
            timestamp=1.0,
            observer_id=f"observer-{index}",
            modality="LOS",
            observer_state_eci=np.concatenate((position, np.zeros(3))),
            measurement=line,
            covariance=np.diag([1e3, 1e3]),
            detection_group_id=f"group-{index}",
            metadata={"lineOfSightCovarianceEci": line_covariance},
        ))

    with_metadata = batch_multi_observer_bearing_update(
        prior, observations, timestamp=1.0,
    )
    without_metadata = batch_multi_observer_bearing_update(
        prior,
        [
            UnlabeledIODObservation(
                timestamp=item.timestamp,
                observer_id=item.observer_id,
                modality=item.modality,
                observer_state_eci=item.observer_state_eci,
                measurement=item.measurement,
                covariance=item.covariance,
                detection_group_id=item.detection_group_id,
            )
            for item in observations
        ],
        timestamp=1.0,
    )

    assert np.trace(with_metadata.covariance_eci[:3, :3]) < (
        np.trace(without_metadata.covariance_eci[:3, :3])
    )
    assert np.linalg.eigvalsh(with_metadata.covariance_eci).min() >= -1e-8
