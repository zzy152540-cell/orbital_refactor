import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.data_contracts import TargetInitialState
from tracking.multi_observer_multimodal_filter import (
    unified_multi_observer_multimodal_update,
)
from tracking.multi_epoch_multimodal_filter import FixedLagMultimodalWindow


def test_unified_update_combines_radar_and_los_on_one_state_prior():
    truth_at_zero = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([100.0, -80.0, 60.0, 0.2, -0.1, 0.1]),
        covariance_eci=np.diag([1e5, 1e5, 1e5, 10.0, 10.0, 10.0]),
    )
    observer_states = (
        np.array([6.9e6, -1.5e6, 0.2e6, 20.0, 7300.0, -10.0]),
        np.array([6.7e6, 1.8e6, -0.5e6, -30.0, 7200.0, 20.0]),
    )
    observations = []
    for index, observer in enumerate(observer_states):
        relative_position = truth[:3] - observer[:3]
        relative_velocity = truth[3:] - observer[3:]
        distance = np.linalg.norm(relative_position)
        line = relative_position / distance
        observations.extend((
            UnlabeledIODObservation(
                timestamp=1.0,
                observer_id=f"observer-{index}",
                modality="RADAR",
                observer_state_eci=observer,
                measurement=np.array([
                    distance, line @ relative_velocity,
                ]),
                covariance=np.diag([4.0, 1e-4]),
                detection_group_id=f"group-{index}",
                metadata={
                    "sourceModality": "RADAR",
                    "sourceMessageId": f"radar-{index}",
                },
            ),
            UnlabeledIODObservation(
                timestamp=1.0,
                observer_id=f"observer-{index}",
                modality="LOS",
                observer_state_eci=observer,
                measurement=line,
                covariance=np.diag([1e-10, 1e-10]),
                detection_group_id=f"group-{index}",
                metadata={
                    "sourceModality": "OPTICAL",
                    "sourceMessageId": f"optical-{index}",
                },
            ),
        ))

    result = unified_multi_observer_multimodal_update(
        prior, observations, timestamp=1.0,
    )
    propagated_prior = rk4_step_absolute(prior.state_eci, 1.0)

    assert np.linalg.norm(result.state_eci[:3] - truth[:3]) < np.linalg.norm(
        propagated_prior[:3] - truth[:3]
    )
    assert np.linalg.norm(result.state_eci[3:] - truth[3:]) < np.linalg.norm(
        propagated_prior[3:] - truth[3:]
    )
    assert result.modality_counts == {"OPTICAL": 2, "RADAR": 2}
    assert result.information_capabilities == (
        "RADIAL_RANGE", "RADIAL_RANGE_RATE",
    )
    assert result.observation_count == 4
    assert result.measurement_dimension == 8
    assert np.linalg.eigvalsh(result.covariance_eci).min() >= -1e-8


def test_unified_update_accepts_los_only_when_radar_is_missing():
    truth_at_zero = np.array([7.0e6, 1.0e5, 2.0e5, -100.0, 7500.0, 30.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([30.0, -20.0, 10.0, 0, 0, 0]),
        covariance_eci=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
    )
    observations = []
    for index, position in enumerate((
        np.array([6.8e6, -1.0e6, 0.0]),
        np.array([7.4e6, 1.2e6, 0.5e6]),
    )):
        line = truth[:3] - position
        line /= np.linalg.norm(line)
        observations.append(UnlabeledIODObservation(
            timestamp=1.0,
            observer_id=f"observer-{index}",
            modality="LOS",
            observer_state_eci=np.concatenate((position, np.zeros(3))),
            measurement=line,
            covariance=np.diag([1e-10, 1e-10]),
            detection_group_id=f"group-{index}",
            metadata={"sourceModality": "INFRARED"},
        ))

    result = unified_multi_observer_multimodal_update(
        prior, observations, timestamp=1.0,
    )

    assert result.modality_counts == {"INFRARED": 2}
    assert result.information_capabilities == ()
    assert result.observation_count == 2
    assert np.all(np.isfinite(result.state_eci))


def test_los_range_proxy_adds_one_radial_measurement_dimension():
    truth_at_zero = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([100.0, 0.0, 0.0, 0, 0, 0]),
        covariance_eci=np.diag([1e4, 100.0, 100.0, 1.0, 1.0, 1.0]),
    )
    observer = np.array([6.8e6, -1.0e6, 0.0, 0.0, 0.0, 0.0])
    relative = truth[:3] - observer[:3]
    distance = float(np.linalg.norm(relative))
    line = relative / distance
    observation = UnlabeledIODObservation(
        timestamp=1.0,
        observer_id="observer-0",
        modality="LOS",
        observer_state_eci=observer,
        measurement=line,
        covariance=np.diag([1e-10, 1e-10]),
        detection_group_id="group-0",
        metadata={
            "sourceModality": "INFRARED",
            "rangeProxyM": distance,
            "rangeProxyVarianceM2": 1.0,
        },
    )

    result = unified_multi_observer_multimodal_update(
        prior, (observation,), timestamp=1.0,
    )
    propagated = rk4_step_absolute(prior.state_eci, 1.0)

    assert result.measurement_dimension == 3
    assert result.information_capabilities == ("RADIAL_RANGE",)
    assert np.linalg.norm(result.state_eci[:3] - truth[:3]) < np.linalg.norm(
        propagated[:3] - truth[:3]
    )


def test_los_range_rate_proxy_adds_direct_radial_velocity_constraint():
    truth_at_zero = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([50.0, 0.0, 0.0, 2.0, 0, 0]),
        covariance_eci=np.diag([1e4, 100.0, 100.0, 10.0, 1.0, 1.0]),
    )
    observer = np.array([6.8e6, -1.0e6, 0.0, 0.0, 0.0, 0.0])
    relative = truth[:3] - observer[:3]
    relative_velocity = truth[3:] - observer[3:]
    distance = float(np.linalg.norm(relative))
    line = relative / distance
    range_rate = float(line @ relative_velocity)
    observation = UnlabeledIODObservation(
        timestamp=1.0,
        observer_id="observer-0",
        modality="LOS",
        observer_state_eci=observer,
        measurement=line,
        covariance=np.diag([1e-10, 1e-10]),
        detection_group_id="group-0",
        metadata={
            "sourceModality": "INFRARED",
            "rangeProxyM": distance,
            "rangeProxyVarianceM2": 1.0,
            "rangeRateProxyMps": range_rate,
            "rangeRateProxyVarianceM2ps2": 1e-4,
        },
    )

    result = unified_multi_observer_multimodal_update(
        prior, (observation,), timestamp=1.0,
    )
    propagated = rk4_step_absolute(prior.state_eci, 1.0)

    assert result.measurement_dimension == 4
    assert result.information_capabilities == (
        "RADIAL_RANGE", "RADIAL_RANGE_RATE",
    )
    assert np.linalg.norm(result.state_eci[3:] - truth[3:]) < np.linalg.norm(
        propagated[3:] - truth[3:]
    )


def test_electro_optical_radial_proxies_fail_over_without_double_counting():
    truth_at_zero = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    truth = rk4_step_absolute(truth_at_zero, 1.0)
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_at_zero + np.array([50.0, 0.0, 0.0, 1.0, 0, 0]),
        covariance_eci=np.diag([1e4, 100.0, 100.0, 10.0, 1.0, 1.0]),
    )
    observer = np.array([6.8e6, -1.0e6, 0.0, 0.0, 0.0, 0.0])
    relative = truth[:3] - observer[:3]
    distance = float(np.linalg.norm(relative))
    line = relative / distance
    range_rate = float(line @ (truth[3:] - observer[3:]))

    def los(source, priority):
        return UnlabeledIODObservation(
            timestamp=1.0,
            observer_id="observer-0",
            modality="LOS",
            observer_state_eci=observer,
            measurement=line,
            covariance=np.diag([1e-10, 1e-10]),
            detection_group_id="group-0",
            metadata={
                "sourceModality": source,
                "rangeProxyM": distance,
                "rangeProxyVarianceM2": 4.0,
                "rangeRateProxyMps": range_rate,
                "rangeRateProxyVarianceM2ps2": 1e-4,
                "radialProxyFusionPolicy": "radar_primary_failover",
                "radialProxyFailoverPriority": priority,
            },
        )

    optical = los("OPTICAL", 2)
    infrared = los("INFRARED", 1)
    fallback = unified_multi_observer_multimodal_update(
        prior, (optical, infrared), timestamp=1.0,
    )
    assert fallback.measurement_dimension == 6

    radar = UnlabeledIODObservation(
        timestamp=1.0,
        observer_id="observer-0",
        modality="RADAR",
        observer_state_eci=observer,
        measurement=np.array([distance, range_rate]),
        covariance=np.diag([4.0, 1e-4]),
        detection_group_id="group-0",
        metadata={"sourceModality": "RADAR"},
    )
    primary = unified_multi_observer_multimodal_update(
        prior, (optical, infrared, radar), timestamp=1.0,
    )
    assert primary.measurement_dimension == 6


def test_fixed_lag_window_slides_without_reusing_the_oldest_measurements():
    truth_zero = np.array([7.1e6, 2.0e5, 3.0e5, -150.0, 7450.0, 50.0])
    prior = TargetInitialState(
        target_id="target-1",
        track_id="track-1",
        timestamp=0.0,
        state_eci=truth_zero + np.array([20.0, -15.0, 10.0, 0.1, -0.1, 0.05]),
        covariance_eci=np.diag([400.0, 400.0, 400.0, 1.0, 1.0, 1.0]),
    )
    observers = (
        np.array([6.9e6, -1.5e6, 0.2e6, 20.0, 7300.0, -10.0]),
        np.array([6.7e6, 1.8e6, -0.5e6, -30.0, 7200.0, 20.0]),
    )
    window = FixedLagMultimodalWindow(
        prior, maximum_epochs=2, process_noise_acceleration=1e-4,
    )
    truth = truth_zero.copy()
    for epoch in (1.0, 2.0, 3.0):
        truth = rk4_step_absolute(truth, 1.0)
        observations = []
        for index, observer in enumerate(observers):
            relative_position = truth[:3] - observer[:3]
            relative_velocity = truth[3:] - observer[3:]
            distance = np.linalg.norm(relative_position)
            line = relative_position / distance
            observations.extend((
                UnlabeledIODObservation(
                    timestamp=epoch,
                    observer_id=f"observer-{index}",
                    modality="RADAR",
                    observer_state_eci=observer,
                    measurement=np.array([distance, line @ relative_velocity]),
                    covariance=np.diag([4.0, 1e-4]),
                    detection_group_id=f"radar-{epoch}-{index}",
                    metadata={
                        "sourceModality": "RADAR",
                        "sourceMessageId": f"radar-{epoch}-{index}",
                    },
                ),
                UnlabeledIODObservation(
                    timestamp=epoch,
                    observer_id=f"observer-{index}",
                    modality="LOS",
                    observer_state_eci=observer,
                    measurement=line,
                    covariance=np.diag([1e-10, 1e-10]),
                    detection_group_id=f"los-{epoch}-{index}",
                    metadata={
                        "sourceModality": "OPTICAL",
                        "sourceMessageId": f"los-{epoch}-{index}",
                    },
                ),
            ))
        window.append(observations, timestamp=epoch)

    result = window.estimate()

    assert window.epoch_count == 2
    assert window.anchor_timestamp == 1.0
    assert result.timestamp == 3.0
    assert result.observation_count == 8
    assert len(result.measurement_ids) == 12
    assert np.linalg.norm(result.state_eci[:3] - truth[:3]) < 5.0
    assert np.linalg.eigvalsh(result.covariance_eci).min() >= -1e-8
