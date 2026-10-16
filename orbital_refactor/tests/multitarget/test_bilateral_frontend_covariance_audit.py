import numpy as np

from experiments.audit_bilateral_frontend_covariance import (
    _summarize_modality,
    audit_bilateral_frontend_covariance,
    reported_sigma_matched_sensor_config,
)
from experiments.audit_bilateral_seed_outlier import (
    _bearing_geometry_by_truth,
)
from tracking.association.data_contracts import UnlabeledIODObservation


def test_covariance_summary_reports_bias_scale_and_coverage():
    summary = _summarize_modality(
        [np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
        [np.eye(2), np.eye(2)],
        [1.0, 1.0],
        total_count=2,
        valid_count=2,
    )

    assert summary["validFraction"] == 1.0
    assert summary["meanResidual"] == [0.0, 0.0]
    assert summary["normalizedMeanSquaredResidual"] == 0.5
    assert summary["coverageAtChiSquare95"] == 1.0


def test_short_bilateral_covariance_audit_covers_all_modalities():
    result = audit_bilateral_frontend_covariance(
        seeds=(0,), duration_seconds=1.0, sample_interval_seconds=1.0,
    )

    assert set(result["modalities"]) == {"RADAR", "INFRARED", "OPTICAL"}
    for metrics in result["modalities"].values():
        assert metrics["validCount"] > 0
        assert metrics["degreesOfFreedom"] == 2
        assert np.isfinite(metrics["normalizedMeanSquaredResidual"])


def test_reported_sigma_matched_profile_enables_physical_jitter():
    config = reported_sigma_matched_sensor_config(17)

    assert config.optical.pointing_jitter_sigma_pixels == 0.2
    assert config.infrared.pointing_jitter_sigma_pixels == 0.25
    assert config.radar.range_jitter_sigma_m == 2.5
    assert config.radar.range_rate_jitter_sigma_mps == 0.0125


def test_bearing_geometry_reports_full_rank_for_diverse_observers():
    truth = np.array([7.1e6, 0.2e6, 0.3e6, 0.0, 0.0, 0.0])
    observations = []
    for index, position in enumerate((
        np.array([6.8e6, -1.0e6, 0.1e6]),
        np.array([6.7e6, 1.2e6, -0.4e6]),
        np.array([7.8e6, 0.3e6, 1.3e6]),
    )):
        line = truth[:3] - position
        line /= np.linalg.norm(line)
        tangent_reference = np.array([0.0, 0.0, 1.0])
        if abs(float(line @ tangent_reference)) > 0.9:
            tangent_reference = np.array([0.0, 1.0, 0.0])
        first = np.cross(line, tangent_reference)
        first /= np.linalg.norm(first)
        tangent = np.vstack((first, np.cross(line, first)))
        line_covariance = tangent.T @ np.diag([1e-10, 2e-10]) @ tangent
        observations.append(UnlabeledIODObservation(
            timestamp=1.0,
            observer_id=f"observer-{index}",
            modality="LOS",
            observer_state_eci=np.concatenate((position, np.zeros(3))),
            measurement=line,
            covariance=np.diag([1e-10, 2e-10]),
            detection_group_id=f"group-{index}",
            metadata={
                "sourceModality": "OPTICAL",
                "lineOfSightCovarianceEci": line_covariance,
            },
        ))

    result = _bearing_geometry_by_truth(
        observations,
        truth_target_by_detection_group={
            f"group-{index}": "target-1" for index in range(3)
        },
        truth_state_by_id={"target-1": truth},
    )["target-1"]

    assert result["informationRank"] == 3
    assert result["observerCount"] == 3
    assert result["measurementOnlyPositionSigmaM"] > 0.0
    assert np.isfinite(result["informationConditionNumber"])
