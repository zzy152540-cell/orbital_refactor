import numpy as np

from experiments.audit_bilateral_seed_outlier import (
    _phase_geometry_diagnostics,
    _phase_rtn_diagnostics,
    _rtn_error_diagnostics,
)


def test_rtn_diagnostics_resolve_directional_error_and_consistency():
    truth = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0])
    estimate = truth + np.array([3.0, 4.0, 12.0, 0.1, 0.2, 0.3])
    covariance = np.diag([9.0, 16.0, 144.0, 1.0, 1.0, 1.0])

    sample = {
        "timestamp": 6.0,
        **_rtn_error_diagnostics(
            estimate_state=estimate,
            truth_state=truth,
            covariance_eci=covariance,
        ),
    }

    np.testing.assert_allclose(sample["positionErrorRtnM"], [3.0, 4.0, 12.0])
    np.testing.assert_allclose(
        sample["positionNormalizedSquaredErrorRtn"], [1.0, 1.0, 1.0]
    )
    assert np.isclose(sample["positionNees"], 3.0)

    phase = _phase_rtn_diagnostics(
        {"target": [sample]}, start=6.0, end=13.0
    )
    assert phase["sampleCount"] == 1
    np.testing.assert_allclose(phase["positionRmseRtnM"], [3.0, 4.0, 12.0])
    np.testing.assert_allclose(phase["meanPositionSigmaRtnM"], [3.0, 4.0, 12.0])
    assert np.isclose(phase["meanPositionNees"], 3.0)


def test_phase_geometry_diagnostics_summarizes_weak_direction():
    samples = [{
        "timestamp": 6.0,
        "positionErrorM": 2.0,
        "bearingGeometry": {
            "observerCount": 3,
            "informationConditionNumber": 10.0,
            "weakestInformationEigenvaluePerM2": 0.25,
            "weakestDirectionAbsoluteRtn": [0.9, 0.3, 0.1],
        },
    }, {
        "timestamp": 7.0,
        "positionErrorM": 3.0,
        "bearingGeometry": {
            "observerCount": 5,
            "informationConditionNumber": 20.0,
            "weakestInformationEigenvaluePerM2": 0.5,
            "weakestDirectionAbsoluteRtn": [0.7, 0.5, 0.2],
        },
    }]

    result = _phase_geometry_diagnostics(
        {"target": samples}, start=6.0, end=7.0,
    )

    assert result["sampleCount"] == 2
    assert np.isclose(result["meanObserverCount"], 4.0)
    assert np.isclose(result["meanInformationConditionNumber"], 15.0)
    np.testing.assert_allclose(
        result["meanWeakestDirectionAbsoluteRtn"], [0.8, 0.4, 0.15],
    )
