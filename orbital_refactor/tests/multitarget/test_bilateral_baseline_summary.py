import pytest

from experiments.summarize_bilateral_baseline_matrix import summarize_payloads


def _payload(seed, position):
    arm = {
        "final_position_rmse_m": position,
        "final_maximum_position_error_m": position * 2.0,
        "final_velocity_rmse_mps": position / 10.0,
        "final_acceleration_rmse_mps2": position / 100.0,
        "worst_target_position_rmse_m": position * 1.5,
        "processing_mean_ms": 10.0,
        "processing_p95_ms": 12.0,
        "identity_switch_count": 0,
        "nearest_truth_disagreement_count": 0,
        "track_replacement_count": 0,
        "missing_truth_sample_count": 0,
        "consensus_worst_target_position_rmse_m": None,
        "consensus_identity_switch_count": 0,
        "consensus_nearest_truth_disagreement_count": 0,
        "consensus_track_replacement_count": 0,
        "consensus_missing_truth_sample_count": 0,
        "position_rmse_m_by_target": {"hostile-1": position},
        "consensus_position_rmse_m_by_target": {},
    }
    return {
        "scenario": {"seed": seed, "durationSeconds": 20.0},
        "arms": {"example": arm},
    }


def test_summary_reports_cross_seed_statistics_and_target_values():
    summary = summarize_payloads((_payload(0, 2.0), _payload(1, 4.0)))

    metric = summary["arms"]["example"]["metrics"][
        "final_position_rmse_m"
    ]
    target = summary["arms"]["example"]["positionRmseByTarget"][
        "hostile-1"
    ]
    assert summary["seeds"] == [0, 1]
    assert metric == {
        "count": 2, "mean": 3.0, "std": 1.0, "min": 2.0, "max": 4.0,
    }
    assert target["mean"] == 3.0
    assert summary["arms"]["example"]["consensusFullRunMetrics"][
        "position_rmse_m"
    ]["count"] == 0
    assert summary["arms"]["example"]["identityAudit"][
        "switchTotal"
    ] == 0


def test_summary_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        summarize_payloads((_payload(0, 2.0), _payload(0, 4.0)))
