from experiments.summarize_bilateral_robustness import summarize_robustness


def _phase(value):
    return {
        "position_sample_count": 15,
        "position_mean_error_m": value * 0.9,
        "velocity_mean_error_mps": value * 0.09,
        "acceleration_mean_error_mps2": value * 0.009,
        "position_rmse_m": value,
        "velocity_rmse_mps": value / 10.0,
        "acceleration_rmse_mps2": value / 100.0,
    }


def _arm(value, *, consensus=False):
    phase_metrics = {
        "DROPOUT": _phase(value),
        "RECOVERY": _phase(value / 2.0),
    }
    return {
        "phase_metrics": phase_metrics,
        "consensus_phase_metrics": (
            {
                "DROPOUT": _phase(value / 2.0),
                "RECOVERY": _phase(value / 4.0),
            }
            if consensus else {}
        ),
        "independent_node_metrics": {"phase_metrics": phase_metrics},
        "consensus_position_rmse_m_by_target": (
            {f"target-{index}": value for index in range(15)}
            if consensus else {}
        ),
        "final_active_track_count": 15,
        "retired_track_count": 0,
        "identity_switch_count": 0,
        "consensus_identity_switch_count": 0,
        "consensus_recovered_link_count": 12,
        "processing_mean_ms": 20.0,
        "missing_truth_sample_count": 0,
    }


def _dropout_payload(seed):
    result = {}
    for modality in ("OPTICAL", "INFRARED", "RADAR"):
        result[modality] = {
            "withoutCann": _arm(10.0),
            "incrementalCann": _arm(9.0),
            "dropoutVsControlPercent": {
                arm: {field: 5.0 for field in _phase(1.0)}
                for arm in ("withoutCann", "incrementalCann")
            },
            "pairedCannDeltaPercent": {
                field: -10.0 for field in _phase(1.0)
            },
        }
    return {
        "scenario": {"seed": seed, "hostileTargetCount": 15},
        "noDropoutControl": {
            "withoutCann": _arm(10.0),
            "incrementalCann": _arm(9.0),
        },
        "resultsByMissingModality": result,
    }


def _failure_payload(seed):
    return {
        "scenario": {"seed": seed, "hostileTargetCount": 15},
        "noFailureControl": {
            "withoutCann": _arm(10.0, consensus=True),
            "incrementalCann": _arm(9.0, consensus=True),
        },
        "threeObserverFailure": {
            "withoutCann": _arm(10.0, consensus=True),
            "incrementalCann": _arm(9.0, consensus=True),
        },
        "failureVsControlPercent": {
            arm: {field: 2.0 for field in _phase(1.0)}
            for arm in ("withoutCann", "incrementalCann")
        },
    }


def test_robustness_summary_aggregates_both_scenario_contracts():
    summary = summarize_robustness(
        [_dropout_payload(0), _dropout_payload(1)],
        [_failure_payload(0), _failure_payload(1)],
    )

    assert set(summary["modalityDropout"]) == {
        "OPTICAL", "INFRARED", "RADAR",
    }
    assert summary["dropoutSeeds"] == [0, 1]
    assert summary["nodeFailureSeeds"] == [0, 1]
    assert summary["modalityDropout"]["OPTICAL"][
        "pairedCannDeltaPercent"
    ]["position_rmse_m"]["mean"] == -10.0
    assert summary["threeObserverFailure"]["arms"][
        "incrementalCann"
    ]["recoveredLinkCount"]["mean"] == 12.0
    assert summary["modalityDropout"]["OPTICAL"]["acceptance"][
        "incrementalCann"
    ]["passed"]
    assert summary["modalityDropout"]["OPTICAL"]["acceptanceMetricScope"] == (
        "POOLED_ALL_SEEDS_ALL_TARGET_TIME_SAMPLES"
    )
    assert summary["modalityDropout"]["OPTICAL"][
        "pooledDropoutVsControl"
    ]["incrementalCann"]["position_rmse_m"]["increasePercent"] == 0.0
    assert summary["modalityDropout"]["OPTICAL"][
        "pooledDropoutVsControl"
    ]["incrementalCann"]["position_rmse_m"][
        "pairedSeedBootstrap95Percent"
    ] == [0.0, 0.0]
    assert summary["threeObserverFailure"]["acceptance"][
        "withoutCann"
    ]["passed"]
    assert summary["threeObserverFailure"]["acceptanceMetricScope"] == (
        "SURVIVING_NODE_CONSENSUS_ALL_TARGET_SAMPLES"
    )
    assert summary["threeObserverFailure"][
        "consensusFailureVsControlPercent"
    ]["incrementalCann"]["position_mean_error_m"]["mean"] == 0.0
