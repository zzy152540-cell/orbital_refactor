import json

from experiments.online_multitarget_local_acceptance import (
    OnlineAcceptanceConfig,
    run_online_multitarget_local_acceptance,
)


def test_local_online_acceptance_runs_paired_incremental_cann_chain():
    result = run_online_multitarget_local_acceptance(OnlineAcceptanceConfig(
        observer_count=2,
        target_count=1,
        duration_seconds=5.0,
        sample_interval_seconds=1.0,
        seed=3,
        image_size=17,
        maximum_final_position_error_m=5_000.0,
    ))
    payload = result.to_dict()

    assert result.without_cann.algorithm_passed
    assert result.incremental_cann.algorithm_passed
    assert result.incremental_cann.cann_preprocessed_observation_count > 0
    assert result.incremental_cann.cann_preprocessing_mean_ms > 0.0
    assert result.without_cann.cann_preprocessed_observation_count == 0
    assert payload["incrementalCann"]["finalActiveTargetCount"] == 1
    assert payload["incrementalCann"]["identitySwitchCount"] == 0
    assert payload["inputGenerationMeanMs"] > 0.0
    assert json.loads(json.dumps(payload))["config"]["sample_interval_seconds"] == 1.0


def test_local_online_acceptance_recovers_from_short_target_outage():
    result = run_online_multitarget_local_acceptance(OnlineAcceptanceConfig(
        observer_count=2,
        target_count=1,
        duration_seconds=7.0,
        sample_interval_seconds=1.0,
        seed=4,
        image_size=17,
        maximum_final_position_error_m=5_000.0,
        dropout_start_seconds=4.0,
        dropout_end_seconds=5.0,
        dropout_modality="ALL",
        dropout_target_index=0,
    ))

    assert result.without_cann.algorithm_passed
    assert result.incremental_cann.algorithm_passed
    assert result.without_cann.initialized_target_count == 1
    assert result.incremental_cann.initialized_target_count == 1
    assert result.without_cann.retired_target_count == 0
    assert result.incremental_cann.retired_target_count == 0
