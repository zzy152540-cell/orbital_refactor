from experiments.multitarget_autonomous_identity_acceptance import (
    run_multitarget_autonomous_identity_acceptance,
)


def test_raw_unlabeled_multimodal_tracks_reacquire_global_identities():
    result = run_multitarget_autonomous_identity_acceptance()

    assert result.minimum_target_separation_m < 100.0
    assert result.initial_birth_count == 6
    assert result.post_outage_birth_count == 6
    assert result.identity_switch_count == 0
    assert result.false_cooperative_birth_count == 0
    assert result.reacquired_cooperative_target_count == 2
    assert result.final_tracking_target_count == 2
    assert result.final_consensus_target_count == 2
    assert result.maximum_final_position_error_m < 2_000.0
    assert result.passed
