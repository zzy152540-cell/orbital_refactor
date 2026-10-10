from experiments.multitarget_identity_reacquisition_acceptance import (
    run_multitarget_identity_reacquisition_acceptance,
)


def test_close_crossing_and_full_outage_preserve_cooperative_identities():
    result = run_multitarget_identity_reacquisition_acceptance()

    assert result.minimum_target_separation_m < 100.0
    assert result.identity_switch_count == 0
    assert result.false_cooperative_birth_count == 0
    assert result.reacquired_cooperative_target_count == 2
    assert result.final_cooperative_target_count == 2
    assert result.passed


def test_reacquisition_acceptance_rejects_invalid_outage_interval():
    try:
        run_multitarget_identity_reacquisition_acceptance(
            outage_start=300.0,
            outage_end=200.0,
        )
    except ValueError as error:
        assert "outage" in str(error).lower()
    else:
        raise AssertionError("Expected an invalid outage interval to fail.")
