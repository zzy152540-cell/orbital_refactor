from experiments.walker_navigation_brain_state_scenario_audit import (
    run_walker_navigation_scenario_audit,
)


def test_short_navigation_scenario_audit_aggregates_cases():
    result = run_walker_navigation_scenario_audit(
        duration=4.0, dt=2.0,
        case_specs=(("a", 0, 1, None), ("b", 1, 2, None)),
    )
    assert set(result.cases) == {"a", "b"}
    assert result.summary["case_count"] == 2
    assert result.summary["all_cases_valid"]
    assert result.summary["minimum_effective_rank"] > 0.0
