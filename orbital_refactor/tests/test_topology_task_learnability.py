import numpy as np

from experiments.topology_ppo_stage1 import Stage1Configuration
from experiments.topology_task_learnability import (
    audit_stage1_task_learnability,
    audit_stage1_horizon_stability,
)


def test_stage1_learnability_audit_enumerates_reproducible_decisions():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4,
        decision_interval_epochs=2, environment_seed_count=1,
    )
    left = audit_stage1_task_learnability(configuration, seeds=(10, 11))
    right = audit_stage1_task_learnability(configuration, seeds=(10, 11))
    assert left == right
    assert len(left.opportunities) == 4
    assert sum(count for _, count in left.best_raw_kind_counts) == 4
    assert sum(count for _, count in left.best_kind_counts) == 4
    assert sum(
        count for _, count in left.best_raw_action_signature_counts
    ) == 4
    assert sum(count for _, count in left.best_action_signature_counts) == 4
    assert all(item.legal_action_count >= 1 for item in left.opportunities)
    assert 0.0 <= left.nontrivial_raw_fraction <= 1.0
    assert 0.0 <= left.nontrivial_penalized_fraction <= 1.0
    assert np.isfinite(left.median_best_to_second_margin)


def test_stage1_learnability_audit_validates_seed_and_gain_inputs():
    configuration = Stage1Configuration(training_episodes=1)
    for seeds, gain in (((), 0.0), ((1, 1), 0.0), ((1,), -1.0)):
        try:
            audit_stage1_task_learnability(
                configuration, seeds=seeds, minimum_meaningful_gain=gain,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid learnability audit input was accepted.")


def test_horizon_stability_audit_reports_action_agreement():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4, decision_interval_epochs=2,
    )
    audit = audit_stage1_horizon_stability(
        configuration, seeds=(10,), lookahead_steps=(1, 2),
    )
    assert audit.decision_count == 2
    assert tuple(name for name, _ in audit.action_agreement_by_horizon_pair) == (
        "1_vs_2",
    )
    assert all(
        0.0 <= value <= 1.0
        for _, value in audit.action_agreement_by_horizon_pair
    )
    assert sum(
        count for _, count in audit.kind_transition_counts_one_to_longest
    ) == 2
