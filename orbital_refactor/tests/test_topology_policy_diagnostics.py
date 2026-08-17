from copy import deepcopy

import numpy as np

from experiments.topology_policy_diagnostics import (
    audit_policy_overrides_against_robust_targets,
    audit_topology_policy_margins,
)
from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_snapshot_counterfactual import (
    build_noise_robust_topology_snapshot_tensor_dataset,
)
from experiments.topology_ppo_stage1 import (
    Stage1Configuration,
    build_stage1_environment,
    train_stage1_ppo,
)


def test_policy_margin_audit_compares_models_on_same_deterministic_path():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4, decision_interval_epochs=2,
        update_epochs=1,
    )
    model = train_stage1_ppo(configuration).model
    records = audit_topology_policy_margins(
        build_stage1_environment(configuration), model, deepcopy(model),
        condition_seeds=(10,), noise_seeds=(2,),
    )
    assert len(records) == 2
    assert all(record.selected_action_id == record.reference_action_id
               for record in records)
    assert all(np.isclose(record.policy_kl_from_reference, 0.0, atol=1e-7)
               for record in records)
    assert all(record.selected_probability > 0.0 for record in records)
    assert all(record.log_probability_margin >= 0.0 for record in records)
    assert all(record.reference_log_probability_margin >= 0.0
               for record in records)
    assert all(record.legal_action_count >= 1 for record in records)
    assert np.isinf(records[-1].log_probability_margin) == (
        records[-1].legal_action_count == 1
    )
    assert all(np.isclose(sum(record.type_probabilities), 1.0)
               for record in records)


def test_policy_margin_audit_rejects_duplicate_seed_axes():
    configuration = Stage1Configuration(training_episodes=1, update_epochs=1)
    model = train_stage1_ppo(configuration).model
    try:
        audit_topology_policy_margins(
            build_stage1_environment(configuration), model, model,
            condition_seeds=(1, 1), noise_seeds=(0,),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate condition seeds were accepted.")


def test_robust_override_audit_reports_zero_for_identical_policy():
    configuration = Stage1Configuration(
        training_episodes=1, episode_epochs=4, decision_interval_epochs=2,
        update_epochs=1,
    )
    environment = build_stage1_environment(configuration)
    model = train_stage1_ppo(configuration).model
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        environment, condition_seeds=(10,), noise_seeds=(0, 1),
        decision_epochs=(0,), baseline_policy=AlwaysKeepPolicy(),
        gain_standard_deviation_penalty=1.0,
    )
    records = audit_policy_overrides_against_robust_targets(
        dataset, model, deepcopy(model),
    )
    assert len(records) == 1
    assert not records[0].changed_action
    assert np.isclose(records[0].robust_gain_over_reference, 0.0)
