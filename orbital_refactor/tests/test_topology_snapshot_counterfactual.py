import numpy as np
import torch

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_snapshot_counterfactual import (
    build_topology_snapshot_tensor_dataset,
    build_online_snapshot_action_tensor,
    build_noise_robust_topology_snapshot_tensor_dataset,
    augment_noise_robust_snapshot_inputs,
    build_topology_action_snapshot_tensor,
    evaluate_topology_action_snapshot,
    export_snapshot_action_values,
    load_topology_snapshot_tensor_dataset,
    merge_topology_snapshot_tensor_datasets,
    save_topology_snapshot_tensor_dataset,
    split_topology_snapshot_dataset_by_seed,
)
from experiments.graph_action_gnn import (
    overfit_single_snapshot_action_group,
    torch_snapshot_action_group,
    train_snapshot_action_network,
    train_snapshot_moment_network,
    build_snapshot_focus_action_weights,
    save_snapshot_action_checkpoint,
    load_snapshot_action_value_checkpoint,
    save_snapshot_action_value_checkpoint,
)


def test_snapshot_counterfactual_labels_all_legal_actions_without_mutation(tmp_path):
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
    )
    records = evaluate_topology_action_snapshot(
        environment, seed=0, decision_epoch=1,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=2,
    )
    assert records[0].action_kind == "keep"
    assert records[0].position_rmse_reduction_vs_keep == 0.0
    assert len(records) == 4
    assert all(record.lookahead_steps == 2 for record in records)
    assert any(record.action_kind == "swap" for record in records)
    path = export_snapshot_action_values(records, tmp_path / "values.csv")
    assert path.exists()
    assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 5


def test_snapshot_counterfactual_can_hold_scenario_conditions_fixed():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    left = evaluate_topology_action_snapshot(
        environment, seed=1, condition_seed=70, decision_epoch=0,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    left_conditions = environment._episode_conditions
    right = evaluate_topology_action_snapshot(
        environment, seed=2, condition_seed=70, decision_epoch=0,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    assert environment._episode_conditions == left_conditions
    assert any(
        a.final_position_rmse != b.final_position_rmse
        for a, b in zip(left, right)
    )


def test_noise_robust_dataset_averages_targets_and_splits_by_condition():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        environment, condition_seeds=(30, 31), noise_seeds=(0, 1),
        decision_epochs=(0,), baseline_policy=AlwaysKeepPolicy(),
    )
    assert dataset.feature_version == (
        "v15.4-noise-robust-moments-snapshot-action-value"
    )
    assert tuple(group.seed for group in dataset.groups) == (30, 31)
    left = tuple(build_topology_action_snapshot_tensor(
        environment, seed=noise_seed, condition_seed=30, decision_epoch=0,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )[0].targets for noise_seed in (0, 1))
    expected = (left[0] + left[1]) / 2.0
    assert np.allclose(dataset.groups[0].targets[:, :5], expected)
    np.testing.assert_allclose(dataset.groups[0].targets[:, 5], expected[:, 0])
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(30,), validation_seeds=(31,),
    )
    assert len(split.training.groups) == len(split.validation.groups) == 1


def test_noise_robust_dataset_can_use_lower_confidence_gain_targets():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        environment, condition_seeds=(30,), noise_seeds=(0, 1),
        decision_epochs=(0,), baseline_policy=AlwaysKeepPolicy(),
        gain_standard_deviation_penalty=1.0,
    )
    samples = np.stack(tuple(build_topology_action_snapshot_tensor(
        environment, seed=noise_seed, condition_seed=30, decision_epoch=0,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )[0].targets for noise_seed in (0, 1)))
    expected_gain = samples[:, :, 0].mean(0) - samples[:, :, 0].std(0)
    assert dataset.feature_version == (
        "v15.4-noise-robust-lcb-moments-snapshot-action-value"
    )
    np.testing.assert_allclose(dataset.groups[0].targets[:, 0], expected_gain)
    np.testing.assert_allclose(
        dataset.groups[0].targets[:, 5], samples[:, :, 0].mean(0)
    )
    np.testing.assert_allclose(
        dataset.groups[0].targets[:, 6], samples[:, :, 0].std(0)
    )


def test_noise_robust_dataset_can_augment_inputs_without_condition_leakage():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        environment, condition_seeds=(30, 31), noise_seeds=(0, 1),
        decision_epochs=(0,), baseline_policy=AlwaysKeepPolicy(),
        gain_standard_deviation_penalty=1.0,
        include_all_noise_observations=True,
    )
    assert dataset.feature_version == (
        "v15.4-noise-augmented-lcb-moments-snapshot-action-value"
    )
    assert tuple(group.seed for group in dataset.groups) == (30, 30, 31, 31)
    assert tuple(group.observation_seed for group in dataset.groups) == (0, 1, 0, 1)
    np.testing.assert_allclose(
        dataset.groups[0].targets, dataset.groups[1].targets
    )
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(30,), validation_seeds=(31,),
    )
    assert len(split.training.groups) == len(split.validation.groups) == 2


def test_existing_robust_targets_can_be_noise_augmented_without_relabeling():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        randomize_stage1_conditions=True,
    )
    robust = build_noise_robust_topology_snapshot_tensor_dataset(
        environment, condition_seeds=(30, 31), noise_seeds=(0, 1),
        decision_epochs=(0,), baseline_policy=AlwaysKeepPolicy(),
        gain_standard_deviation_penalty=1.0,
    )
    augmented = augment_noise_robust_snapshot_inputs(
        environment, robust, noise_seeds=(0, 1),
        baseline_policy=AlwaysKeepPolicy(),
    )
    assert len(augmented.groups) == 4
    for index, reference in enumerate(robust.groups):
        pair = augmented.groups[2 * index:2 * index + 2]
        assert {group.observation_seed for group in pair} == {0, 1}
        assert all(group.seed == reference.seed for group in pair)
        assert all(np.array_equal(group.targets, reference.targets)
                   for group in pair)


def test_snapshot_counterfactual_rejects_epoch_beyond_horizon():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=1, relative_modalities=("RANGE",),
    )
    try:
        evaluate_topology_action_snapshot(
            environment, seed=0, decision_epoch=2,
            baseline_policy=AlwaysKeepPolicy(),
        )
    except ValueError as error:
        assert "beyond" in str(error)
    else:
        raise AssertionError("Out-of-horizon snapshot must be rejected.")


def test_snapshot_tensor_aligns_no_truth_graph_actions_and_targets():
    group, records = build_topology_action_snapshot_tensor(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        ),
        seed=0, decision_epoch=1, baseline_policy=AlwaysKeepPolicy(),
        lookahead_steps=2,
    )
    assert group.policy_tensor.schema_version == "v15.0-policy-normalized"
    assert group.action_features.shape == (len(records), 7)
    assert group.active_edge_mask.shape == (
        len(records), len(group.policy_tensor.candidate_edges)
    )
    assert group.targets.shape == (len(records), len(group.target_names))
    assert group.target_names[0] == "position_rmse_reduction_vs_keep"
    assert group.targets[group.action_kinds.index("keep"), 0] == 0.0
    assert not group.targets.flags.writeable
    assert not any("truth" in name for name in group.policy_tensor.node_feature_names)

    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
    )
    online, action_ids = build_online_snapshot_action_tensor(
        environment.reset(seed=0)
    )
    assert len(action_ids) == len(online.action_kinds)
    assert (online.targets == 0.0).all()
    assert not online.targets.flags.writeable


def test_snapshot_tensor_adapts_to_gnn_and_overfits_one_group():
    group, _ = build_topology_action_snapshot_tensor(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        ),
        seed=0, decision_epoch=1, baseline_policy=AlwaysKeepPolicy(),
        lookahead_steps=2,
    )
    tensors = torch_snapshot_action_group(group)
    edge_count = len(group.policy_tensor.candidate_edges)
    assert tensors.measurement_edge_index.shape == (2, 2 * edge_count)
    assert tensors.measurement_features.shape[0] == 2 * edge_count
    global_count = len(group.policy_tensor.global_feature_names)
    assert tensors.action_features.shape[1] == 7 + global_count
    assert tensors.action_features[0, -global_count:].abs().sum() > 0.0

    result = overfit_single_snapshot_action_group(
        group, steps=500, learning_rate=3e-3, random_seed=0,
    )
    assert result.final_loss < 0.25 * result.initial_loss
    assert result.predicted_best_action == result.target_best_action
    assert result.final_utility_correlation is not None
    assert result.final_utility_correlation > 0.95


def test_snapshot_dataset_training_keeps_validation_seed_disjoint():
    dataset = build_topology_snapshot_tensor_dataset(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=3, relative_modalities=("RANGE",),
        ),
        seeds=(0, 1, 2), decision_epochs=(0, 1),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(0, 1), validation_seeds=(2,),
    )
    assert len(split.training.groups) == 4
    assert len(split.validation.groups) == 2
    result = train_snapshot_action_network(
        split.training, split.validation, epochs=40, patience=10,
        hidden_size=16, learning_rate=2e-3, random_seed=0,
    )
    assert 0 <= result.best_epoch <= result.epochs_run <= 40
    assert result.best_validation.mean_loss <= result.initial_validation.mean_loss
    assert result.best_validation.mean_oracle_regret >= 0.0


def test_snapshot_moment_training_uses_separate_mean_and_deviation_targets():
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        ),
        condition_seeds=(0, 1), noise_seeds=(0, 1), decision_epochs=(0,),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
        gain_standard_deviation_penalty=1.0,
    )
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(0,), validation_seeds=(1,),
    )
    result = train_snapshot_moment_network(
        split.training, split.validation, epochs=3, patience=2,
        hidden_size=8, learning_rate=2e-3, random_seed=0,
    )

    assert result.target_scale > 0.0
    assert 0 <= result.best_epoch <= result.epochs_run <= 3
    assert result.best_validation.action_count > 0
    assert np.isfinite(result.best_validation.mean_gain_rmse)
    assert np.isfinite(result.best_validation.standard_deviation_rmse)
    assert 0.0 <= result.best_validation.negative_gain_recall <= 1.0
    weights = build_snapshot_focus_action_weights(
        result.model, split.training, focused_action_weight=4.0,
        hierarchical=False,
    )
    assert len(weights) == len(split.training.groups)
    assert all(np.count_nonzero(group_weights == 4.0) == 1
               for group_weights in weights)
    weighted = train_snapshot_moment_network(
        split.training, split.validation, epochs=1, patience=1,
        hidden_size=8, random_seed=0,
        mean_sign_weight=0.1,
        training_action_weights=weights,
        validation_action_weights=build_snapshot_focus_action_weights(
            result.model, split.validation, focused_action_weight=4.0,
            hierarchical=False,
        ),
    )
    assert np.isfinite(weighted.best_validation.mean_loss)


def test_hierarchical_snapshot_checkpoint_is_warm_start_compatible(tmp_path):
    dataset = build_topology_snapshot_tensor_dataset(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        ), seeds=(0, 1), decision_epochs=(0,),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(0,), validation_seeds=(1,),
    )
    result = train_snapshot_action_network(
        split.training, split.validation, epochs=1, patience=1,
        hidden_size=16, loss_mode="hierarchical",
    )
    path = save_snapshot_action_checkpoint(
        result, split.training, tmp_path / "initializer.pt",
        configuration={
            "hidden_size": 16, "message_passing_steps": 2,
            "explicit_action_pairing": True, "loss_mode": "hierarchical",
        },
    )
    assert path.exists()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["configuration"]["loss_mode"] == "hierarchical"
    assert checkpoint["node_feature_count"] == len(dataset.node_feature_names)


def test_auxiliary_action_value_checkpoint_round_trips_exactly(tmp_path):
    dataset = build_topology_snapshot_tensor_dataset(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        ), seeds=(0, 1), decision_epochs=(0,),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    split = split_topology_snapshot_dataset_by_seed(
        dataset, training_seeds=(0,), validation_seeds=(1,),
    )
    result = train_snapshot_action_network(
        split.training, split.validation, epochs=1, patience=1,
        hidden_size=16, loss_mode="decision",
    )
    path = save_snapshot_action_value_checkpoint(
        result, split.training, tmp_path / "value.pt",
        configuration={
            "hidden_size": 16, "message_passing_steps": 2,
            "explicit_action_pairing": True, "loss_mode": "decision",
        },
    )
    loaded = load_snapshot_action_value_checkpoint(path)
    group = torch_snapshot_action_group(split.validation.groups[0])
    with torch.no_grad():
        expected = result.model(group).utility
        actual = loaded(group).utility
    torch.testing.assert_close(actual, expected)




def test_snapshot_dataset_round_trip_is_pickle_free_and_exact(tmp_path):
    dataset = build_topology_snapshot_tensor_dataset(
        TopologyControlEnvironment(
            node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
        ), seeds=(0, 1), decision_epochs=(0,),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    path = save_topology_snapshot_tensor_dataset(dataset, tmp_path / "snapshot.npz")
    restored = load_topology_snapshot_tensor_dataset(path)

    assert restored.feature_version == dataset.feature_version
    assert restored.action_feature_names == dataset.action_feature_names
    assert len(restored.groups) == len(dataset.groups)
    for expected, actual in zip(dataset.groups, restored.groups):
        assert (actual.seed, actual.decision_epoch) == (
            expected.seed, expected.decision_epoch
        )
        assert actual.policy_tensor.node_ids == expected.policy_tensor.node_ids
        assert actual.policy_tensor.candidate_edges == (
            expected.policy_tensor.candidate_edges
        )
        assert (actual.policy_tensor.node_features ==
                expected.policy_tensor.node_features).all()
        assert (actual.policy_tensor.edge_features ==
                expected.policy_tensor.edge_features).all()
        assert (actual.targets == expected.targets).all()
        assert not actual.targets.flags.writeable


def test_snapshot_dataset_shards_merge_in_seed_epoch_order():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
    )
    shards = tuple(build_topology_snapshot_tensor_dataset(
        environment, seeds=(seed,), decision_epochs=(0,),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    ) for seed in (1, 0))
    merged = merge_topology_snapshot_tensor_datasets(shards)
    assert [(group.seed, group.decision_epoch) for group in merged.groups] == [
        (0, 0), (1, 0)
    ]
    try:
        merge_topology_snapshot_tensor_datasets((shards[0], shards[0]))
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Duplicate snapshot groups must be rejected.")
