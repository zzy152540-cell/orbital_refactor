from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_snapshot_counterfactual import (
    build_topology_snapshot_tensor_dataset,
    build_online_snapshot_action_tensor,
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
