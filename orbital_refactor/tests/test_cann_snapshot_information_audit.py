import numpy as np

from experiments.cann_snapshot_information_audit import (
    audit_cann_snapshot_information,
    mask_cann_snapshot_features,
    shuffle_cann_snapshot_features,
    strip_cann_snapshot_features,
)
from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_snapshot_counterfactual import (
    build_topology_snapshot_tensor_dataset,
)
from experiments.snapshot_cost_aware_targets import (
    apply_stage1_cost_aware_utility,
    summarize_cost_aware_oracles,
)


def test_cann_snapshot_dataset_keeps_its_schema_and_can_be_audited():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=4, dt=0.2,
        relative_modalities=("RANGE",), cann_policy_features=True,
    )
    dataset = build_topology_snapshot_tensor_dataset(
        environment, seeds=(1,), decision_epochs=(0, 1, 2),
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    report = audit_cann_snapshot_information(dataset)
    assert dataset.feature_version == (
        "v15.1-cann-online-snapshot-action-value"
    )
    assert report.group_count == 3
    assert report.node_count_range == (3, 3)
    assert 0 <= report.varying_feature_count <= len(report.features)
    base = strip_cann_snapshot_features(dataset)
    assert base.groups[0].targets is dataset.groups[0].targets
    assert base.groups[0].policy_tensor.schema_version == (
        "v15.0-policy-normalized"
    )
    assert base.groups[0].policy_tensor.node_features.shape[1] < (
        dataset.groups[0].policy_tensor.node_features.shape[1]
    )
    zero = mask_cann_snapshot_features(dataset)
    base_width = base.groups[0].policy_tensor.node_features.shape[1]
    assert zero.groups[0].policy_tensor.node_features.shape == (
        dataset.groups[0].policy_tensor.node_features.shape
    )
    assert np.count_nonzero(
        zero.groups[0].policy_tensor.node_features[:, base_width:]
    ) == 0
    shuffled = shuffle_cann_snapshot_features(dataset, random_seed=7)
    assert shuffled.groups[0].targets is dataset.groups[0].targets
    for original_group, shuffled_group in zip(dataset.groups, shuffled.groups):
        assert not np.array_equal(
            original_group.policy_tensor.node_features[:, base_width:],
            shuffled_group.policy_tensor.node_features[:, base_width:],
        )
    original = np.concatenate([
        group.policy_tensor.node_features[:, base_width:].reshape(-1)
        for group in dataset.groups
    ])
    randomized = np.concatenate([
        group.policy_tensor.node_features[:, base_width:].reshape(-1)
        for group in shuffled.groups
    ])
    np.testing.assert_allclose(np.sort(original), np.sort(randomized))
    cost_aware = apply_stage1_cost_aware_utility(dataset)
    keep_index = cost_aware.groups[0].action_kinds.index("keep")
    assert cost_aware.groups[0].targets[keep_index, 0] == 0.0
    summary = summarize_cost_aware_oracles(cost_aware)
    assert summary.keep_oracle_rate + summary.non_keep_oracle_rate == 1.0
    assert sum(rate for _, rate in summary.action_kind_rates) == 1.0
