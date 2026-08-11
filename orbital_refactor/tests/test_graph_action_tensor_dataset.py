import numpy as np
import pytest

from experiments.counterfactual_action_value import (
    build_counterfactual_action_value_dataset,
)
from experiments.graph_action_tensor_dataset import (
    build_graph_action_tensor_dataset,
    split_graph_action_tensor_dataset_by_seed,
    split_graph_action_tensor_dataset_three_way,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def _tensor_dataset(modalities=("RANGE",)):
    study = run_short_horizon_counterfactual_study(
        seeds=(0,), decision_epochs=(1,), horizon_epochs=(1,),
        relative_modalities=modalities,
    )
    return build_graph_action_tensor_dataset(
        build_counterfactual_action_value_dataset(study)
    )


def test_graph_action_tensor_group_aligns_actions_edges_and_targets():
    dataset = _tensor_dataset()
    group = dataset.groups[0]

    assert dataset.feature_version == "v14.3-graph-action-causal"
    assert group.node_features.shape == (3, 12)
    assert group.candidate_edge_index.shape == (2, 3)
    assert group.candidate_edge_features.shape[0] == 3
    assert group.action_features.shape == (6, 7)
    assert group.active_edge_mask.shape == (6, 3)
    assert group.targets.shape == (6, 7)
    keep = group.action_kinds.index("keep")
    assert np.sum(group.active_edge_mask[keep]) == 2.0
    assert np.sum(group.added_edge_mask[keep]) == 0.0
    assert np.allclose(group.targets[keep], 0.0)
    with pytest.raises(ValueError):
        group.node_features[0, 0] = 0.0


def test_physical_tensor_encodes_directed_body_measurements_without_future_labels():
    dataset = _tensor_dataset(("RADAR", "INFRARED", "OPTICAL"))
    group = dataset.groups[0]
    names = dataset.measurement_feature_names
    body = names.index("frame_BODY")
    quaternion = names.index("quaternion_available")
    optical = names.index("modality_OPTICAL")

    assert group.measurement_edge_index.shape[0] == 2
    assert group.measurement_features.shape[1] == len(names)
    optical_rows = group.measurement_features[:, optical] == 1.0
    assert np.any(optical_rows)
    assert np.all(group.measurement_features[optical_rows, body] == 1.0)
    assert np.all(group.measurement_features[optical_rows, quaternion] == 1.0)
    assert not any("rmse" in name or "nees" in name
                   for name in dataset.node_feature_names)


def test_action_masks_reconstruct_active_topology_from_keep_baseline():
    group = _tensor_dataset().groups[0]
    keep = group.action_kinds.index("keep")
    baseline = group.active_edge_mask[keep]

    assert np.allclose(
        group.active_edge_mask,
        baseline[None, :] + group.added_edge_mask - group.removed_edge_mask,
    )


def test_tensor_seed_split_is_disjoint_and_preserves_shared_schema():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,)
    )
    dataset = build_graph_action_tensor_dataset(
        build_counterfactual_action_value_dataset(study)
    )
    split = split_graph_action_tensor_dataset_by_seed(
        dataset, training_seeds=(0,), validation_seeds=(1,),
    )

    assert {group.group_id[1] for group in split.training.groups} == {0}
    assert {group.group_id[1] for group in split.validation.groups} == {1}
    assert split.training.node_feature_names == split.validation.node_feature_names
    with pytest.raises(ValueError, match="disjoint"):
        split_graph_action_tensor_dataset_by_seed(
            dataset, training_seeds=(0,), validation_seeds=(0,)
        )


def test_tensor_three_way_split_is_strictly_seed_disjoint():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1, 2, 3), decision_epochs=(1,), horizon_epochs=(1,),
    )
    dataset = build_graph_action_tensor_dataset(
        build_counterfactual_action_value_dataset(study)
    )
    split = split_graph_action_tensor_dataset_three_way(
        dataset,
        training_seeds=(0, 1),
        validation_seeds=(2,),
        test_seeds=(3,),
    )

    assert {group.group_id[1] for group in split.training.groups} == {0, 1}
    assert {group.group_id[1] for group in split.validation.groups} == {2}
    assert {group.group_id[1] for group in split.test.groups} == {3}
    with pytest.raises(ValueError, match="disjoint"):
        split_graph_action_tensor_dataset_three_way(
            dataset,
            training_seeds=(0, 1),
            validation_seeds=(1, 2),
            test_seeds=(3,),
        )
