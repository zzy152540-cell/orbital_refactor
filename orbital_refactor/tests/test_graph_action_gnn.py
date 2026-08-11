from experiments.counterfactual_action_value import (
    build_counterfactual_action_value_dataset,
)
from experiments.graph_action_gnn import (
    GraphActionValueNetwork,
    graph_action_multitask_loss,
    overfit_single_graph_action_group,
    train_graph_action_network,
    torch_graph_action_group,
)
from experiments.graph_action_tensor_dataset import (
    build_graph_action_tensor_dataset,
    split_graph_action_tensor_dataset_by_seed,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)
import torch


def _group():
    study = run_short_horizon_counterfactual_study(
        seeds=(0,), decision_epochs=(1,), horizon_epochs=(1,),
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
    )
    dataset = build_counterfactual_action_value_dataset(study)
    return build_graph_action_tensor_dataset(dataset).groups[0]


def test_minimal_gnn_outputs_one_utility_and_risk_per_action():
    group = torch_graph_action_group(_group())
    model = GraphActionValueNetwork(
        node_feature_count=group.node_features.shape[1],
        candidate_edge_feature_count=group.candidate_edge_features.shape[1],
        measurement_feature_count=group.measurement_features.shape[1],
        action_feature_count=group.action_features.shape[1],
    )

    prediction = model(group)
    loss = graph_action_multitask_loss(prediction, group.targets)

    assert prediction.utility.shape == (6,)
    assert prediction.risk_logit.shape == (6,)
    assert torch.isfinite(loss.total)


def test_minimal_gnn_can_overfit_one_causal_decision_group():
    result = overfit_single_graph_action_group(
        _group(), steps=500, learning_rate=3e-3, random_seed=0,
    )

    assert result.final_loss < 0.2 * result.initial_loss
    assert result.predicted_best_action == result.target_best_action
    assert result.final_utility_correlation is not None
    assert result.final_utility_correlation > 0.95


def test_multigroup_training_uses_seed_disjoint_validation_and_restores_best():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1, 2), decision_epochs=(1, 2), horizon_epochs=(1,),
        relative_modalities=("RANGE",),
    )
    dataset = build_graph_action_tensor_dataset(
        build_counterfactual_action_value_dataset(study)
    )
    split = split_graph_action_tensor_dataset_by_seed(
        dataset, training_seeds=(0, 1), validation_seeds=(2,),
    )
    result = train_graph_action_network(
        split.training, split.validation,
        epochs=80, patience=20, learning_rate=2e-3,
        hidden_size=24, random_seed=0,
    )

    assert 0 <= result.best_epoch <= result.epochs_run <= 80
    assert result.best_validation.mean_loss <= result.initial_validation.mean_loss

    with torch.no_grad():
        overlapping = split_graph_action_tensor_dataset_by_seed(
            dataset, training_seeds=(0,), validation_seeds=(1,),
        )
        try:
            train_graph_action_network(
                overlapping.training, overlapping.training, epochs=1,
            )
        except ValueError as error:
            assert "disjoint" in str(error)
        else:
            raise AssertionError("Overlapping seeds must be rejected.")
