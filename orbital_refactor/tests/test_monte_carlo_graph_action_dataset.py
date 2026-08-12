from dataclasses import replace

import pytest

from experiments.monte_carlo_graph_action_dataset import (
    build_monte_carlo_graph_action_dataset,
    combine_monte_carlo_graph_action_datasets,
    split_monte_carlo_dataset_by_scenario,
)


def test_monte_carlo_targets_share_prefix_and_aggregate_future_noise():
    dataset = build_monte_carlo_graph_action_dataset(
        scenario_id="baseline",
        prefix_seeds=(0,),
        future_noise_seeds=(100, 101, 102),
        decision_epochs=(1,),
        horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    group = dataset.groups[0]
    keep = next(action for action in group.actions
                if action.action_kind == "keep")

    assert dataset.prefix_seeds == (0,)
    assert dataset.future_noise_seeds == (100, 101, 102)
    assert len(group.actions) == 6
    assert keep.mean_position_rmse_reduction == 0.0
    assert keep.position_rmse_reduction_standard_deviation == 0.0
    assert keep.mean_relative_position_rmse_reduction == 0.0
    assert keep.severe_relative_loss_probability == 0.0
    assert keep.safe_positive_gain_probability == 0.0
    assert {
        node for node, _ in keep.mean_position_rmse_reduction_by_node
    } == {"sat_01", "sat_02", "sat_03"}
    assert keep.mean_covariance_trace_reduction == 0.0
    assert {
        node for node, _ in keep.mean_nees_by_node
    } == {"sat_01", "sat_02", "sat_03"}
    assert {
        (node, modality)
        for node, modality, *_ in keep.mean_nis_by_node_and_modality
    } == {
        (node, modality)
        for node in ("sat_01", "sat_02", "sat_03")
        for modality in ("AZ_EL", "RANGE", "RANGE_RATE")
    }
    assert all(
        sample_count > 0.0
        for *_, sample_count in keep.mean_nis_by_node_and_modality
    )
    assert all(
        0.0 <= action.safe_positive_gain_probability <= 1.0
        for action in group.actions
    )
    assert all(
        action.mean_relative_position_rmse_reduction_confidence_interval[0]
        <= action.mean_relative_position_rmse_reduction
        <= action.mean_relative_position_rmse_reduction_confidence_interval[1]
        for action in group.actions
    )


def test_monte_carlo_dataset_validates_seed_axes():
    with pytest.raises(ValueError, match="future_noise_seeds"):
        build_monte_carlo_graph_action_dataset(
            scenario_id="baseline",
            prefix_seeds=(0,),
            future_noise_seeds=(1, 1),
            decision_epochs=(1,),
            horizon_epochs=(1,),
        )
    with pytest.raises(ValueError, match="severe_relative_loss_threshold"):
        build_monte_carlo_graph_action_dataset(
            scenario_id="baseline",
            prefix_seeds=(0,),
            future_noise_seeds=(1, 2),
            decision_epochs=(1,),
            horizon_epochs=(1,),
            severe_relative_loss_threshold=-0.1,
        )


def test_monte_carlo_online_backend_aggregates_future_branches():
    dataset = build_monte_carlo_graph_action_dataset(
        scenario_id="online",
        prefix_seeds=(0,), future_noise_seeds=(100, 101),
        decision_epochs=(1,), horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        backend="online_orchestrator",
        future_batch_relative_observations=True,
        packet_loss_by_edge={("sat_01", "sat_03"): 0.2},
        communication_delay_by_edge={("sat_01", "sat_03"): 1.0},
    )

    assert len(dataset.groups) == 1
    assert all(
        action.future_noise_seeds == (100, 101)
        for action in dataset.groups[0].actions
    )


def test_monte_carlo_split_is_by_physical_scenario_not_noise_seed():
    baseline = build_monte_carlo_graph_action_dataset(
        scenario_id="baseline",
        prefix_seeds=(0,),
        future_noise_seeds=(100, 101),
        decision_epochs=(1,),
        horizon_epochs=(1,),
    )

    def renamed(name):
        return replace(
            baseline,
            scenario_ids=(name,),
            groups=tuple(
                replace(group, scenario_id=name) for group in baseline.groups
            ),
        )

    combined = combine_monte_carlo_graph_action_datasets(
        baseline, renamed("phase_b"), renamed("phase_c")
    )
    split = split_monte_carlo_dataset_by_scenario(
        combined,
        training_scenarios=("baseline",),
        validation_scenarios=("phase_b",),
        test_scenarios=("phase_c",),
    )

    assert split.training.scenario_ids == ("baseline",)
    assert split.validation.scenario_ids == ("phase_b",)
    assert split.test.scenario_ids == ("phase_c",)
    assert split.training.future_noise_seeds == (100, 101)
    with pytest.raises(ValueError, match="disjoint"):
        split_monte_carlo_dataset_by_scenario(
            combined,
            training_scenarios=("baseline",),
            validation_scenarios=("baseline",),
            test_scenarios=("phase_c",),
        )
