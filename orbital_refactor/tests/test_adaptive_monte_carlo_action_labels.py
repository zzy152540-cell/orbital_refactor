from experiments.adaptive_monte_carlo_action_labels import (
    build_adaptive_monte_carlo_graph_action_dataset,
)
from experiments.monte_carlo_action_opportunity import RobustOpportunityCriteria
from experiments.monte_carlo_graph_action_dataset import (
    build_monte_carlo_graph_action_dataset,
)


def test_adaptive_sampling_extends_only_borderline_legal_actions():
    report = build_adaptive_monte_carlo_graph_action_dataset(
        scenario_id="adaptive",
        prefix_seeds=(0,),
        initial_future_noise_seeds=(100, 101),
        extended_future_noise_seeds=(100, 101, 102),
        criteria=RobustOpportunityCriteria(
            minimum_safe_positive_probability=0.0,
            minimum_consistency_non_degrading_probability=0.0,
        ),
        probability_margin=0.0,
        gain_margin=1e6,
        node_count=3,
        decision_epochs=(1,),
        horizon_epochs=(1,),
        relative_modalities=("RANGE",),
    )

    group = report.dataset.groups[0]
    assert report.total_action_count == 6
    assert report.extended_action_count == 3
    assert report.dataset.future_noise_seeds == (100, 101, 102)
    assert next(action for action in group.actions
                if action.action_kind == "keep").future_noise_seeds == (
                    100, 101, 102,
                )
    assert all(
        action.future_noise_seeds == (100, 101, 102)
        for action in group.actions if action.action_kind in {"add", "swap"}
    )
    assert all(
        action.future_noise_seeds == (100, 101)
        for action in group.actions if action.action_kind == "remove"
    )
    full = build_monte_carlo_graph_action_dataset(
        scenario_id="adaptive",
        prefix_seeds=(0,),
        future_noise_seeds=(100, 101, 102),
        node_count=3,
        decision_epochs=(1,),
        horizon_epochs=(1,),
        relative_modalities=("RANGE",),
    ).groups[0]
    full_by_edges = {action.active_edges: action for action in full.actions}
    assert all(
        action == full_by_edges[action.active_edges]
        for action in group.actions
        if len(action.future_noise_seeds) == 3
    )
