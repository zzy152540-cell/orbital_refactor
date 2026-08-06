import csv
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.v14_comparison import (
    build_v14_comparison_case,
    export_v14_comparison,
    run_v14_comparison,
)


def test_v14_comparison_runs_four_architectures_on_one_case():
    case = build_v14_comparison_case(duration=2.0, dt=2.0, random_seed=17)
    result = run_v14_comparison(case, fleet_ci_grid_points=3)

    assert set(result.metrics_by_algorithm) == {
        "independent",
        "v14_distributed",
        "centralized_fleet_ekf",
        "distributed_fleet_state_ci",
    }
    for algorithm, metrics in result.metrics_by_algorithm.items():
        assert set(metrics.by_node) == set(case.scenario.node_ids)
        assert np.isfinite(metrics.fleet_position_rmse)
        assert np.isfinite(metrics.fleet_velocity_rmse)
        assert np.isfinite(metrics.mean_nees)
        for node_metrics in metrics.by_node.values():
            assert np.isfinite(node_metrics.position_rmse)
            assert np.isfinite(node_metrics.velocity_rmse)
            assert np.isfinite(node_metrics.mean_nees)
        if algorithm == "independent":
            assert metrics.mean_nis is None
        else:
            assert metrics.mean_nis is not None
            assert np.isfinite(metrics.mean_nis)

    expected_attempts = (
        len(case.scenario.timestamps)
        * sum(
            len(case.topology.neighbors(node_id))
            for node_id in case.topology.node_ids
        )
    )
    assert (
        result.v14_history.communication_stats.attempted_state_count
        == expected_attempts
    )


def test_v14_comparison_is_reproducible():
    left = run_v14_comparison(
        build_v14_comparison_case(duration=2.0, dt=2.0, random_seed=23),
        fleet_ci_grid_points=3,
    )
    right = run_v14_comparison(
        build_v14_comparison_case(duration=2.0, dt=2.0, random_seed=23),
        fleet_ci_grid_points=3,
    )

    for algorithm in left.metrics_by_algorithm:
        left_metrics = left.metrics_by_algorithm[algorithm]
        right_metrics = right.metrics_by_algorithm[algorithm]
        assert left_metrics.fleet_position_rmse == right_metrics.fleet_position_rmse
        assert left_metrics.fleet_velocity_rmse == right_metrics.fleet_velocity_rmse
        assert left_metrics.mean_nees == right_metrics.mean_nees
        assert left_metrics.mean_nis == right_metrics.mean_nis


def test_v14_case_supports_three_observation_strategies():
    single = build_v14_comparison_case(
        duration=0.0,
        observation_strategy="single_endpoint",
    )
    shared = build_v14_comparison_case(
        duration=0.0,
        observation_strategy="shared",
    )
    reciprocal = build_v14_comparison_case(
        duration=0.0,
        observation_strategy="independent_reciprocal",
    )

    assert single.observation_usage == "observer_only"
    assert shared.observation_usage == "both_endpoints"
    assert reciprocal.observation_usage == "observer_only"
    assert len(single.observation_messages) == len(shared.observation_messages)
    assert len(reciprocal.observation_messages) == 2 * len(shared.observation_messages)


def test_v14_comparison_exports_json_and_per_node_csv():
    result = run_v14_comparison(
        build_v14_comparison_case(duration=0.0, dt=2.0, random_seed=5),
        fleet_ci_grid_points=3,
    )
    output = Path(".pytest_v14_export")
    if output.exists():
        shutil.rmtree(output)
    try:
        paths = export_v14_comparison(result, output)
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert set(payload) == set(result.metrics_by_algorithm)
        with paths["csv"].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 * 3
        assert {row["algorithm"] for row in rows} == set(
            result.metrics_by_algorithm
        )
    finally:
        if output.exists():
            shutil.rmtree(output)
