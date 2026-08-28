from experiments.deterministic_topology_visualization import (
    load_deterministic_topology_comparison,
)


def test_deterministic_topology_comparison_is_paired(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text(
        "seed,policy,final_position_rmse,cumulative_penalized_return,"
        "transmitted_messages,dropped_messages,replay_count,"
        "resynchronization_count,topology_switch_count\n"
        "0,keep,2.0,1.0,10,1,11,0,0\n"
        "0,information_greedy,1.5,0.8,12,2,13,1,1\n",
        encoding="utf-8",
    )
    result = load_deterministic_topology_comparison(path)
    record = result["records"][0]
    assert record["rmse_improvement"] == 0.5
    assert record["rmse_improvement_percent"] == 25.0
    assert record["topology_switch_count"] == 1.0


def test_deterministic_topology_comparison_rejects_unpaired_seed(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text(
        "seed,policy,final_position_rmse,cumulative_penalized_return,"
        "transmitted_messages,dropped_messages,replay_count,"
        "resynchronization_count,topology_switch_count\n"
        "0,keep,2.0,1.0,10,1,11,0,0\n",
        encoding="utf-8",
    )
    try:
        load_deterministic_topology_comparison(path)
    except ValueError as error:
        assert "keep and information_greedy" in str(error)
    else:
        raise AssertionError("An unpaired deterministic comparison was accepted.")
