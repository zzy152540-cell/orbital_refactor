from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_reward_calibration import (
    RewardCalibrationRecord,
    RewardCostWeights,
    load_reward_calibration_records,
    run_reward_calibration_scan,
    summarize_reward_calibration,
)
import csv


def _environment():
    return TopologyControlEnvironment(
        node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
    )


def test_reward_weight_validation():
    try:
        RewardCostWeights(topology_switch=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative cost weights must be rejected.")


def test_scan_is_resumable_and_summary_groups_cells(tmp_path):
    path = tmp_path / "calibration.csv"
    weights = (RewardCostWeights(), RewardCostWeights(topology_switch=0.1))
    first = run_reward_calibration_scan(
        path, seeds=(0,), weight_grid=weights,
        environment_factory=_environment, policies=("keep",),
    )
    second = run_reward_calibration_scan(
        path, seeds=(0,), weight_grid=weights,
        environment_factory=_environment, policies=("keep",),
    )
    assert len(first) == len(second) == 2
    assert load_reward_calibration_records(path) == second
    summaries = summarize_reward_calibration(second)
    assert len(summaries) == 2
    assert all(summary.sample_count == 1 for summary in summaries)


def test_scan_accepts_generators_without_losing_grid_cells(tmp_path):
    records = run_reward_calibration_scan(
        tmp_path / "generator_grid.csv", seeds=(seed for seed in (0, 1)),
        weight_grid=(RewardCostWeights(topology_switch=value)
                     for value in (0.0, 0.1)),
        environment_factory=_environment,
        policies=(name for name in ("keep",)),
    )
    assert len(records) == 4


def test_oracle_cell_records_constraint_penalty(tmp_path):
    records = run_reward_calibration_scan(
        tmp_path / "oracle.csv", seeds=(1,),
        weight_grid=(RewardCostWeights(topology_switch=0.25),),
        environment_factory=_environment,
        policies=("short_horizon_oracle",),
    )
    assert len(records) == 1
    record = records[0]
    assert record.cumulative_penalized_return <= record.cumulative_task_reward


def test_oracle_horizons_have_distinct_resume_keys(tmp_path):
    path = tmp_path / "horizons.csv"
    common = dict(
        output_path=path, seeds=(0,), weight_grid=(RewardCostWeights(),),
        environment_factory=_environment, policies=("short_horizon_oracle",),
    )
    run_reward_calibration_scan(**common, oracle_lookahead_steps=1)
    records = run_reward_calibration_scan(**common, oracle_lookahead_steps=2)
    assert len(records) == 2
    assert {record.oracle_lookahead_steps for record in records} == {1, 2}
    assert {summary.oracle_lookahead_steps for summary in
            summarize_reward_calibration(records)} == {1, 2}


def test_legacy_csv_is_readable_but_cannot_be_appended(tmp_path):
    path = tmp_path / "legacy.csv"
    fields = [
        name for name in RewardCalibrationRecord.__dataclass_fields__
        if name not in {"oracle_lookahead_steps", "configuration_id"}
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "seed": 0, "policy": "keep", "communication_weight": 0,
            "topology_switch_weight": 0, "resynchronization_weight": 0,
            "final_position_rmse": 1, "cumulative_task_reward": 0,
            "cumulative_penalized_return": 0, "transmitted_messages": 0,
            "dropped_messages": 0, "replay_count": 0,
            "resynchronization_count": 0, "topology_switch_count": 0,
        })
    assert load_reward_calibration_records(path)[0].oracle_lookahead_steps == 1
    try:
        run_reward_calibration_scan(
            path, seeds=(0,), weight_grid=(RewardCostWeights(),),
            environment_factory=_environment, policies=("keep",),
        )
    except ValueError as error:
        assert "incompatible schema" in str(error)
    else:
        raise AssertionError("Legacy CSV append must be rejected.")
