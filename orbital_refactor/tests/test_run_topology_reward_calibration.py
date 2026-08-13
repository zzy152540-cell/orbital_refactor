import pytest

from experiments.run_topology_reward_calibration import _csv_values, main
from experiments.topology_reward_calibration import (
    load_reward_calibration_records,
)


def test_csv_values_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one seeds"):
        _csv_values(" , ", int, "seeds")


def test_cli_runs_and_resumes_minimal_scan(tmp_path, capsys):
    output = tmp_path / "cli.csv"
    arguments = (
        "--output", str(output), "--nodes", "3", "--epochs", "1",
        "--seeds", "0", "--switch-weights", "0,0.1",
        "--resync-weights", "0", "--policies", "keep",
        "--oracle-lookahead", "2",
        "--packet-loss", "0", "--communication-delay", "0",
    )
    assert main(list(arguments)) == 0
    assert main(list(arguments)) == 0
    assert len(load_reward_calibration_records(output)) == 2
    assert {record.oracle_lookahead_steps for record in
            load_reward_calibration_records(output)} == {2}
    assert "records=2" in capsys.readouterr().out


def test_cli_rejects_unknown_policy(tmp_path):
    with pytest.raises(ValueError, match="Unknown policies"):
        main([
            "--output", str(tmp_path / "bad.csv"),
            "--policies", "not_a_policy",
        ])


def test_cli_accepts_visibility_range(tmp_path):
    assert main([
        "--output", str(tmp_path / "visible.csv"), "--nodes", "3",
        "--epochs", "1", "--seeds", "0", "--policies", "keep",
        "--switch-weights", "0", "--resync-weights", "0",
        "--maximum-measurement-range", "1",
    ]) == 0


def test_cli_validates_walker_node_count(tmp_path):
    with pytest.raises(ValueError, match="requires node_count=20"):
        main([
            "--output", str(tmp_path / "walker.csv"),
            "--scenario", "walker_20_5_3", "--nodes", "5",
            "--epochs", "1", "--seeds", "0", "--policies", "keep",
            "--switch-weights", "0", "--resync-weights", "0",
        ])
