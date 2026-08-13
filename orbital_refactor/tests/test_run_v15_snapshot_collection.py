from experiments.run_v15_snapshot_collection import main
from experiments.topology_snapshot_counterfactual import (
    load_topology_snapshot_tensor_dataset,
)


def test_snapshot_collection_cli_writes_restartable_shard(tmp_path):
    output = tmp_path / "walker_shard.npz"
    path = main([
        "--seeds", "0", "--epochs", "0", "--lookahead", "1",
        "--episode-epochs", "2", "--output", str(output),
    ])
    dataset = load_topology_snapshot_tensor_dataset(path)
    assert len(dataset.groups) == 1
    assert dataset.groups[0].seed == 0
    assert dataset.groups[0].decision_epoch == 0
    assert len(dataset.groups[0].policy_tensor.node_ids) == 20
