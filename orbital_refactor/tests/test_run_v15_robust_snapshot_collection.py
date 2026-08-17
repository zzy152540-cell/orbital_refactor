from experiments.run_v15_robust_snapshot_collection import main
from experiments.topology_snapshot_counterfactual import (
    load_topology_snapshot_tensor_dataset,
)


def test_robust_snapshot_collection_cli_writes_restartable_shard(tmp_path):
    output = tmp_path / "robust.npz"
    result = main([
        "--condition-seeds", "30",
        "--noise-seeds", "0", "1",
        "--epochs", "0",
        "--episode-epochs", "3",
        "--output", str(output),
    ])
    assert result == output
    dataset = load_topology_snapshot_tensor_dataset(output)
    assert dataset.feature_version == "v15.1-noise-robust-snapshot-action-value"
    assert len(dataset.groups) == 1
    assert dataset.groups[0].seed == 30
