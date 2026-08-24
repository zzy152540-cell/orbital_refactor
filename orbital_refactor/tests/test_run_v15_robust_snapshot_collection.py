import pytest

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
        "--heterogeneous-links",
        "--output", str(output),
    ])
    assert result == output
    dataset = load_topology_snapshot_tensor_dataset(output)
    assert dataset.feature_version == (
        "v15.4-noise-robust-moments-snapshot-action-value"
    )
    assert len(dataset.groups) == 1
    assert dataset.groups[0].seed == 30
    packet_loss_index = dataset.edge_feature_names.index("packet_loss_rate")
    assert len(set(dataset.groups[0].policy_tensor.edge_features[
        :, packet_loss_index
    ])) > 1


def test_robust_collection_rejects_two_condition_distributions(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "--condition-seeds", "1", "--noise-seeds", "0",
            "--heterogeneous-links", "--randomized-physical-scenarios",
            "--output", str(tmp_path / "invalid.npz"),
        ])
