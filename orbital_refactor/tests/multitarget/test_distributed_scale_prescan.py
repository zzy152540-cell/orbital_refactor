from experiments.multitarget_distributed_scale_prescan import (
    run_multitarget_distributed_scale_prescan,
)


def test_short_scale_prescan_covers_sparse_and_dense_topologies():
    summary = run_multitarget_distributed_scale_prescan(
        node_counts=(4,),
        target_count=2,
        epochs=3,
    )

    assert summary["kind"] == "MULTITARGET_DISTRIBUTED_SCALE_PRESCAN"
    records = {item["topology"]: item for item in summary["records"]}
    assert set(records) == {"chain", "fully_connected"}
    assert records["chain"]["maximumContributorsPerTarget"] == 3
    assert records["fully_connected"]["maximumContributorsPerTarget"] == 4
    assert records["fully_connected"]["receivedMessageCount"] > records[
        "chain"
    ]["receivedMessageCount"]
    assert all(item["pendingMessageCount"] == 0 for item in records.values())
