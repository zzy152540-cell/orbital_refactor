from experiments.online_counterfactual_stability_scan import (
    run_online_counterfactual_stability_scan,
)


def test_online_stability_scan_separates_stale_and_protocol_rejections():
    scan = run_online_counterfactual_stability_scan(
        seeds=(0,), horizon_epochs=2,
        packet_loss=0.2, communication_delay=2.0,
    )

    assert len(scan.records) == 1
    assert scan.positive_best_gain_rate == 1.0
    assert scan.total_stale_topology_message_count > 0
    assert scan.total_protocol_rejection_count == 0
    assert sum(count for _, _, count in scan.best_action_counts) == 1


def test_online_stability_scan_requires_unique_nonempty_seeds():
    for seeds in ((), (0, 0)):
        try:
            run_online_counterfactual_stability_scan(seeds=seeds)
        except ValueError as error:
            assert "nonempty and unique" in str(error)
        else:
            raise AssertionError("Invalid seeds should be rejected.")
