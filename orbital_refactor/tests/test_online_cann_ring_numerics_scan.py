from experiments.online_cann_ring_numerics_scan import (
    run_online_cann_ring_numerics_scan,
)


def test_ring_numerics_scan_preserves_filter_contract():
    results = run_online_cann_ring_numerics_scan(
        seed=3, node_count=3, episode_epochs=3, dt=0.2,
    )
    assert [item.label for item in results] == [
        "reference_180_1ms", "stable_180_2ms",
        "compact_90_2ms", "compact_60_2ms",
    ]
    assert all(item.base_filter_identical for item in results)
    assert all(item.base_tensor_prefix_identical for item in results)
    assert all(item.runtime_seconds > 0.0 for item in results)
    assert results[0].mean_cann_feature_difference == 0.0
    assert results[0].maximum_cann_feature_difference == 0.0
