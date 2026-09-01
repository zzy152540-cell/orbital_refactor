from experiments.angle_tracker_evaluation import evaluate_frozen_angle_trackers


def test_evaluation_reports_each_method_and_seed():
    result = evaluate_frozen_angle_trackers(
        seeds=(0,), duration=20.0, dt=2.0, outage_window=(8.0, 12.0),
    )
    assert result["evaluation_seeds"] == [0]
    assert set(result["summary"]) == {
        "bias_adaptive_cann", "circular_kalman", "gated_cann",
        "coupled_ring_line_cann", "gated_complementary", "gated_pll",
        "measurement_hold", "ordinary_integration",
    }
    assert all(metrics["seed_count"] == 1 for metrics in result["summary"].values())
