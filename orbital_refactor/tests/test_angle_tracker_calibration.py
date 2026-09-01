from experiments.angle_tracker_calibration import calibrate_angle_trackers


def test_short_calibration_keeps_seed_partition_explicit():
    result = calibrate_angle_trackers(
        seeds=(10,), duration=20.0, dt=2.0, outage_window=(8.0, 12.0),
    )
    assert result["calibration_seeds"] == [10]
    assert result["best_pll"]["method"] == "pll"
    assert result["best_circular_kalman"]["method"] == "circular_kalman"
