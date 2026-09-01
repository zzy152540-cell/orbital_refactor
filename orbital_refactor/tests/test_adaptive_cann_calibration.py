from experiments.adaptive_cann_calibration import calibrate_adaptive_cann


def test_short_adaptive_cann_calibration_selects_tested_gain():
    result = calibrate_adaptive_cann(
        seeds=(10,), gains=(0.0, 0.1), duration=12.0, dt=2.0,
        outage_window=(4.0, 8.0),
    )
    assert result["calibration_seeds"] == [10]
    assert result["best"]["gain"] in (0.0, 0.1)
    assert len(result["rows"]) == 2
