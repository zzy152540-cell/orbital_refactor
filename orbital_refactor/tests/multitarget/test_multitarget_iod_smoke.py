from experiments.run_multitarget_iod_smoke import run_multitarget_iod_smoke


def test_noisy_raw_frontend_initializes_two_targets_with_one_radar_outlier(tmp_path):
    result = run_multitarget_iod_smoke(
        tmp_path,
        duration=20.0,
        dt=10.0,
        random_seed=8,
        image_size=17,
        inject_radar_outlier=True,
    )

    assert result["initializedTargetIds"] == ["target-1", "target-2"]
    for value in result["resultsByTarget"].values():
        assert value["status"] == "SUCCESS"
        assert value["positionErrorM"] < 200.0
        assert value["velocityErrorMps"] < 2.0
    affected = result["resultsByTarget"]["target-1"]
    assert affected["diagnostics"]["rejectedObservationCount"] >= 1
    assert affected["qualityScore"] > 0.5
