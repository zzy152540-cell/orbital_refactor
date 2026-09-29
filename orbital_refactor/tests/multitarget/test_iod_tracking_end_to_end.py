from experiments.run_iod_tracking_end_to_end import run_iod_tracking_end_to_end


def test_raw_iod_hands_two_targets_to_continuous_tracking(tmp_path):
    result = run_iod_tracking_end_to_end(
        tmp_path, random_seed=9, image_size=17,
    )

    assert result["initializedTargetIds"] == ["target-1", "target-2"]
    assert result["localFilterTaskCount"] == 6
    for values in result["resultsByTarget"].values():
        assert values["lifecycle"] == ["TRACKING", "TRACKING", "TRACKING"]
        assert values["positionRmseM"] < 100.0
        assert values["velocityRmseMps"] < 1.0
