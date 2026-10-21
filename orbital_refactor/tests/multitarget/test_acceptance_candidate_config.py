import json
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "multitarget_acceptance_candidate_v3.json"
)


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_acceptance_candidate_freezes_common_scenario_and_primary_metrics():
    config = _config()
    common = config["commonScenario"]
    metrics = config["metricConfigurations"]

    assert common["friendlyObserverCount"] == 15
    assert common["hostileTargetCount"] == 15
    assert common["durationSeconds"] == 120.0
    assert common["seeds"] == [0, 1, 2, 3, 4]
    assert metrics["P05"]["primaryMetric"] == "position_rmse_m"
    assert metrics["P05"]["minimumImprovementPercent"] == 8.0
    assert metrics["P07"]["primaryMetric"] == "position_mean_error_m"
    assert metrics["P07"]["minimumImprovementPercent"] == 6.0
    assert metrics["R06"]["maximumIncreasePercent"] == 15.0
    assert metrics["R08"]["primaryMetric"] == "position_mean_error_m"
    assert metrics["R08"]["maximumIncreasePercent"] == 12.0
    assert set(metrics) == {
        "F01", "F02", "F03", "F04", "P05",
        "R06", "P07", "R08", "P09", "F10",
    }
    assert metrics["P09"]["maximumMeanLatencyMs"] == 100.0
    assert metrics["P09"]["evidence"]["meanMs"] < 1.0
    assert metrics["F10"]["manualGuiChecklistRequired"] is True


def test_r06_temporary_acceptance_uses_only_the_passing_radial_configuration():
    config = _config()
    r06 = config["metricConfigurations"]["R06"]
    radial = r06["radarDropoutOverride"]
    boundaries = config["acceptanceBoundaries"]

    assert radial == {
        "enabledOnAllFriendlyObservers": True,
        "source": "INDEPENDENT_RADIAL_BACKUP_SHADOW",
        "rangeSigmaM": 1.0,
        "rangeRateSigmaMps": 0.0125,
        "physicalValidation": False,
        "physicalValidationWaivedForThisAcceptance": True,
    }
    assert boundaries["excludePartialRadialCoverage"] is True
    assert boundaries["excludeMultiHopCoverageExperiment"] is True
    assert boundaries["excludeCompleteRadialLineageGate"] is True
    assert boundaries["formalPhysicalSensorClaimed"] is False
    assert r06["evidence"]["perModalityStatus"] == {
        "OPTICAL": "PASS_L2",
        "INFRARED": "PASS_L2",
        "RADAR": "PASS_L2_WITH_NUMERICAL_BACKUP",
    }
    assert r06["modalitySpecificOverridePolicy"] == {
        "OPTICAL": "NONE",
        "INFRARED": "NONE",
        "RADAR": "USE_RADAR_DROPOUT_OVERRIDE",
    }
    assert r06["evidence"]["opticalPositionIncreasePercent"] <= 15.0
    assert r06["evidence"]["infraredPositionIncreasePercent"] <= 15.0
    assert r06["evidence"]["radarPositionIncreasePercent"] <= 15.0


def test_r08_failure_cases_are_reproducibly_frozen_by_seed():
    cases = _config()["metricConfigurations"]["R08"]["failureCases"]

    assert cases["spatiallyDistributed"] == [0, 5, 10]
    assert cases["topologicallyAdjacent"] == [0, 1, 2]
    assert cases["highVisibility"] == [0, 12, 14]
    assert cases["seededRandom"] == {
        "0": [7, 8, 11],
        "1": [6, 7, 11],
        "2": [1, 3, 10],
        "3": [1, 2, 10],
        "4": [9, 13, 14],
    }
