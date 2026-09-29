import json

from experiments.run_multitarget_validation_matrix import run_validation_matrix


def test_validation_matrix_runner_writes_all_three_cases(tmp_path):
    result = run_validation_matrix(
        tmp_path,
        duration=2.0,
        dt=1.0,
        random_seed=3,
        image_size=17,
    )

    assert set(result["cases"]) == {
        "A_3observer_1target_full",
        "B_3observer_2target_full",
        "C_3observer_2target_partial_outage",
        "D_3observer_2target_full_target_outage",
    }
    saved = json.loads(
        (tmp_path / "validation_matrix_summary.json").read_text(encoding="utf-8")
    )
    assert saved["frame"] == "J2000_ECI"
    assert set(saved["cases"]["B_3observer_2target_full"]["metricsByTarget"]) == {
        "target-1", "target-2",
    }
    for case_name in result["cases"]:
        assert (tmp_path / case_name / "summary.json").is_file()
    diagnostics = saved["cases"]["A_3observer_1target_full"][
        "modalityDiagnosticsByLink"
    ]
    assert set(diagnostics["observer-1->target-1"]) == {
        "RADAR", "INFRARED", "OPTICAL",
    }
