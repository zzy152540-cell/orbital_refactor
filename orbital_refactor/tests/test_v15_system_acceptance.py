import json
from pathlib import Path

import pytest

from experiments.v15_system_acceptance import (
    SYSTEM_ACCEPTANCE_CASE_IDS,
    SystemAcceptanceRecord,
    SystemAcceptanceReport,
    run_v15_system_acceptance,
    save_system_acceptance_csv,
    save_system_acceptance_report,
)


def test_system_acceptance_rejects_unknown_or_duplicate_cases():
    with pytest.raises(ValueError):
        run_v15_system_acceptance(case_ids=("unknown",))
    with pytest.raises(ValueError):
        run_v15_system_acceptance(case_ids=(
            SYSTEM_ACCEPTANCE_CASE_IDS[0], SYSTEM_ACCEPTANCE_CASE_IDS[0],
        ))


def test_system_acceptance_report_is_pickle_free_json():
    report = SystemAcceptanceReport(
        profile="smoke", passed=True,
        records=(SystemAcceptanceRecord(
            case_id="case", passed=True, runtime_seconds=0.1,
            python_peak_memory_mb=2.0,
            metrics={"finite": 1.0},
        ),),
    )
    path = save_system_acceptance_report(
        report, Path("results/cann/test_v15_system_acceptance_report.json")
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["records"][0]["case_id"] == "case"
    csv_path = save_system_acceptance_csv(
        report, Path("results/cann/test_v15_system_acceptance_report.csv")
    )
    assert "python_peak_memory_mb" in csv_path.read_text(encoding="utf-8-sig")
