from __future__ import annotations

import argparse
from pathlib import Path

from experiments.v15_system_acceptance import (
    SYSTEM_ACCEPTANCE_CASE_IDS,
    run_v15_system_acceptance,
    save_system_acceptance_csv,
    save_system_acceptance_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run V15 system smoke acceptance.")
    parser.add_argument("--profile", default="smoke", choices=("smoke",))
    parser.add_argument(
        "--cases", nargs="+", choices=SYSTEM_ACCEPTANCE_CASE_IDS,
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v15_system_acceptance_smoke.json"),
    )
    parser.add_argument("--csv-output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_v15_system_acceptance(
        profile=arguments.profile, case_ids=arguments.cases,
    )
    path = save_system_acceptance_report(report, arguments.output)
    csv_path = save_system_acceptance_csv(
        report,
        arguments.csv_output or arguments.output.with_suffix(".csv"),
    )
    for record in report.records:
        print(
            f"{'PASS' if record.passed else 'FAIL'} {record.case_id} "
            f"({record.runtime_seconds:.3f} s)"
        )
        for reason in record.failure_reasons:
            print(f"  - {reason}")
    print(f"saved report to {path}")
    print(f"saved summary to {csv_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
