from __future__ import annotations

import argparse
from pathlib import Path

from experiments.external_single_cann_robustness_audit import (
    run_external_single_cann_robustness_audit,
    save_external_single_cann_robustness_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit CANN around P08 outages.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_cann_audit.json"),
    )
    arguments = parser.parse_args()
    report = run_external_single_cann_robustness_audit(seeds=arguments.seeds)
    print(save_external_single_cann_robustness_audit(report, arguments.output))


if __name__ == "__main__":
    main()
