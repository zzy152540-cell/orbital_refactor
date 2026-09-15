from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_extent_observability_audit import (
    run_infrared_extent_observability_audit,
    save_infrared_extent_observability_audit,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit a shadow infrared angular-extent measurement."
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--effective-target-diameter", type=float, default=10.0)
    parser.add_argument(
        "--extent-fractional-sigmas", nargs="+", type=float,
        default=[0.01, 0.05, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_ir_extent_shadow_audit"),
    )
    args = parser.parse_args()
    report = run_infrared_extent_observability_audit(
        duration=args.duration, dt=args.dt,
        effective_target_diameter_m=args.effective_target_diameter,
        extent_fractional_sigmas=args.extent_fractional_sigmas,
    )
    paths = save_infrared_extent_observability_audit(report, args.output)
    print(f"records={len(report.records)} baseline_unchanged={report.baseline_unchanged}")
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
