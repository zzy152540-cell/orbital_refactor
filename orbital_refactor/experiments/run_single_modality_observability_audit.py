from __future__ import annotations

import argparse
from pathlib import Path

from experiments.single_modality_observability_audit import (
    run_single_modality_observability_audit,
    save_single_modality_observability_audit,
)


def main():
    parser = argparse.ArgumentParser(description="Audit P08 sensor observability.")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument(
        "--infrared-noise-scales", nargs="+", type=float,
        default=[0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p08_observability_audit"),
    )
    args = parser.parse_args()
    report = run_single_modality_observability_audit(
        duration=args.duration, dt=args.dt,
        infrared_noise_scales=args.infrared_noise_scales,
    )
    paths = save_single_modality_observability_audit(report, args.output)
    print(
        "optical_infrared_row_space_overlap_mean="
        f"{report.optical_infrared_row_space_overlap_mean:.9f}"
    )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
