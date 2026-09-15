from __future__ import annotations

import argparse
from pathlib import Path

from experiments.single_multi_measurement_source_acceptance import (
    run_single_multi_measurement_source_acceptance,
    save_single_multi_measurement_source_acceptance,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed aligned measurement-source acceptance."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--calibration-seed", type=int, default=701)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "p08_single_multi_measurement_source_acceptance"
        ),
    )
    args = parser.parse_args()
    summary, rows = run_single_multi_measurement_source_acceptance(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        calibration_seed=args.calibration_seed,
        calibration_samples=args.calibration_samples,
    )
    paths = save_single_multi_measurement_source_acceptance(
        summary, rows, args.output,
    )
    print(f"acceptance_passed={summary.acceptance_passed}")
    print(f"walker_raw_mean_nis={summary.walker_raw_mean_nis}")
    print(f"mean_position_rmse_m={summary.mean_position_rmse_m}")
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()
