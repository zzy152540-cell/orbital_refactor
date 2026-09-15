from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infrared_boresight_shadow import run_infrared_boresight_shadow, save_infrared_boresight_shadow


def main():
    parser = argparse.ArgumentParser(description="Compare infrared boresight modes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--optical-outage-start", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/external_acceptance/p08_ir_boresight_shadow"))
    args = parser.parse_args()
    report = run_infrared_boresight_shadow(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
        calibration_samples=args.calibration_samples,
        optical_outage_start=args.optical_outage_start,
    )
    print(*save_infrared_boresight_shadow(report, args.output), sep="\n")
    for run in report.runs:
        print(run.seed, run.mode, f"increase={run.position_rmse_increase_percent:.6g}%", f"valid={run.outage_valid_fraction:.3f}", f"pointing={run.outage_boresight_mean_error_deg:.6g}deg")


if __name__ == "__main__":
    main()
