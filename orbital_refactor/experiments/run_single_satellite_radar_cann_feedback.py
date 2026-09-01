import argparse
from pathlib import Path

from experiments.single_satellite_radar_cann_feedback import (
    run_single_satellite_radar_cann_feedback,
    write_single_satellite_radar_cann_feedback,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fault-mode", choices=("impulsive", "none"), default="impulsive",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cann/single_satellite_radar_feedback"),
    )
    args = parser.parse_args(argv)
    result = run_single_satellite_radar_cann_feedback(
        seed=args.seed,
        radar_fault_mode=None if args.fault_mode == "none" else args.fault_mode,
    )
    print(result["summary"])
    print(write_single_satellite_radar_cann_feedback(
        result, args.output_dir,
    ))


if __name__ == "__main__":
    main()
