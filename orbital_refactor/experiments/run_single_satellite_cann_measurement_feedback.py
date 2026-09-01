import argparse
from pathlib import Path

from experiments.single_satellite_cann_measurement_feedback import (
    run_single_satellite_cann_measurement_feedback,
    write_single_satellite_cann_measurement_feedback,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cann/single_satellite_measurement_feedback"),
    )
    args = parser.parse_args(argv)
    result = run_single_satellite_cann_measurement_feedback(seed=args.seed)
    print(result["summary"])
    print(write_single_satellite_cann_measurement_feedback(
        result, args.output_dir,
    ))


if __name__ == "__main__":
    main()
