import argparse
from pathlib import Path

from experiments.single_satellite_three_modal_cann_feedback import (
    run_staggered_modality_outage_comparison,
    run_single_satellite_three_modal_cann_feedback,
    write_single_satellite_three_modal_cann_feedback,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-faults", action="store_true")
    parser.add_argument(
        "--staggered-outages", action="store_true",
        help="Use independent optical, infrared, and radar outage windows.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cann/single_satellite_three_modal_feedback"),
    )
    args = parser.parse_args(argv)
    if args.staggered_outages:
        result = run_staggered_modality_outage_comparison(
            seed=args.seed,
            outage_windows={
            "opt": (1300.0, 1500.0),
            "ir": (600.0, 800.0),
            "rad": (900.0, 1100.0),
            },
        )
    else:
        result = run_single_satellite_three_modal_cann_feedback(
            seed=args.seed, inject_faults=not args.no_faults,
        )
    print(result["summary"])
    for kind, path in write_single_satellite_three_modal_cann_feedback(
        result, args.output_dir,
    ).items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
