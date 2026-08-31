from __future__ import annotations

import argparse
from pathlib import Path

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
    write_single_satellite_cann_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outage-modalities", default="opt,ir,rad",
        help="Comma-separated subset of opt,ir,rad; empty disables outages.",
    )
    parser.add_argument("--disable-cann", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cann/single_satellite"))
    args = parser.parse_args()
    result = run_single_satellite_cann_comparison(
        duration=args.duration, dt=args.dt, seed=args.seed,
        outage_modalities=tuple(
            name.strip() for name in args.outage_modalities.split(",") if name.strip()
        ),
        enable_cann=not args.disable_cann,
    )
    paths = write_single_satellite_cann_results(result, args.output_dir)
    print(result["summary"])
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    main()
