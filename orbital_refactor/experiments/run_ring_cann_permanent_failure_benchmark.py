from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ring_cann_permanent_failure_benchmark import (
    generate_permanent_failure_figure,
    run_ring_cann_permanent_failure_benchmark,
    write_permanent_failure_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test a ring CANN with permanently disabled neurons."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/permanent_failure"))
    args = parser.parse_args(argv)
    traces = run_ring_cann_permanent_failure_benchmark(
        duration=args.duration, sample_dt=args.sample_dt, seed=args.seed,
    )
    summary = write_permanent_failure_summary(
        traces, args.output_dir / "ring_cann_permanent_failure_summary.json",
    )
    figure = generate_permanent_failure_figure(
        traces, args.output_dir / "ring_cann_permanent_failure.png",
    )
    print(summary)
    print(figure)
    return traces


if __name__ == "__main__":
    main()
