from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ring_cann_failure_reanchoring import (
    generate_failure_reanchoring_figure,
    run_failure_reanchoring_benchmark,
    write_failure_reanchoring_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test sparse phase reanchoring under permanent neuron failures."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--cue-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/failure_reanchoring"))
    args = parser.parse_args(argv)
    traces = run_failure_reanchoring_benchmark(
        duration=args.duration, sample_dt=args.sample_dt,
        cue_interval=args.cue_interval, seed=args.seed,
    )
    summary = write_failure_reanchoring_summary(
        traces, args.output_dir / "failure_reanchoring_summary.json",
    )
    figure = generate_failure_reanchoring_figure(
        traces, args.output_dir / "failure_reanchoring.png",
    )
    print(summary)
    print(figure)
    return traces


if __name__ == "__main__":
    main()
