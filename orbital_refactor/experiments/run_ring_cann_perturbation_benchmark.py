from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ring_cann_perturbation_benchmark import (
    generate_ring_cann_perturbation_figure,
    run_ring_cann_perturbation_benchmark,
    write_ring_cann_perturbation_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure ring-CANN recovery after transient neural damage."
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sample-dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/perturbation"))
    args = parser.parse_args(argv)
    traces = run_ring_cann_perturbation_benchmark(
        recovery_duration=args.duration, sample_dt=args.sample_dt, seed=args.seed,
    )
    summary = write_ring_cann_perturbation_summary(
        traces, args.output_dir / "ring_cann_perturbation_summary.json",
    )
    figure = generate_ring_cann_perturbation_figure(
        traces, args.output_dir / "ring_cann_perturbation.png",
    )
    print(summary)
    print(figure)
    return traces


if __name__ == "__main__":
    main()
