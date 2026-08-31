from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ring_cann_benchmark import (
    generate_ring_cann_benchmark_figure,
    run_ring_cann_benchmark,
    write_ring_cann_benchmark_csv,
)


def main(argv=None) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(
        description="Run the standalone Zhang-1996 ring CANN benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/cann"))
    parser.add_argument("--sample-dt", type=float, default=0.02)
    arguments = parser.parse_args(argv)
    traces = run_ring_cann_benchmark(sample_dt=arguments.sample_dt)
    csv_path = write_ring_cann_benchmark_csv(
        traces, arguments.output_dir / "ring_cann_benchmark.csv",
    )
    figure_path = generate_ring_cann_benchmark_figure(
        traces, arguments.output_dir / "ring_cann_benchmark.png",
    )
    print(csv_path)
    print(figure_path)
    return csv_path, figure_path


if __name__ == "__main__":
    main()
