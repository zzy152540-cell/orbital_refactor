from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.walker_cann_sidecar_comparison import (
    generate_walker_cann_comparison_figure,
    run_walker_cann_sidecar_comparison,
    write_walker_cann_comparison_csv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a passive ring CANN on Walker truth/filter ECI states."
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cue-interval-samples", type=int, default=5)
    parser.add_argument("--node-id")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/walker_sidecar"))
    args = parser.parse_args(argv)
    result = run_walker_cann_sidecar_comparison(
        duration=args.duration, dt=args.dt, seed=args.seed,
        cue_interval_samples=args.cue_interval_samples, node_id=args.node_id,
    )
    csv_path = write_walker_cann_comparison_csv(
        result, args.output_dir / "walker_cann_sidecar.csv",
    )
    figure_path = generate_walker_cann_comparison_figure(
        result, args.output_dir / "walker_cann_sidecar.png",
    )
    summary = {
        "node_id": result.node_id,
        "source_position_rmse_m": result.source_position_rmse_m,
        "phase_rmse_deg_by_mode": result.phase_rmse_deg_by_mode,
        "maximum_phase_error_deg_by_mode": result.maximum_phase_error_deg_by_mode,
        "csv": str(csv_path), "figure": str(figure_path),
    }
    summary_path = args.output_dir / "walker_cann_sidecar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    main()
