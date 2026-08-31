from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.ring_cann_stress_benchmark import (
    generate_ring_cann_stress_figure,
    run_ring_cann_stress_benchmark,
    write_ring_cann_stress_csv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the ring-CANN stress benchmark.")
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--sample-dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cue-gain", type=float, default=0.05)
    parser.add_argument("--gate-threshold-deg", type=float, default=1.0)
    parser.add_argument("--complementary-gain", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/stress"))
    args = parser.parse_args(argv)
    result = run_ring_cann_stress_benchmark(
        duration=args.duration, sample_dt=args.sample_dt, seed=args.seed,
        cue_gain=args.cue_gain, gate_threshold_deg=args.gate_threshold_deg,
        complementary_gain=args.complementary_gain,
    )
    csv_path = write_ring_cann_stress_csv(
        result, args.output_dir / "ring_cann_stress.csv",
    )
    figure_path = generate_ring_cann_stress_figure(
        result, args.output_dir / "ring_cann_stress.png",
    )
    summary = {
        "phase_rmse_deg_by_mode": result.phase_rmse_deg_by_mode,
        "outage_rmse_deg_by_mode": result.outage_rmse_deg_by_mode,
        "final_error_deg_by_mode": result.final_error_deg_by_mode,
        "available_cue_count": int(result.hint_available.sum()),
        "accepted_cue_count": int(result.hint_accepted.sum()),
        "csv": str(csv_path), "figure": str(figure_path),
    }
    (args.output_dir / "ring_cann_stress_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    main()
