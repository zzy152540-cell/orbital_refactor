import argparse
from pathlib import Path

from experiments.external_update_frequency_ablation import (
    run_external_update_frequency_ablation,
    save_external_update_frequency_ablation,
)


def main():
    parser = argparse.ArgumentParser(description="Run controlled P06 timing ablation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/external_acceptance/p06_timing_ablation.json"),
    )
    args = parser.parse_args()
    report = run_external_update_frequency_ablation(
        seeds=args.seeds, duration=args.duration, dt=args.dt,
    )
    path = save_external_update_frequency_ablation(report, args.output)
    for profile, latency in report.mean_latency_by_profile_ms.items():
        print(f"{profile}: {latency:.3f} ms/epoch")
    print(f"exact_replay_increment={report.exact_replay_increment_ms:.3f} ms/epoch")
    print(
        "relative_measurement_increment="
        f"{report.relative_measurement_increment_ms:.3f} ms/epoch"
    )
    print(path)


if __name__ == "__main__":
    main()
