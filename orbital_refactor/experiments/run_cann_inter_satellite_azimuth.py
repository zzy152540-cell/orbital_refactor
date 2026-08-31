import argparse
from pathlib import Path

from experiments.cann_inter_satellite_azimuth import (
    generate_inter_satellite_azimuth_figure,
    run_inter_satellite_azimuth_benchmark,
    write_inter_satellite_azimuth_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Walker optical/IR azimuth CANN comparison."
    )
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outage-start", type=float, default=200.0)
    parser.add_argument("--outage-end", type=float, default=400.0)
    parser.add_argument("--rotate-sensor-frame-to-boundary", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cann/inter_satellite_azimuth"))
    args = parser.parse_args(argv)
    result = run_inter_satellite_azimuth_benchmark(
        duration=args.duration, dt=args.dt, seed=args.seed,
        outage_window=(args.outage_start, args.outage_end),
        rotate_sensor_frame_to_boundary=args.rotate_sensor_frame_to_boundary,
    )
    root = args.output_dir
    print(write_inter_satellite_azimuth_summary(result, root / "summary.json"))
    print(generate_inter_satellite_azimuth_figure(result, root / "overview.png"))
    return result


if __name__ == "__main__":
    main()
