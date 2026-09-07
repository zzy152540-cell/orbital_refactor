from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from brain_inspired.line_cann import LineCANNConfig
from brain_inspired.orbital_radial_state import OrbitalRadialConfig
from experiments.walker_radial_navigation_comparison import (
    run_walker_radial_navigation_comparison,
)


@dataclass(frozen=True)
class RadialParameterRecord:
    num_neurons: int
    tuning_width_m: float
    maximum_anchor_gain: float
    radial_rmse_m: float
    maximum_radial_error_m: float
    posterior_tracking_rmse_m: float
    boundary_saturation_fraction: float


def run_walker_radial_parameter_scan(
    *, neuron_counts=(81, 161, 321), tuning_widths_m=(500.0, 1_000.0),
    anchor_gains=(0.25, 0.4), duration=120.0, dt=5.0, seed=0,
    radial_rate_bias_mps=0.2,
):
    """Screen radial coding resolution and bounded anchor gain."""
    records = []
    for neurons in neuron_counts:
        for width in tuning_widths_m:
            for gain in anchor_gains:
                config = OrbitalRadialConfig(
                    line=LineCANNConfig(
                        num_neurons=int(neurons), minimum_value=-20_000.0,
                        maximum_value=20_000.0, tuning_width=float(width),
                    ),
                    maximum_anchor_gain=float(gain),
                )
                comparison = run_walker_radial_navigation_comparison(
                    duration=duration, dt=dt, seed=seed,
                    radial_rate_bias_mps=radial_rate_bias_mps,
                    radial_config=config,
                )
                summary = comparison.metrics["periodic_anchor"]
                records.append(RadialParameterRecord(
                    num_neurons=int(neurons), tuning_width_m=float(width),
                    maximum_anchor_gain=float(gain),
                    radial_rmse_m=summary["radial_rmse_m"],
                    maximum_radial_error_m=summary["maximum_radial_error_m"],
                    posterior_tracking_rmse_m=(
                        summary["posterior_tracking_rmse_m"]
                    ),
                    boundary_saturation_fraction=(
                        summary["boundary_saturation_fraction"]
                    ),
                ))
    return records


def save_radial_parameter_scan(
    records, output_path="results/cann/walker_radial_parameter_scan.csv",
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RadialParameterRecord.__annotations__)
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    return path


if __name__ == "__main__":
    result = run_walker_radial_parameter_scan()
    print(min(result, key=lambda record: record.radial_rmse_m))
    print(save_radial_parameter_scan(result))
