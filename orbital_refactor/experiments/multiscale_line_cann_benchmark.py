from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.line_cann import LineCANN
from brain_inspired.multiscale_line_cann import (
    MultiScaleLineCANN,
    MultiScaleLineCANNConfig,
)


@dataclass(frozen=True)
class MultiScaleLineBenchmark:
    timestamps: np.ndarray
    truth: np.ndarray
    single_scale: np.ndarray
    multiscale: np.ndarray
    metrics: dict[str, float]


def run_multiscale_line_cann_benchmark(
    *, duration=600.0, dt=2.0, amplitude=20_000.0, period=300.0,
    config=None,
):
    cfg = config or MultiScaleLineCANNConfig()
    cfg.validate()
    times = np.arange(0.0, float(duration) + 0.5 * float(dt), float(dt))
    angular_rate = 2.0 * np.pi / float(period)
    truth = float(amplitude) * np.sin(angular_rate * times)
    interval_rates = np.diff(truth) / float(dt)
    single = LineCANN(cfg.coarse)
    multi = MultiScaleLineCANN(cfg)
    single_values = [single.reset(truth[0], timestamp=times[0]).decoded_value]
    first = multi.initialize(truth[0], timestamp=times[0])
    multi_values = [first.decoded_value]
    coarse_widths = [first.coarse_output.bump_width]
    fine_widths = [first.fine_output.bump_width]
    disagreements = [first.cross_scale_disagreement]
    cell_changes = [first.coarse_cell_changed]
    saturated = [first.saturated_at_boundary]
    for index in range(1, times.size):
        single_values.append(
            single.step(interval_rates[index - 1], dt).decoded_value
        )
        output = multi.step(interval_rates[index - 1], dt)
        multi_values.append(output.decoded_value)
        coarse_widths.append(output.coarse_output.bump_width)
        fine_widths.append(output.fine_output.bump_width)
        disagreements.append(output.cross_scale_disagreement)
        cell_changes.append(output.coarse_cell_changed)
        saturated.append(output.saturated_at_boundary)
    single_values = np.asarray(single_values)
    multi_values = np.asarray(multi_values)
    single_error = single_values - truth
    multi_error = multi_values - truth
    return MultiScaleLineBenchmark(
        timestamps=times, truth=truth, single_scale=single_values,
        multiscale=multi_values,
        metrics={
            "single_rmse": float(np.sqrt(np.mean(single_error**2))),
            "multiscale_rmse": float(np.sqrt(np.mean(multi_error**2))),
            "single_maximum_error": float(np.max(np.abs(single_error))),
            "multiscale_maximum_error": float(np.max(np.abs(multi_error))),
            "coarse_mean_bump_width": float(np.mean(coarse_widths)),
            "fine_mean_bump_width": float(np.mean(fine_widths)),
            "mean_cross_scale_disagreement": float(np.mean(np.abs(disagreements))),
            "coarse_cell_change_count": float(np.count_nonzero(cell_changes)),
            "boundary_saturation_fraction": float(np.mean(saturated)),
        },
    )


if __name__ == "__main__":
    print(run_multiscale_line_cann_benchmark().metrics)
