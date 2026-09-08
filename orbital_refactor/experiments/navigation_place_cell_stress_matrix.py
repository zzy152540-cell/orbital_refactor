from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from brain_inspired.navigation_place_cells import (
    NavigationPlaceCellConfig,
    NavigationPlaceCellEncoder,
)
from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellEncoder,
)


@dataclass(frozen=True)
class NavigationPlaceCellStressResult:
    timestamps: np.ndarray
    metrics: dict[str, dict[str, dict[str, float]]]
    configuration_cell_counts: dict[str, int]


def run_navigation_place_cell_stress_matrix(*, duration=1800.0, dt=5.0):
    """Exercise place-cell support without invoking or modifying the filter."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    phase = np.mod(0.3 + 2.0 * np.pi * times / 5700.0, 2.0 * np.pi)
    normalized = times / duration
    profiles = {
        "nominal": np.column_stack([
            40.0 * np.sin(2.0 * np.pi * normalized),
            60.0 * np.cos(2.0 * np.pi * normalized),
        ]),
        "boundary_crossing": np.column_stack([
            160.0 * np.sin(2.0 * np.pi * normalized),
            180.0 * np.cos(2.0 * np.pi * normalized),
        ]),
        "large_smooth_offset": np.column_stack([
            15_000.0 * np.sin(2.0 * np.pi * normalized),
            12_000.0 * np.cos(2.0 * np.pi * normalized),
        ]),
        "unanchored_ramp": np.column_stack([
            -15_000.0 + 30_000.0 * normalized,
            10_000.0 - 20_000.0 * normalized,
        ]),
        "coarse_support_excursion": np.column_stack([
            25_000.0 * np.cos(2.0 * np.pi * normalized),
            30_000.0 * np.cos(2.0 * np.pi * normalized),
        ]),
    }
    compact = NavigationPlaceCellEncoder()
    wide_config = NavigationPlaceCellConfig(
        radial_centers_m=tuple(np.linspace(-20_000.0, 20_000.0, 9)),
        along_track_centers_m=tuple(np.linspace(-20_000.0, 20_000.0, 9)),
        radial_sigma_m=2_500.0, along_track_sigma_m=2_500.0,
    )
    wide = NavigationPlaceCellEncoder(wide_config)
    metrics = {}
    for case_name, rt in profiles.items():
        compact_outputs, compact_seconds = _encode(compact, phase, rt)
        wide_outputs, wide_seconds = _encode(wide, phase, rt)
        hierarchical = HierarchicalNavigationPlaceCellEncoder()
        hierarchical_outputs, hierarchical_seconds = _encode(
            hierarchical, phase, rt,
        )
        metrics[case_name] = {
            "compact": _metrics(compact_outputs, phase, rt, compact_seconds),
            "wide": _metrics(wide_outputs, phase, rt, wide_seconds),
            "dual_scale": _dual_scale_metrics(
                compact_outputs, wide_outputs, phase, rt,
                compact_seconds + wide_seconds,
            ),
            "hierarchical": _metrics(
                hierarchical_outputs, phase, rt, hierarchical_seconds,
            ),
        }
    return NavigationPlaceCellStressResult(
        timestamps=times, metrics=metrics,
        configuration_cell_counts={
            "compact": compact.config.cell_count,
            "wide": wide.config.cell_count,
            "dual_scale": compact.config.cell_count + wide.config.cell_count,
            "hierarchical": (
                HierarchicalNavigationPlaceCellEncoder().config.cell_count
            ),
        },
    )


def _encode(encoder, phase, rt):
    started = perf_counter()
    outputs = [encoder.encode(
        phase=phase[index], radial_position=rt[index, 0],
        along_track_position=rt[index, 1],
    ) for index in range(phase.size)]
    return outputs, perf_counter() - started


def _metrics(outputs, phase, rt, elapsed):
    decoded_phase = np.asarray([item.decoded_phase for item in outputs])
    decoded_rt = np.asarray([[
        item.decoded_radial_position, item.decoded_along_track_position,
    ] for item in outputs])
    summary = _summary(
        decoded_phase=decoded_phase, decoded_rt=decoded_rt, phase=phase, rt=rt,
        saturation=np.asarray([item.boundary_saturated for item in outputs]),
        entropy=np.asarray([item.normalized_entropy for item in outputs]),
        peaks=np.asarray([item.peak_activity for item in outputs]),
        elapsed=elapsed,
    )
    if hasattr(outputs[0], "fine_scale_active"):
        summary.update({
            "fine_scale_active_fraction": float(np.mean([
                item.fine_scale_active for item in outputs
            ])),
            "scale_transition_count": int(np.count_nonzero([
                item.scale_transition for item in outputs
            ])),
        })
    return summary


def _dual_scale_metrics(compact, wide, phase, rt, elapsed):
    use_wide = np.asarray([item.boundary_saturated for item in compact])
    decoded_phase = np.asarray([
        (wide[index] if use_wide[index] else compact[index]).decoded_phase
        for index in range(phase.size)
    ])
    decoded_rt = np.asarray([[
        (wide[index] if use_wide[index] else compact[index]).decoded_radial_position,
        (wide[index] if use_wide[index] else compact[index]).decoded_along_track_position,
    ] for index in range(phase.size)])
    active = [wide[index] if use_wide[index] else compact[index]
              for index in range(phase.size)]
    return _summary(
        decoded_phase=decoded_phase, decoded_rt=decoded_rt, phase=phase, rt=rt,
        saturation=use_wide & np.asarray([
            item.boundary_saturated for item in wide
        ]),
        entropy=np.asarray([item.normalized_entropy for item in active]),
        peaks=np.asarray([item.peak_activity for item in active]),
        elapsed=elapsed,
    )


def _summary(*, decoded_phase, decoded_rt, phase, rt, saturation, entropy,
             peaks, elapsed):
    phase_error = np.arctan2(
        np.sin(decoded_phase - phase), np.cos(decoded_phase - phase),
    )
    rt_error = decoded_rt - rt
    return {
        "phase_reconstruction_rmse_rad": float(np.sqrt(np.mean(
            phase_error ** 2
        ))),
        "rt_reconstruction_rmse_m": float(np.sqrt(np.mean(np.sum(
            rt_error ** 2, axis=1
        )))),
        "boundary_saturation_fraction": float(np.mean(saturation)),
        "mean_normalized_entropy": float(np.mean(entropy)),
        "mean_peak_activity": float(np.mean(peaks)),
        "mean_encoding_time_us_per_sample": float(
            1.0e6 * elapsed / phase.size
        ),
    }


if __name__ == "__main__":
    result = run_navigation_place_cell_stress_matrix()
    print("cells", result.configuration_cell_counts)
    for case_name, configurations in result.metrics.items():
        print(case_name, configurations)
