from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from brain_inspired.modality_shadow_quality import build_modality_shadow_quality
from brain_inspired.navigation_shadow_quality import build_navigation_shadow_quality
from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_radial_runner import run_orbital_radial_states
from experiments.walker_direction_navigation_comparison import (
    _periodic_anchor_mask,
    _posterior_anchor_confidence,
)
from experiments.walker_optical_image_comparison import (
    replace_optical_messages_with_images,
)
from experiments.walker_raw_sensor_comparison import (
    _run,
    _walker_case,
    replace_infrared_messages_with_body_pair,
    replace_radar_messages_with_power_maps,
)


def run_walker_modality_shadow_quality_audit(
    *, duration=20.0, dt=2.0, seed=0, anchor_interval_samples=2,
):
    """Build diagnostic-only modality quality from real raw Walker messages."""
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    messages, _ = replace_optical_messages_with_images(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"],
        config=OpticalCameraConfig(
            width=128, height=128, focal_length_x_pixels=100.0,
            focal_length_y_pixels=100.0, source_peak=100.0,
            background=10.0, read_noise_sigma=10.0,
        ),
        rng=np.random.default_rng(20262001 + seed),
    )
    _, messages = replace_infrared_messages_with_body_pair(
        messages, timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"],
        config=InfraredCameraConfig(
            width=128, height=128, focal_length_x_pixels=300.0,
            focal_length_y_pixels=300.0, source_peak=100.0,
            background=10.0, read_noise_sigma=10.0,
        ),
        analytic_rng=np.random.default_rng(20262002 + seed),
        image_rng=np.random.default_rng(20262003 + seed),
    )
    messages = replace_radar_messages_with_power_maps(
        messages, timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"],
        config=RadarRangeDopplerConfig(
            width=129, height=129, range_bin_size_m=8.0,
            range_rate_bin_size_mps=0.2, source_peak=100.0,
            background=10.0, read_noise_sigma=10.0,
        ),
        rng=np.random.default_rng(20262004 + seed),
    )
    filtered = _run(case, messages)
    times = np.asarray(case["timestamps"], dtype=float)
    nodes = tuple(filtered.node_ids)
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(case["initial_states"][node])
        for node in nodes
    }
    mask = _periodic_anchor_mask(times.size, anchor_interval_samples)
    masks = {node: mask for node in nodes}
    confidence = {
        node: _posterior_anchor_confidence(
            filtered.active_covariance_history_by_node[node],
        )
        for node in nodes
    }
    common = dict(
        timestamps=times,
        posterior_state_history_by_node=filtered.active_state_history_by_node,
        frame_by_node=frames, anchor_mask_by_node=masks,
        anchor_confidence_by_node=confidence,
    )
    direction = run_orbital_direction_states(**common)
    radial = run_orbital_radial_states(**common)
    navigation = build_navigation_shadow_quality(
        direction_by_node=direction, radial_by_node=radial,
    )
    modality = build_modality_shadow_quality(
        observations=messages, navigation_quality_by_node=navigation,
    )
    summary = {}
    for name in sorted({item.modality for item in modality}):
        selected = [item for item in modality if item.modality == name]
        summary[name] = {
            "count": len(selected),
            "valid_fraction": float(np.mean([item.valid for item in selected])),
            "mean_frontend_quality": float(np.mean([
                item.frontend_quality for item in selected
            ])),
            "mean_navigation_quality": float(np.mean([
                item.navigation_quality for item in selected
            ])),
            "mean_shadow_quality": float(np.mean([
                item.shadow_quality for item in selected
            ])),
            "minimum_shadow_quality": float(np.min([
                item.shadow_quality for item in selected
            ])),
        }
    return {"records": modality, "summary": summary}


def save_walker_modality_shadow_quality(
    result, output_path="results/cann/walker_modality_shadow_quality.csv",
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(result["records"][0].__dict__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item.__dict__ for item in result["records"])
    return path


if __name__ == "__main__":
    audit = run_walker_modality_shadow_quality_audit()
    print(audit["summary"])
    print(save_walker_modality_shadow_quality(audit))
