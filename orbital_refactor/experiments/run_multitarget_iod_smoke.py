"""Run a noisy raw-front-end 3-observer/2-target IOD smoke scenario."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MultimodalSensorSimulationConfig,
    simulate_multimodal_sensor_history,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    IODBufferConfig,
    MultiTargetIODManager,
    iod_observations_from_messages,
)


def run_multitarget_iod_smoke(
    output_directory,
    *,
    duration: float = 20.0,
    dt: float = 10.0,
    random_seed: int = 0,
    image_size: int = 33,
    inject_radar_outlier: bool = True,
):
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    if timestamps.size < 3:
        raise ValueError("IOD smoke scenario requires at least three epochs.")
    observers = {
        "observer-1": _state(0.0, 54.8, 699e3),
        "observer-2": _state(0.2, 54.9, 701e3),
        "observer-3": _state(-0.2, 54.7, 698e3),
    }
    targets = {
        "target-1": _state(0.6, 55.2, 703e3),
        "target-2": _state(-0.7, 55.3, 704e3),
    }
    observer_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in observers.items()
    }
    target_histories = {
        key: propagate_absolute_orbit(value, timestamps)
        for key, value in targets.items()
    }
    sensors = _sensors(image_size)
    messages_by_epoch = {float(timestamp): [] for timestamp in timestamps}
    state_by_epoch = {}
    for target_index, (target_id, target_history) in enumerate(target_histories.items()):
        for observer_index, (observer_id, observer_history) in enumerate(
            observer_histories.items()
        ):
            attitude = np.asarray([
                dcm_to_quat_wxyz(
                    _tracking_camera_basis(
                        target_history[index, :3] - observer_history[index, :3]
                    ).T
                )
                for index in range(len(timestamps))
            ])
            raw = simulate_multimodal_sensor_history(
                timestamps=timestamps,
                observer_id=observer_id,
                target_id=target_id,
                observer_state_history=observer_history,
                target_state_history=target_history,
                quaternion_i2b_wxyz_history_by_modality={
                    "OPTICAL": attitude,
                    "INFRARED": attitude,
                },
                config=sensors,
                random_seed=int(random_seed) + 1000 * target_index + 10 * observer_index,
            )
            for message in raw.messages:
                messages_by_epoch[float(message.timestamp)].append(message)
            for index, timestamp in enumerate(timestamps):
                state_by_epoch[(observer_id, float(timestamp))] = observer_history[index]
    if inject_radar_outlier:
        middle = float(timestamps[len(timestamps) // 2])
        values = messages_by_epoch[middle]
        for index, message in enumerate(values):
            if message.modality == "RADAR" and message.target_id == "target-1":
                values[index] = replace(
                    message,
                    measurement=np.asarray(message.measurement) + np.array([2000.0, 2.0]),
                    metadata={**message.metadata, "injectedOutlier": True},
                )
                break
    manager = MultiTargetIODManager(
        track_id_by_target={target_id: f"track-{target_id}" for target_id in targets},
        buffer_config=IODBufferConfig(
            minimum_epochs=3,
            minimum_observers=2,
            minimum_time_span_seconds=min(20.0, duration),
            maximum_time_span_seconds=max(60.0, duration),
        ),
    )
    updates = []
    for timestamp in timestamps:
        iod = iod_observations_from_messages(
            messages_by_epoch[float(timestamp)],
            observer_state_by_epoch=state_by_epoch,
        )
        updates.append(manager.ingest(iod))
    result_by_target = {}
    for target_id, estimate in updates[-1].estimates_by_target.items():
        truth = targets[target_id]
        result_by_target[target_id] = {
            "status": estimate.status.value,
            "converged": estimate.converged,
            "positionErrorM": float(np.linalg.norm(estimate.state_eci[:3] - truth[:3])),
            "velocityErrorMps": float(np.linalg.norm(estimate.state_eci[3:] - truth[3:])),
            "qualityScore": estimate.quality_score,
            "diagnostics": dict(estimate.diagnostics),
        }
    summary = {
        "scenario": "raw-front-end 3-observer 2-target IOD",
        "observerCount": len(observers),
        "targetCount": len(targets),
        "epochCount": len(timestamps),
        "radarOutlierInjected": bool(inject_radar_outlier),
        "initializedTargetIds": sorted(manager.initialized_states),
        "resultsByTarget": result_by_target,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def _state(anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m, 0.001, np.deg2rad(inclination_deg),
        np.deg2rad(15.0), 0.0, np.deg2rad(anomaly_deg),
    )


def _sensors(image_size):
    common = {
        "width": int(image_size), "height": int(image_size),
        "read_noise_sigma": 1.0,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def _tracking_camera_basis(boresight):
    forward = np.asarray(boresight, dtype=float).copy()
    forward /= np.linalg.norm(forward)
    lateral = np.cross([0.0, 0.0, 1.0], forward)
    if np.linalg.norm(lateral) < 1e-8:
        lateral = np.cross([0.0, 1.0, 0.0], forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return np.column_stack((forward, lateral, vertical))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/multitarget_iod_smoke")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=33)
    parser.add_argument("--no-radar-outlier", action="store_true")
    args = parser.parse_args()
    summary = run_multitarget_iod_smoke(
        args.output, duration=args.duration, dt=args.dt,
        random_seed=args.seed, image_size=args.image_size,
        inject_radar_outlier=not args.no_radar_outlier,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
