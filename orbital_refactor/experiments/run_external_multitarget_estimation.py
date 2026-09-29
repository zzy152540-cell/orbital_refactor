"""Run known-target cooperative estimation from an external scene JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from tracking import (
    build_standard_multitarget_scenario_from_input,
    load_external_multitarget_input,
    run_known_target_sequence,
)


def run_external_multitarget_estimation(
    input_path,
    output_directory,
    *,
    random_seed: int = 0,
    image_size: int = 65,
    max_coast_epochs: int = 3,
):
    value = load_external_multitarget_input(input_path)
    sensors = _sensor_config(image_size)
    scenario = build_standard_multitarget_scenario_from_input(
        value,
        sensor_config=sensors,
        random_seed=random_seed,
    )
    history = run_known_target_sequence(
        scene_id=value.scene_id,
        module_inputs=scenario.module_inputs,
        initial_target_states=scenario.initial_target_states,
        max_coast_epochs=max_coast_epochs,
    )
    start_time_ms = int(value.config["startTimeMs"])
    frames = []
    for epoch_index, timestamp in enumerate(history.timestamps):
        target_list = []
        for target_id, tracks in history.track_history_by_target.items():
            track = tracks[epoch_index]
            estimate = track.estimate
            state = np.asarray(estimate.state_eci, dtype=float)
            covariance = np.asarray(estimate.covariance_eci, dtype=float)
            target_list.append({
                "targetId": target_id,
                "trackId": estimate.track_id,
                "frame": estimate.frame,
                "lifecycle": track.lifecycle.value,
                "x": float(state[0]),
                "y": float(state[1]),
                "z": float(state[2]),
                "vx": float(state[3]),
                "vy": float(state[4]),
                "vz": float(state[5]),
                "covarianceDiagonal": np.diag(covariance).tolist(),
                "contributingObserverIds": list(
                    getattr(estimate, "contributing_observer_ids", ())
                ),
                "nodeWeights": dict(getattr(estimate, "node_weights", {})),
            })
        time_ms = start_time_ms + int(round(float(timestamp) * 1000.0))
        frames.append({
            "frameIndex": epoch_index,
            "timeMs": time_ms,
            "time": _format_utc_milliseconds(time_ms),
            "targetList": target_list,
        })
    payload = {
        "type": "MULTI_TARGET_ESTIMATION",
        "protocolVersion": "MULTI-TARGET-ESTIMATION-1.0",
        "sceneId": value.scene_id,
        "coordinateFrame": "J2000_ECI",
        "stateUnits": {"position": "m", "velocity": "m/s"},
        "initializationMethod": value.config["initializationMethod"],
        "initialOrbitDeterminationPerformed": False,
        "observerIds": list(value.observer_states),
        "targetIds": list(value.target_initial_states),
        "sampleIntervalSeconds": float(value.config["stepSeconds"]),
        "frameCount": len(frames),
        "frames": frames,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    estimate_path = output / "estimated_targets.json"
    estimate_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    summary = {
        "sceneId": value.scene_id,
        "observerCount": len(value.observer_states),
        "targetCount": len(value.target_initial_states),
        "localFilterTaskCount": len(scenario.module_inputs),
        "frameCount": len(frames),
        "coordinateFrame": "J2000_ECI",
        "estimatedTargets": str(estimate_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def _sensor_config(image_size):
    size = int(image_size)
    if size < 9:
        raise ValueError("image_size must be at least 9 pixels.")
    common = {"width": size, "height": size, "read_noise_sigma": 0.5}
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def _format_utc_milliseconds(time_ms):
    value = datetime.fromtimestamp(float(time_ms) / 1000.0, tz=timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="External scene JSON with estimationConfig.")
    parser.add_argument("--output", default="results/external_multitarget_estimation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=65)
    parser.add_argument("--max-coast-epochs", type=int, default=3)
    args = parser.parse_args()
    summary = run_external_multitarget_estimation(
        args.input,
        args.output,
        random_seed=args.seed,
        image_size=args.image_size,
        max_coast_epochs=args.max_coast_epochs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
