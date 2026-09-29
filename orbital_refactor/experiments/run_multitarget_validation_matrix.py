"""Run the standard cooperative multi-target validation matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import MultimodalSensorSimulationConfig
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from tracking import (
    build_standard_multitarget_scenario,
    run_standard_multitarget_validation,
)


def run_validation_matrix(
    output_directory,
    *,
    duration: float = 120.0,
    dt: float = 2.0,
    random_seed: int = 0,
    image_size: int = 65,
):
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    if image_size < 9:
        raise ValueError("image_size must be at least 9 pixels.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    observers = _default_observers()
    targets = _default_targets()
    sensors = _sensor_config(image_size)
    middle_start = float(timestamps[max(1, len(timestamps) // 3)])
    middle_end = float(timestamps[min(len(timestamps) - 2, 2 * len(timestamps) // 3)])
    cases = {
        "A_3observer_1target_full": {
            "targets": {"target-1": targets["target-1"]},
            "dropouts": {},
        },
        "B_3observer_2target_full": {
            "targets": targets,
            "dropouts": {},
        },
        "C_3observer_2target_partial_outage": {
            "targets": targets,
            "dropouts": {
                ("observer-1", "target-1", modality): ((middle_start, middle_end),)
                for modality in ("RADAR", "INFRARED", "OPTICAL")
            } | {
                ("observer-2", "target-2", "INFRARED"): ((middle_start, middle_end),),
                ("observer-3", "target-2", "RADAR"): ((middle_start, middle_end),),
            },
        },
        "D_3observer_2target_full_target_outage": {
            "targets": targets,
            "dropouts": {
                (observer_id, "target-1", modality): ((middle_start, middle_end),)
                for observer_id in observers
                for modality in ("RADAR", "INFRARED", "OPTICAL")
            },
        },
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    initial_error = np.array([50.0, -40.0, 30.0, 0.05, -0.04, 0.03])
    covariance = np.diag([100.0, 100.0, 100.0, 0.2, 0.2, 0.2]) ** 2
    for case_index, (case_name, case) in enumerate(cases.items()):
        selected_targets = case["targets"]
        scenario = build_standard_multitarget_scenario(
            scene_id=case_name,
            timestamps=timestamps,
            observer_initial_states_eci=observers,
            target_initial_states_eci=selected_targets,
            initial_error_by_link={
                (observer_id, target_id): initial_error
                for observer_id in observers for target_id in selected_targets
            },
            initial_covariance=covariance,
            sensor_config=sensors,
            dropout_windows=case["dropouts"],
            random_seed=int(random_seed) + case_index * 100_000,
        )
        result = run_standard_multitarget_validation(scenario)
        case_summary = {
            "case": case_name,
            "durationSeconds": float(duration),
            "sampleIntervalSeconds": float(dt),
            "epochCount": len(timestamps),
            "observerIds": list(observers),
            "targetIds": list(selected_targets),
            "measurementSource": "raw_three_modality_frontend",
            "modalities": ["RADAR", "INFRARED", "OPTICAL"],
            "dropoutWindows": _json_dropout_windows(case["dropouts"]),
            "metricsByTarget": {
                target_id: asdict(metrics)
                for target_id, metrics in result.metrics_by_target.items()
            },
            "lifecycleByTarget": {
                target_id: [track.lifecycle.value for track in tracks]
                for target_id, tracks in result.history.track_history_by_target.items()
            },
            "activeObserversByTarget": {
                target_id: [
                    list(epoch.estimates_by_target[target_id].contributing_observer_ids)
                    if target_id in epoch.estimates_by_target else []
                    for epoch in result.history.output_by_epoch
                ]
                for target_id in selected_targets
            },
            "modalityDiagnosticsByLink": _modality_diagnostics(
                scenario.module_inputs,
                result.history.local_report_history_by_link,
            ),
        }
        case_directory = output / case_name
        case_directory.mkdir(parents=True, exist_ok=True)
        (case_directory / "summary.json").write_text(
            json.dumps(case_summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        summaries[case_name] = case_summary
    matrix = {
        "validationMatrix": "known-target cooperative estimation A/B/C/D",
        "frame": "J2000_ECI",
        "cases": summaries,
    }
    matrix_path = output / "validation_matrix_summary.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return matrix


def _default_observers():
    return {
        "observer-1": _state(0.0, 54.8, 699e3),
        "observer-2": _state(0.2, 54.9, 701e3),
        "observer-3": _state(-0.2, 54.7, 698e3),
    }


def _default_targets():
    return {
        "target-1": _state(0.6, 55.2, 703e3),
        "target-2": _state(-0.7, 55.3, 704e3),
    }


def _state(anomaly_deg, inclination_deg, altitude_m):
    return keplerian_to_eci(
        R_EARTH + altitude_m,
        0.001,
        np.deg2rad(inclination_deg),
        np.deg2rad(15.0),
        0.0,
        np.deg2rad(anomaly_deg),
    )


def _sensor_config(image_size):
    common = {
        "width": int(image_size), "height": int(image_size),
        "read_noise_sigma": 0.5,
    }
    return MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(**common),
        infrared=InfraredCameraConfig(**common),
        radar=RadarRangeDopplerConfig(**common),
    )


def _json_dropout_windows(windows):
    return [
        {
            "observerId": observer_id,
            "targetId": target_id,
            "modality": modality,
            "windowsSeconds": [list(window) for window in values],
        }
        for (observer_id, target_id, modality), values in windows.items()
    ]


def _modality_diagnostics(module_inputs, report_histories):
    aliases = {"RADAR": "rad", "INFRARED": "ir", "OPTICAL": "opt"}
    result = {}
    for module_input in module_inputs:
        target_id = str(module_input.initial_state.target_id)
        runtime = module_input.config.get("runtime", {})
        observer_id = str(runtime.get("node_id", ""))
        reports = report_histories[(observer_id, target_id)]
        epoch_count = len(reports)
        by_modality = {}
        for public_name, filter_name in aliases.items():
            observations = [
                item for item in module_input.sensor_measurements
                if str(item.modality).upper() == public_name
            ]
            configured_valid = sum(bool(item.valid_flag) for item in observations)
            active_weights = [
                float(report.modality_weights.get(filter_name, 0.0))
                for report in reports
            ]
            active_epochs = sum(weight > 0.0 for weight in active_weights)
            by_modality[public_name] = {
                "configuredValidEpochs": configured_valid,
                "configuredInvalidEpochs": len(observations) - configured_valid,
                "configuredValidFraction": (
                    configured_valid / len(observations) if observations else 0.0
                ),
                "filterActiveEpochs": active_epochs,
                "filterActiveFraction": active_epochs / epoch_count,
                "meanFusionWeightWhenActive": (
                    float(np.mean([weight for weight in active_weights if weight > 0.0]))
                    if active_epochs else 0.0
                ),
            }
        result[f"{observer_id}->{target_id}"] = by_modality
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/multitarget_validation_matrix")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=65)
    args = parser.parse_args()
    result = run_validation_matrix(
        args.output,
        duration=args.duration,
        dt=args.dt,
        random_seed=args.seed,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
