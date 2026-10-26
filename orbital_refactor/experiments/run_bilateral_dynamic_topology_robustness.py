"""Supplemental R06/R08 audit on the connected dynamic-topology geometry."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    derive_bilateral_fault_frontend,
    run_bilateral_cann_acceptance_frontend,
)
from experiments.run_bilateral_dynamic_topology_prescan import (
    build_dynamic_topology_schedule,
)
from orbital_core.constants import R_EARTH
from scenarios.bilateral_multitarget import generate_bilateral_walker_truth
from scenarios.walker_scenario import WalkerDeltaConfig


R08_FIXED_INDICES = {
    "SPATIAL": (0, 5, 10),
    "ADJACENT": (0, 1, 2),
    "HIGH_VISIBILITY": (0, 12, 14),
}
R08_RANDOM_INDICES = {
    0: (7, 8, 11),
    1: (6, 7, 11),
    2: (1, 3, 10),
    3: (1, 2, 10),
    4: (9, 13, 14),
}


def _truth(duration: float, dt: float):
    return generate_bilateral_walker_truth(
        duration_seconds=duration,
        sample_interval_seconds=dt,
        friendly_config=WalkerDeltaConfig(
            total_satellites=15,
            plane_count=15,
            phasing=1,
            semi_major_axis=R_EARTH + 700e3,
            eccentricity=0.0,
            inclination=np.deg2rad(53.0),
            node_prefix="friendly",
        ),
        scene_id="bilateral-dynamic-topology-candidate-15x15",
    )


def _base_config(*, seed, duration, dt, start, end, radar_backup=False):
    return BilateralFrontendConfig(
        duration_seconds=float(duration),
        sample_interval_seconds=float(dt),
        random_seed=int(seed),
        use_local_unified_multimodal_measurement_update=True,
        backup_radial_range_sigma_m=(1.0 if radar_backup else None),
        backup_radial_range_rate_sigma_mps=(0.0125 if radar_backup else None),
        dropout_modality="OPTICAL",
        dropout_start_seconds=float(start),
        dropout_end_seconds=float(end),
        dropout_target_indices=(),
    )


def _run_arm(frontend, *, cache, topology=None):
    return run_bilateral_cann_acceptance_frontend(
        frontend,
        enable_consensus=topology is not None,
        topology_by_timestamp=topology,
        iod_initialization_cache=cache,
    )


def run_seed(*, seed: int, duration: float, dt: float, start: float, end: float):
    truth = _truth(duration, dt)
    topology, topology_summary = build_dynamic_topology_schedule(truth)
    cache = {}

    nominal_config = _base_config(
        seed=seed, duration=duration, dt=dt, start=start, end=end,
    )
    nominal_frontend = build_bilateral_raw_frontend(nominal_config, truth=truth)
    nominal_control = _run_arm(nominal_frontend, cache=cache)
    r06 = {}
    for modality in ("OPTICAL", "INFRARED"):
        fault_config = replace(
            nominal_config,
            dropout_modality=modality,
            dropout_target_indices=None,
        )
        affected = _run_arm(
            derive_bilateral_fault_frontend(nominal_frontend, fault_config),
            cache=cache,
        )
        r06[modality] = _r06_record(nominal_control, affected)

    radar_config = _base_config(
        seed=seed, duration=duration, dt=dt, start=start, end=end,
        radar_backup=True,
    )
    radar_frontend = build_bilateral_raw_frontend(radar_config, truth=truth)
    radar_control = _run_arm(radar_frontend, cache=cache)
    radar_fault_config = replace(
        radar_config, dropout_modality="RADAR", dropout_target_indices=None,
    )
    radar_affected = _run_arm(
        derive_bilateral_fault_frontend(radar_frontend, radar_fault_config),
        cache=cache,
    )
    r06["RADAR"] = _r06_record(radar_control, radar_affected)

    failure_control_config = BilateralFrontendConfig(
        duration_seconds=float(duration),
        sample_interval_seconds=float(dt),
        random_seed=int(seed),
        use_local_unified_multimodal_measurement_update=True,
        observer_failure_start_seconds=float(start),
        observer_failure_end_seconds=float(end),
        failed_observer_indices=(),
    )
    failure_control_frontend = build_bilateral_raw_frontend(
        failure_control_config, truth=truth,
    )
    failure_control = _run_arm(
        failure_control_frontend, cache=cache, topology=topology,
    )
    combinations = dict(R08_FIXED_INDICES)
    combinations["FROZEN_RANDOM"] = R08_RANDOM_INDICES[int(seed)]
    r08 = {}
    for name, indices in combinations.items():
        failed_ids = tuple(
            truth.manifest.friendly_observer_ids[index]
            for index in indices
        )
        fault_topology, fault_topology_summary = (
            build_dynamic_topology_schedule(
                truth,
                inactive_node_ids_by_timestamp={
                    float(timestamp): failed_ids
                    for timestamp in truth.timestamps
                    if start <= float(timestamp) <= end
                },
            )
        )
        fault_config = replace(
            failure_control_config, failed_observer_indices=tuple(indices),
        )
        affected = _run_arm(
            derive_bilateral_fault_frontend(
                failure_control_frontend, fault_config,
            ),
            cache=cache,
            topology=fault_topology,
        )
        r08[name] = _r08_record(
            failure_control, affected, failed_indices=indices,
        )
        r08[name]["topology"] = fault_topology_summary
    return {
        "seed": int(seed),
        "topology": topology_summary,
        "r06": r06,
        "r08": r08,
    }


def _r06_record(control, affected):
    reference = control.phase_metrics["DROPOUT"]
    experiment = affected.phase_metrics["DROPOUT"]
    increases = {
        "position_rmse_m": _percent_change(
            experiment.position_rmse_m, reference.position_rmse_m,
        ),
        "velocity_rmse_mps": _percent_change(
            experiment.velocity_rmse_mps, reference.velocity_rmse_mps,
        ),
        "acceleration_rmse_mps2": _percent_change(
            experiment.acceleration_rmse_mps2,
            reference.acceleration_rmse_mps2,
        ),
    }
    coverage = bool(
        control.final_active_track_count == 15
        and affected.final_active_track_count == 15
        and len(control.position_rmse_m_by_target) == 15
        and len(affected.position_rmse_m_by_target) == 15
        and reference.position_sample_count == experiment.position_sample_count
        and reference.position_sample_count > 0
    )
    return {
        "thresholdPercent": 15.0,
        "primaryMetric": "position_rmse_m",
        "coverageComparable": coverage,
        "passed": bool(
            coverage and increases["position_rmse_m"] <= 15.0
        ),
        "increasePercent": increases,
        "control": asdict(reference),
        "affected": asdict(experiment),
        "controlArm": _arm_summary(control),
        "affectedArm": _arm_summary(affected),
    }


def _r08_record(control, affected, *, failed_indices):
    reference = control.consensus_phase_metrics["DROPOUT"]
    experiment = affected.consensus_phase_metrics["DROPOUT"]
    increases = {
        "position_mean_error_m": _percent_change(
            experiment.position_mean_error_m,
            reference.position_mean_error_m,
        ),
        "velocity_mean_error_mps": _percent_change(
            experiment.velocity_mean_error_mps,
            reference.velocity_mean_error_mps,
        ),
        "acceleration_mean_error_mps2": _percent_change(
            experiment.acceleration_mean_error_mps2,
            reference.acceleration_mean_error_mps2,
        ),
    }
    coverage = bool(
        reference.position_sample_count > 0
        and experiment.position_sample_count > 0
        and len(control.consensus_position_rmse_m_by_target) == 15
        and len(affected.consensus_position_rmse_m_by_target) == 15
    )
    return {
        "failedObserverIndices": list(failed_indices),
        "thresholdPercent": 12.0,
        "primaryMetric": "position_mean_error_m",
        "coverageComparable": coverage,
        "passed": bool(
            coverage and increases["position_mean_error_m"] <= 12.0
        ),
        "increasePercent": increases,
        "control": asdict(reference),
        "affected": asdict(experiment),
        "controlArm": _arm_summary(control),
        "affectedArm": _arm_summary(affected),
    }


def _arm_summary(value):
    return {
        "finalActiveTrackCount": value.final_active_track_count,
        "identitySwitchCount": value.identity_switch_count,
        "consensusIdentitySwitchCount": value.consensus_identity_switch_count,
        "missingTruthSampleCount": value.missing_truth_sample_count,
        "consensusMissingTruthSampleCount": (
            value.consensus_missing_truth_sample_count
        ),
        "targetCoverageCount": len(value.position_rmse_m_by_target),
        "consensusTargetCoverageCount": len(
            value.consensus_position_rmse_m_by_target
        ),
    }


def _percent_change(value, baseline):
    value = float(value)
    baseline = float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return None
    return 100.0 * (value / baseline - 1.0)


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=6.0)
    parser.add_argument("--end", type=float, default=13.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "bilateral_dynamic_topology_robustness_prescan.json"
        ),
    )
    args = parser.parse_args()
    print(f"running supplemental robustness seed {args.seed}", flush=True)
    record = run_seed(
        seed=args.seed, duration=args.duration, dt=args.dt,
        start=args.start, end=args.end,
    )
    payload = {
        "schemaVersion": "bilateral-dynamic-topology-robustness.v1",
        "role": "supplemental_only",
        "affectsFrozenAcceptanceConfiguration": False,
        "scenario": {
            "durationSeconds": args.duration,
            "sampleIntervalSeconds": args.dt,
            "faultStartSeconds": args.start,
            "faultEndSeconds": args.end,
            "seed": args.seed,
            "friendlyWalker": {"satellites": 15, "planes": 15, "phasing": 1},
            "hostileSatellites": 15,
        },
        "record": record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "r06": {
            name: {"passed": item["passed"], "increasePercent": item[
                "increasePercent"
            ]["position_rmse_m"]}
            for name, item in record["r06"].items()
        },
        "r08": {
            name: {"passed": item["passed"], "increasePercent": item[
                "increasePercent"
            ]["position_mean_error_m"]}
            for name, item in record["r08"].items()
        },
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
