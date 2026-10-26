"""Supplemental P05/P07 audit on the physically connected Walker geometry.

The frozen acceptance scenario remains unchanged.  This runner preserves its
sensor, estimator, duration, sampling, seed and metric definitions while
replacing only the friendly Walker geometry and, for P07, the explicitly
supplied low-churn physical communication topology.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    run_bilateral_baseline_matrix,
)
from experiments.run_bilateral_dynamic_topology_prescan import (
    build_dynamic_topology_schedule,
)
from orbital_core.constants import R_EARTH
from scenarios.bilateral_multitarget import generate_bilateral_walker_truth
from scenarios.walker_scenario import WalkerDeltaConfig


def run_seed(*, seed: int, duration: float, dt: float) -> dict:
    config = BilateralFrontendConfig(
        duration_seconds=float(duration),
        sample_interval_seconds=float(dt),
        random_seed=int(seed),
    )
    truth = generate_bilateral_walker_truth(
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
    frontend = build_bilateral_raw_frontend(config, truth=truth)
    topology_schedule, topology_summary = build_dynamic_topology_schedule(
        truth, maximum_range_km=7000.0, maximum_degree=3,
    )
    p05_result = run_bilateral_baseline_matrix(
        config,
        acceptance_scope="p05-candidate",
        frontend=frontend,
    )
    p07_result = run_bilateral_baseline_matrix(
        config,
        acceptance_scope="p07-candidate",
        frontend=frontend,
        topology_by_timestamp=topology_schedule,
    )
    p05 = p05_result.matched_single_observer_joint_cann_metrics
    p07 = p07_result.matched_joint_independent_consensus_metrics
    if p05 is None or p07 is None:
        raise RuntimeError("Required matched acceptance metrics were not produced.")
    return {
        "seed": int(seed),
        "topology": topology_summary,
        "p05": {
            "thresholdPercent": 5.0,
            "metric": "matched position RMSE improvement",
            "valuePercent": p05.position_rmse_improvement_percent,
            "passed": bool(p05.position_rmse_improvement_percent >= 5.0),
            "matched": asdict(p05),
        },
        "p07": {
            "thresholdPercent": 6.0,
            "metric": "matched position mean-error improvement",
            "valuePercent": p07.position_mean_error_improvement_percent,
            "passed": bool(p07.position_mean_error_improvement_percent >= 6.0),
            "matched": asdict(p07),
            "activeEdgeCountByTimestamp": dict(
                p07_result.arms[
                    "fixed_topology_joint_multimodal_cann"
                ].active_communication_edge_count_by_timestamp
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/external_acceptance/"
            "bilateral_dynamic_topology_acceptance_prescan.json"
        ),
    )
    args = parser.parse_args()
    records = []
    for seed in args.seeds:
        print(f"running supplemental dynamic-topology seed {seed}", flush=True)
        records.append(run_seed(
            seed=seed, duration=args.duration, dt=args.dt,
        ))
    payload = {
        "schemaVersion": "bilateral-dynamic-topology-acceptance.v1",
        "role": "supplemental_only",
        "affectsFrozenAcceptanceConfiguration": False,
        "scenario": {
            "friendlyWalker": {
                "satellites": 15, "planes": 15, "phasing": 1,
                "altitudeKm": 700.0, "inclinationDeg": 53.0,
            },
            "hostileWalker": "unchanged frozen 15-satellite Walker",
            "durationSeconds": float(args.duration),
            "sampleIntervalSeconds": float(args.dt),
            "seeds": list(args.seeds),
            "communication": {
                "maximumRangeKm": 7000.0,
                "earthOccultation": True,
                "policy": "LowChurnConnectedTreePolicy",
                "maximumDegree": 3,
            },
        },
        "records": records,
        "summary": {
            "p05AllSeedsPass": all(row["p05"]["passed"] for row in records),
            "p07AllSeedsPass": all(row["p07"]["passed"] for row in records),
            "p05MeanImprovementPercent": float(np.mean([
                row["p05"]["valuePercent"] for row in records
            ])),
            "p07MeanImprovementPercent": float(np.mean([
                row["p07"]["valuePercent"] for row in records
            ])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
