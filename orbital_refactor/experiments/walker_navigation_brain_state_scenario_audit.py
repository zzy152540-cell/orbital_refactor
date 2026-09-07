from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.walker_navigation_brain_state_audit import (
    WalkerNavigationBrainStateAudit,
    run_walker_navigation_brain_state_audit,
)
from orbital_core.constants import R_EARTH
from scenarios.walker_scenario import WalkerDeltaConfig


@dataclass(frozen=True)
class WalkerNavigationScenarioAudit:
    cases: dict[str, WalkerNavigationBrainStateAudit]
    summary: dict[str, object]


def run_walker_navigation_scenario_audit(
    *, duration=120.0, dt=5.0,
    case_specs=None,
):
    selected_specs = case_specs or _default_geometry_specs()
    cases = {
        str(name): run_walker_navigation_brain_state_audit(
            duration=duration, dt=dt, seed=int(seed),
            anchor_interval_samples=int(interval),
            walker_config=walker_config,
        )
        for name, seed, interval, walker_config in selected_specs
    }
    ranks = np.asarray([
        case.summary["effective_rank"] for case in cases.values()
    ])
    separations = np.asarray([
        case.summary["mean_node_separation_ratio"] for case in cases.values()
    ])
    pairs = {
        name: case.summary["most_correlated_feature_pair"]
        for name, case in cases.items()
    }
    return WalkerNavigationScenarioAudit(
        cases=cases,
        summary={
            "case_count": len(cases),
            "minimum_effective_rank": float(np.min(ranks)),
            "maximum_effective_rank": float(np.max(ranks)),
            "mean_effective_rank": float(np.mean(ranks)),
            "minimum_node_separation_ratio": float(np.min(separations)),
            "maximum_node_separation_ratio": float(np.max(separations)),
            "most_correlated_feature_pair_by_case": pairs,
            "all_cases_valid": bool(all(
                case.summary["valid_fraction"] == 1.0
                for case in cases.values()
            )),
        },
    )


def _default_geometry_specs():
    return (
        ("walker5_p1_f0", 0, 2, WalkerDeltaConfig(
            total_satellites=5, plane_count=1, phasing=0,
            semi_major_axis=R_EARTH + 550e3, eccentricity=0.001,
            inclination=np.deg2rad(35.0), base_true_anomaly=np.deg2rad(17.0),
        )),
        ("walker10_p2_f1", 1, 5, WalkerDeltaConfig(
            total_satellites=10, plane_count=2, phasing=1,
            semi_major_axis=R_EARTH + 700e3, eccentricity=0.005,
            inclination=np.deg2rad(53.0), raan_origin=np.deg2rad(23.0),
            argument_of_perigee=np.deg2rad(41.0),
            base_true_anomaly=np.deg2rad(71.0),
        )),
        ("walker20_p5_f3", 2, 8, WalkerDeltaConfig(
            total_satellites=20, plane_count=5, phasing=3,
            semi_major_axis=R_EARTH + 900e3, eccentricity=0.01,
            inclination=np.deg2rad(70.0), raan_origin=np.deg2rad(11.0),
            argument_of_perigee=np.deg2rad(67.0),
            base_true_anomaly=np.deg2rad(131.0),
        )),
    )


if __name__ == "__main__":
    result = run_walker_navigation_scenario_audit()
    print(result.summary)
    for name, case in result.cases.items():
        print(name, case.summary)
