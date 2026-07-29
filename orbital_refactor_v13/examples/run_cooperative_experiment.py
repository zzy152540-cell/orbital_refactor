"""Unified cooperative experiment entry."""

from __future__ import annotations
import sys
from pathlib import Path
import json

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_multi_sat_interface import build_demo_case
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from cooperative.delay_channel import DelayChannel
from exporters.cooperative_result_exporter import save_cooperative_summary

def main():
    scenario, observations, initial_errors, modality_config = build_demo_case()
    config=json.loads((PROJECT_ROOT/"configs"/"cooperative_config.json").read_text())

    delay=DelayChannel(
        delay_by_node=config["communication"]["delay"]["values"]
    )

    result=run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node=initial_errors,
        architecture="federated_ci",
        modality_config_by_node=modality_config,
        delay_channel=delay,
        age_aware=config["fusion"]["age_aware"],
        age_penalty=config["fusion"]["age_penalty"],
    )

    print("Experiment:", config["experiment"]["name"])
    print("Position RMSE:", result.metrics.cooperative_position_rmse)
    print("Velocity RMSE:", result.metrics.cooperative_velocity_rmse)

    save_cooperative_summary(result)
    print("Saved results/cooperative")

if __name__=="__main__":
    main()
