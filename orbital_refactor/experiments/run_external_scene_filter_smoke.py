"""Exercise the full multimodal filter route from an external scene JSON.

The generated state histories are a deterministic two-body+J2 integration
fixture.  They validate interface and pipeline compatibility; they are not an
independent orbit-accuracy reference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from adapters.external_scene_input import adapt_external_scene_input
from experiments.run_project_31sat_filter_smoke import run_filter_smoke
from orbital_core.dynamics import propagate_absolute_orbit


def run_external_scene_filter_smoke(
    scene_path, output_directory, *, duration=20, epoch_transfer_step=120.0,
):
    payload = json.loads(Path(scene_path).read_text(encoding="utf-8-sig"))
    scene = adapt_external_scene_input(
        payload, propagation_step_seconds=float(epoch_transfer_step),
    )
    count = int(duration) + 1
    times = scene.timestamps[:count]
    histories = {
        node: propagate_absolute_orbit(initial, times)
        for node, initial in scene.initial_state_by_node.items()
    }
    nodes = tuple(histories)
    if len(nodes) < 2:
        raise ValueError("At least two satellites are required for relative filtering.")
    functional_count = min(6, len(nodes))
    background_nodes = nodes[:-functional_count]
    functional_nodes = nodes[-functional_count:]
    if not background_nodes:
        raise ValueError("At least one background observer is required.")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    start = datetime.fromtimestamp(scene.start_time_ms / 1000.0, tz=timezone.utc)
    background_payload = {
        "sceneId": scene.scene_id,
        "satellites": [
            {
                "name": node,
                "norad": next(item.norad_id for item in scene.satellites if item.node_id == node),
                "points": [
                    _state_point(start + timedelta(seconds=float(times[index])), state)
                    for index, state in enumerate(histories[node])
                ],
            }
            for node in background_nodes
        ],
    }
    functional_payload = [
        {
            "run_id": scene.scene_id,
            "time": _format_time(start + timedelta(seconds=float(times[index]))),
            "satelliteList": [
                {
                    "satObj": node,
                    "satId": next(item.norad_id for item in scene.satellites if item.node_id == node),
                    "position": _state_mapping(histories[node][index]),
                }
                for node in functional_nodes
            ],
        }
        for index in range(count)
    ]
    background_path = output / "synthetic_background_truth.json"
    functional_path = output / "synthetic_functional_truth.json"
    background_path.write_text(
        json.dumps(background_payload, ensure_ascii=False), encoding="utf-8",
    )
    functional_path.write_text(
        json.dumps(functional_payload, ensure_ascii=False), encoding="utf-8",
    )
    filter_summary = run_filter_smoke(
        background_path, functional_path, output / "filter",
        duration=int(duration), target_scope="nonmaneuver_all",
    )
    result = {
        "sceneId": scene.scene_id,
        "inputSatelliteCount": len(nodes),
        "durationSeconds": int(duration),
        "sceneStartTimeMs": scene.start_time_ms,
        "sceneEndTimeMs": scene.end_time_ms,
        "orbitEpochAfterSceneStartCount": sum(
            item.epoch_ms > scene.start_time_ms for item in scene.satellites
        ),
        "truthSource": "same_model_two_body_j2_fixture",
        "accuracyClaimAllowed": False,
        "filterSummary": filter_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def _state_mapping(state):
    return dict(zip(("x", "y", "z", "vx", "vy", "vz"), map(float, state)))


def _state_point(time, state):
    return {"time": _format_time(time), **_state_mapping(state)}


def _format_time(value):
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--epoch-transfer-step", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(run_external_scene_filter_smoke(
        args.scene, args.output, duration=args.duration,
        epoch_transfer_step=args.epoch_transfer_step,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
