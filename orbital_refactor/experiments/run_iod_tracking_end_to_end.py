"""Run raw multimodal IOD followed by cooperative continuous tracking."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from adapters.multimodal_sensor_simulator import simulate_multimodal_sensor_history
from experiments.run_multitarget_iod_smoke import _sensors, _state, _tracking_camera_basis
from orbital_core.coordinates import build_rtn_quaternion_history, dcm_to_quat_wxyz
from orbital_core.dynamics import propagate_absolute_orbit
from tracking import (
    IODBufferConfig, MultiTargetIODManager, build_tracking_handoff,
    iod_observations_from_messages, run_known_target_sequence,
    tracking_observation_from_message,
)


def run_iod_tracking_end_to_end(output_directory, *, random_seed=0, image_size=33):
    timestamps = np.arange(0.0, 51.0, 10.0)
    iod_times, tracking_times = timestamps[:3], timestamps[3:]
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
    messages_by_epoch = {float(timestamp): [] for timestamp in timestamps}
    state_by_epoch = {}
    for target_index, (target_id, target_history) in enumerate(target_histories.items()):
        for observer_index, (observer_id, observer_history) in enumerate(observer_histories.items()):
            attitude = np.asarray([
                dcm_to_quat_wxyz(_tracking_camera_basis(
                    target_history[index, :3] - observer_history[index, :3]
                ).T)
                for index in range(len(timestamps))
            ])
            raw = simulate_multimodal_sensor_history(
                timestamps=timestamps, observer_id=observer_id, target_id=target_id,
                observer_state_history=observer_history, target_state_history=target_history,
                quaternion_i2b_wxyz_history_by_modality={
                    "OPTICAL": attitude, "INFRARED": attitude,
                },
                config=_sensors(image_size),
                random_seed=random_seed + 1000 * target_index + 10 * observer_index,
            )
            for message in raw.messages:
                messages_by_epoch[float(message.timestamp)].append(message)
            for index, timestamp in enumerate(timestamps):
                state_by_epoch[(observer_id, float(timestamp))] = observer_history[index]
    for index, message in enumerate(messages_by_epoch[10.0]):
        if message.modality == "RADAR" and message.target_id == "target-1":
            messages_by_epoch[10.0][index] = replace(
                message, measurement=np.asarray(message.measurement) + [2000.0, 2.0],
                metadata={**message.metadata, "injectedOutlier": True},
            )
            break
    manager = MultiTargetIODManager(
        track_id_by_target={key: f"track-{key}" for key in targets},
        buffer_config=IODBufferConfig(
            minimum_epochs=3, minimum_observers=2, minimum_time_span_seconds=20.0,
        ),
    )
    for timestamp in iod_times:
        manager.ingest(iod_observations_from_messages(
            messages_by_epoch[float(timestamp)], observer_state_by_epoch=state_by_epoch,
        ))
    observations_by_link = {
        (observer_id, target_id): []
        for observer_id in observers for target_id in targets
    }
    observer_tracking = {key: value[3:] for key, value in observer_histories.items()}
    q_histories = {
        key: build_rtn_quaternion_history(value)
        for key, value in observer_tracking.items()
    }
    for epoch_index, timestamp in enumerate(tracking_times):
        for message in messages_by_epoch[float(timestamp)]:
            observations_by_link[(message.observer_id, message.target_id)].append(
                tracking_observation_from_message(
                    message, q_eci2pri=q_histories[message.observer_id][epoch_index],
                )
            )
    handoff = build_tracking_handoff(
        initialized_states=manager.initialized_states, timestamps=tracking_times,
        observer_state_history_by_id=observer_tracking,
        q_eci2pri_history_by_id=q_histories,
        observations_by_link=observations_by_link,
    )
    history = run_known_target_sequence(
        scene_id="iod-tracking-end-to-end", module_inputs=handoff.module_inputs,
        initial_target_states=handoff.target_initial_states,
    )
    results = {}
    for target_id, tracks in history.track_history_by_target.items():
        truth = target_histories[target_id][3:]
        estimate = np.asarray([track.estimate.state_eci for track in tracks])
        error = estimate - truth
        results[target_id] = {
            "lifecycle": [track.lifecycle.value for track in tracks],
            "positionRmseM": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            "velocityRmseMps": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        }
    summary = {
        "scenario": "raw IOD to cooperative continuous tracking",
        "iodEpochs": iod_times.tolist(), "trackingEpochs": tracking_times.tolist(),
        "initializedTargetIds": sorted(manager.initialized_states),
        "localFilterTaskCount": len(handoff.module_inputs), "resultsByTarget": results,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/iod_tracking_end_to_end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=33)
    args = parser.parse_args()
    print(json.dumps(run_iod_tracking_end_to_end(
        args.output, random_seed=args.seed, image_size=args.image_size,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
