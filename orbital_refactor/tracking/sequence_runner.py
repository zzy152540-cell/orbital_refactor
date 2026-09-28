from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

import numpy as np

from interfaces.data_objects import ModuleInput

from .cooperative_pipeline import run_cooperative_target_fusion
from .data_contracts import MultiTargetOutput, TargetInitialState, TargetTrack
from .local_target_filter import run_local_target_history
from .track_manager import TrackManager


@dataclass(frozen=True)
class KnownTargetSequenceHistory:
    timestamps: np.ndarray
    output_by_epoch: tuple[MultiTargetOutput, ...]
    track_history_by_target: Mapping[str, tuple[TargetTrack, ...]]


def run_known_target_sequence(
    *,
    scene_id: str,
    module_inputs: Iterable[ModuleInput],
    initial_target_states: Mapping[str, TargetInitialState],
    process_noise_acceleration: float = 1e-4,
    max_coast_epochs: int = 3,
    objective: str = "trace",
    grid_points: int = 31,
) -> KnownTargetSequenceHistory:
    """Run aligned known-target local histories, target CI, and lifecycle updates."""

    inputs = list(module_inputs)
    if not inputs:
        raise ValueError("module_inputs must contain at least one local target task.")
    target_ids = tuple(initial_target_states)
    local_histories = []
    task_keys = set()
    common_timestamps = None
    for module_input in inputs:
        target_id = str(module_input.initial_state.target_id)
        if target_id not in initial_target_states:
            raise ValueError(f"Local task references unregistered target {target_id!r}.")
        runtime = module_input.config.get("runtime", {})
        observer_id = str(runtime.get("node_id", module_input.config.get("node_id", "")))
        key = (observer_id, target_id)
        if key in task_keys:
            raise ValueError("Duplicate (observer_id, target_id) local sequence task.")
        task_keys.add(key)
        timestamps = np.asarray(runtime.get("timestamps"), dtype=float).reshape(-1)
        if common_timestamps is None:
            common_timestamps = timestamps
        elif not np.array_equal(timestamps, common_timestamps):
            raise ValueError("All local sequence tasks must use identical timestamps.")
        local_histories.append(run_local_target_history(
            module_input,
            track_id=initial_target_states[target_id].track_id,
        ))

    assert common_timestamps is not None
    manager = TrackManager(
        scene_id=scene_id,
        initial_states=initial_target_states,
        process_noise_acceleration=process_noise_acceleration,
        max_coast_epochs=max_coast_epochs,
    )
    outputs = []
    track_history = {target_id: [] for target_id in target_ids}
    for index, timestamp in enumerate(common_timestamps):
        output = run_cooperative_target_fusion(
            scene_id=scene_id,
            timestamp=float(timestamp),
            reports=[history[index] for history in local_histories],
            expected_target_ids=target_ids,
            objective=objective,
            grid_points=grid_points,
        )
        tracks = manager.step(output)
        outputs.append(output)
        for target_id in target_ids:
            track_history[target_id].append(tracks[target_id])
    timestamps = common_timestamps.copy()
    timestamps.setflags(write=False)
    return KnownTargetSequenceHistory(
        timestamps=timestamps,
        output_by_epoch=tuple(outputs),
        track_history_by_target={
            target_id: tuple(values)
            for target_id, values in track_history.items()
        },
    )
