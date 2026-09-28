from __future__ import annotations

from collections.abc import Iterable, Mapping

from interfaces.data_objects import ModuleInput

from .cooperative_pipeline import run_cooperative_target_fusion
from .data_contracts import MultiTargetOutput
from .local_target_filter import run_local_target_filter


def run_known_target_batch(
    *,
    scene_id: str,
    module_inputs: Iterable[ModuleInput],
    track_id_by_target: Mapping[str, str] | None = None,
    expected_target_ids: Iterable[str] | None = None,
    objective: str = "trace",
    grid_points: int = 31,
) -> MultiTargetOutput:
    """Run known-identity local tracks and target-level CI as one batch."""

    inputs = list(module_inputs)
    if not inputs:
        raise ValueError("module_inputs must contain at least one local target task.")
    track_ids = dict(track_id_by_target or {})
    reports = []
    task_keys = set()
    for module_input in inputs:
        target_id = str(module_input.initial_state.target_id)
        runtime = module_input.config.get("runtime", {})
        observer_id = str(
            runtime.get("node_id", module_input.config.get("node_id", ""))
        )
        task_key = (observer_id, target_id)
        if task_key in task_keys:
            raise ValueError(
                "Only one local ModuleInput is allowed per (observer_id, target_id) task."
            )
        task_keys.add(task_key)
        reports.append(
            run_local_target_filter(
                module_input,
                track_id=track_ids.get(target_id, target_id),
            )
        )
    timestamps = {report.timestamp for report in reports}
    if len(timestamps) != 1:
        raise ValueError("All local target tasks must finish at the same fusion timestamp.")
    return run_cooperative_target_fusion(
        scene_id=scene_id,
        timestamp=next(iter(timestamps)),
        reports=reports,
        expected_target_ids=expected_target_ids,
        objective=objective,
        grid_points=grid_points,
    )
