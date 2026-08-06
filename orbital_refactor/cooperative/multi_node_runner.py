from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from interfaces.data_objects import ModuleInput
from interfaces.state_awareness_module import StateAwarenessModule


@dataclass(frozen=True)
class MultiNodeRunResult:
    histories: dict[str, Any]

    @property
    def node_ids(self) -> list[str]:
        return list(self.histories)


def run_multi_node_histories(
    module_inputs: dict[str, ModuleInput],
    *,
    module_factory=StateAwarenessModule,
) -> MultiNodeRunResult:
    if not module_inputs:
        raise ValueError("module_inputs cannot be empty.")
    histories: dict[str, Any] = {}
    for node_id, module_input in module_inputs.items():
        configured_id = str(module_input.config.get("runtime", {}).get("node_id", node_id))
        if configured_id != node_id:
            raise ValueError(f"Node key {node_id!r} does not match configured node_id {configured_id!r}.")
        histories[node_id] = module_factory().run_history(module_input)
    return MultiNodeRunResult(histories=histories)


def extract_fused_local_history(history: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(history, "fused_state_history"):
        return history.fused_state_history, history.fused_covariance_history
    if hasattr(history, "state_history"):
        return history.state_history, history.covariance_history
    raise TypeError("Unsupported history object.")
