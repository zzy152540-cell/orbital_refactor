from __future__ import annotations

from adapters.module_input_adapter import adapt_module_input, adapt_module_input_centralized
from interfaces.data_objects import ModuleInput, ModuleOutput
from pipelines.centralized import run_centralized_filter
from pipelines.federated_ci import run_federated_ci_filter


class StateAwarenessModule:
    """Stable external entry point matching the software-interface document.

    Select the internal single-node algorithm with ``config['filter']['architecture']``:
    ``'federated_ci'`` (default) or ``'centralized'``. The external input and output
    objects remain unchanged.
    """

    def run_history(self, module_input: ModuleInput):
        filter_config = module_input.config.get("filter", {})
        architecture = str(filter_config.get("architecture", "federated_ci")).lower()
        if architecture in {"centralized", "centralized_ekf"}:
            adapted = adapt_module_input_centralized(module_input)
            history = run_centralized_filter(
                timestamps=adapted.timestamps,
                chief_state_history_eci=adapted.chief_state_history_eci,
                q_eci2pri_history=adapted.q_eci2pri_history,
                measurements_by_modality=adapted.measurements_by_modality,
                valid_flags_by_modality=adapted.valid_flags_by_modality,
                ekf=adapted.centralized_filter,
                initial_state=adapted.initial_state,
                initial_covariance=adapted.initial_covariance,
                node_id=adapted.node_id,
                target_id=adapted.target_id,
            )
        elif architecture in {"federated", "federated_ci", "ci"}:
            adapted = adapt_module_input(module_input)
            history = run_federated_ci_filter(
                timestamps=adapted.timestamps,
                chief_state_history_eci=adapted.chief_state_history_eci,
                q_eci2pri_history=adapted.q_eci2pri_history,
                measurements_by_modality=adapted.measurements_by_modality,
                valid_flags_by_modality=adapted.valid_flags_by_modality,
                local_filters=adapted.local_filters,
                initial_state=adapted.initial_state,
                initial_covariance=adapted.initial_covariance,
                reset_feedback=adapted.reset_feedback,
                ci_objective=adapted.ci_objective,
                ci_grid_points=adapted.ci_grid_points,
                node_id=adapted.node_id,
                target_id=adapted.target_id,
            )
        else:
            raise ValueError(f"Unsupported filter architecture: {architecture}")
        return history

    def run(self, module_input: ModuleInput) -> ModuleOutput:
        history = self.run_history(module_input)
        runtime = module_input.config.get("runtime", {})
        node_id = str(runtime.get("node_id", module_input.config.get("node_id", "node_0")))
        target_id = str(module_input.initial_state.target_id)
        return history.to_module_output(node_id=node_id, target_id=target_id)
