from __future__ import annotations

from dataclasses import replace

from adapters.module_input_adapter import adapt_module_input, adapt_module_input_centralized
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import run_orbital_phase_sidecar
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
        return _attach_passive_cann_sidecar(history, adapted, module_input.config)

    def run(self, module_input: ModuleInput) -> ModuleOutput:
        history = self.run_history(module_input)
        runtime = module_input.config.get("runtime", {})
        node_id = str(runtime.get("node_id", module_input.config.get("node_id", "node_0")))
        target_id = str(module_input.initial_state.target_id)
        return history.to_module_output(node_id=node_id, target_id=target_id)


def _attach_passive_cann_sidecar(history, adapted, config):
    """Optionally observe the fused target orbit without filter feedback."""

    brain_config = config.get("brain_inspired", {})
    cann_config = brain_config.get("cann", {}) if isinstance(brain_config, dict) else {}
    if not isinstance(cann_config, dict) or not bool(cann_config.get("enabled", False)):
        return history
    relative_history = getattr(history, "fused_state_history", None)
    if relative_history is None:
        relative_history = history.state_history
    target_history_eci = adapted.chief_state_history_eci + relative_history
    frame = OrbitalPlaneFrame.from_state_eci(target_history_eci[0])
    cue_interval = cann_config.get("cue_interval_samples", None)
    if cue_interval is not None:
        cue_interval = int(cue_interval)
    sidecar = run_orbital_phase_sidecar(
        timestamps=adapted.timestamps,
        state_history_eci=target_history_eci,
        frame=frame,
        cue_interval_samples=cue_interval,
        source_id=f"{adapted.node_id}:{adapted.target_id}:fused",
    )
    return replace(history, cann_sidecar_history=sidecar)
