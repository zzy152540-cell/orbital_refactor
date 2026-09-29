"""Target-centric contracts and fusion helpers for cooperative tracking."""

from .data_contracts import (
    ECI_FRAME,
    GlobalTargetEstimate,
    LocalTargetEstimate,
    MultiTargetInput,
    MultiTargetOutput,
    ObserverState,
    TargetInitialState,
    TargetNodeReport,
    TargetTrack,
    TargetTrackKey,
)
from .target_fusion import fuse_target_reports, group_reports_by_target
from .cooperative_pipeline import run_cooperative_target_fusion
from .local_target_filter import run_local_target_filter, run_local_target_history
from .known_target_pipeline import run_known_target_batch
from .track_manager import TrackManager
from .sequence_runner import KnownTargetSequenceHistory, run_known_target_sequence
from .track_lifecycle import TrackLifecycle
from .standard_validation import (
    StandardMultiTargetScenario,
    StandardValidationResult,
    TargetValidationMetrics,
    build_standard_multitarget_scenario,
    build_standard_multitarget_scenario_from_input,
    run_standard_multitarget_validation,
)
from .external_scene_adapter import (
    adapt_external_multitarget_input,
    adapt_external_scene_to_multitarget,
    load_external_multitarget_input,
)
from .initial_orbit import (
    InitialOrbitEstimate,
    IODObservation,
    IODStatus,
    initialize_target,
    iod_observations_from_messages,
    tracking_observation_from_message,
    IODBufferConfig,
    MultiTargetIODManager,
    MultiTargetIODUpdate,
    TargetIODObservationBuffer,
    IODTrackingHandoff,
    build_tracking_handoff,
    target_initial_state_from_iod,
)

__all__ = [
    "ECI_FRAME",
    "GlobalTargetEstimate",
    "LocalTargetEstimate",
    "MultiTargetInput",
    "MultiTargetOutput",
    "ObserverState",
    "TargetInitialState",
    "TargetNodeReport",
    "TargetTrack",
    "TargetTrackKey",
    "TrackLifecycle",
    "fuse_target_reports",
    "group_reports_by_target",
    "run_cooperative_target_fusion",
    "run_local_target_filter",
    "run_local_target_history",
    "run_known_target_batch",
    "TrackManager",
    "KnownTargetSequenceHistory",
    "run_known_target_sequence",
    "StandardMultiTargetScenario",
    "StandardValidationResult",
    "TargetValidationMetrics",
    "build_standard_multitarget_scenario",
    "build_standard_multitarget_scenario_from_input",
    "run_standard_multitarget_validation",
    "adapt_external_multitarget_input",
    "adapt_external_scene_to_multitarget",
    "load_external_multitarget_input",
    "InitialOrbitEstimate",
    "IODObservation",
    "IODStatus",
    "initialize_target",
    "iod_observations_from_messages",
    "tracking_observation_from_message",
    "target_initial_state_from_iod",
    "IODBufferConfig",
    "MultiTargetIODManager",
    "MultiTargetIODUpdate",
    "TargetIODObservationBuffer",
    "IODTrackingHandoff",
    "build_tracking_handoff",
]
