"""Target-centric contracts and fusion helpers for cooperative tracking."""

from .data_contracts import (
    ECI_FRAME,
    MAX_INFORMATION_IDS,
    GlobalTargetEstimate,
    LocalTargetEstimate,
    MultiTargetInput,
    MultiTargetOutput,
    ObserverState,
    TargetInitialState,
    TargetNodeReport,
    TargetEstimateMessage,
    TargetTrack,
    TargetTrackKey,
)
from .target_fusion import fuse_target_reports, group_reports_by_target
from .cooperative_pipeline import run_cooperative_target_fusion
from .distributed_target_consensus import (
    DistributedTargetConsensus,
    DistributedTargetConsensusStep,
)
from .distributed_online_tracker import (
    DistributedOnlineMultiTargetTracker,
    DistributedOnlineTrackingUpdate,
)
from .distributed_node_network import (
    DistributedNodeNetworkUpdate,
    DistributedTargetNodeNetwork,
)
from .distributed_identity import (
    CooperativeTrackIdentityResolver,
    TargetIdentityAssignment,
    TargetIdentityResolution,
)
from .local_target_filter import run_local_target_filter, run_local_target_history
from .known_target_pipeline import run_known_target_batch
from .track_manager import TrackManager
from .maneuver_detection import (
    ManeuverAssessment,
    ManeuverDetectionConfig,
    ManeuverDetector,
)
from .duplicate_tracks import (
    DuplicateResolution,
    DuplicateTrackConfig,
    DuplicateTrackResolver,
)
from .online_json import (
    PROTOCOL_VERSION as ONLINE_PROTOCOL_VERSION,
    parse_online_json_frame,
    run_online_json_replay,
    run_online_json_replay_file,
)
from .association import (
    AssociationMatch,
    AssociationPipelineUpdate,
    AssociationResult,
    AutonomousIODManager,
    AutonomousTrackingSequenceResult,
    ConsensusFeedbackApplication,
    MultiTargetAssociationPipeline,
    OnlineMultiTargetTracker,
    OnlineTrackingUpdate,
    UnlabeledIODObservation,
    UnlabeledObservationMessage,
    associate_to_tracks,
    label_matches,
    unlabeled_iod_observations_from_messages,
    run_unlabeled_tracking_sequence,
)
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
    "MAX_INFORMATION_IDS",
    "GlobalTargetEstimate",
    "LocalTargetEstimate",
    "MultiTargetInput",
    "MultiTargetOutput",
    "ObserverState",
    "TargetInitialState",
    "TargetNodeReport",
    "TargetEstimateMessage",
    "TargetTrack",
    "TargetTrackKey",
    "TrackLifecycle",
    "fuse_target_reports",
    "group_reports_by_target",
    "run_cooperative_target_fusion",
    "DistributedTargetConsensus",
    "DistributedTargetConsensusStep",
    "DistributedOnlineMultiTargetTracker",
    "DistributedOnlineTrackingUpdate",
    "DistributedNodeNetworkUpdate",
    "DistributedTargetNodeNetwork",
    "CooperativeTrackIdentityResolver",
    "TargetIdentityAssignment",
    "TargetIdentityResolution",
    "run_local_target_filter",
    "run_local_target_history",
    "run_known_target_batch",
    "TrackManager",
    "ManeuverAssessment",
    "ManeuverDetectionConfig",
    "ManeuverDetector",
    "DuplicateResolution",
    "DuplicateTrackConfig",
    "DuplicateTrackResolver",
    "ONLINE_PROTOCOL_VERSION",
    "parse_online_json_frame",
    "run_online_json_replay",
    "run_online_json_replay_file",
    "AssociationMatch",
    "AssociationPipelineUpdate",
    "AssociationResult",
    "AutonomousIODManager",
    "AutonomousTrackingSequenceResult",
    "ConsensusFeedbackApplication",
    "MultiTargetAssociationPipeline",
    "OnlineMultiTargetTracker",
    "OnlineTrackingUpdate",
    "UnlabeledIODObservation",
    "UnlabeledObservationMessage",
    "associate_to_tracks",
    "label_matches",
    "unlabeled_iod_observations_from_messages",
    "run_unlabeled_tracking_sequence",
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
