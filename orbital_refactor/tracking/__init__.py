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
]
