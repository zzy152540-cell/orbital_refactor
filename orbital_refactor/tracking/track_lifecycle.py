from __future__ import annotations

from enum import Enum


class TrackLifecycle(str, Enum):
    """Lifecycle of one non-cooperative target track."""

    TENTATIVE = "TENTATIVE"
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    MANEUVER_SUSPECTED = "MANEUVER_SUSPECTED"
    COASTING = "COASTING"
    LOST = "LOST"
    TERMINATED = "TERMINATED"
