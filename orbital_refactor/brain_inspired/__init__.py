"""Brain-inspired state representations that remain outside the filters."""

from .passive_phase_observer import (
    PassiveCANNObservation,
    PassiveRingCANNObserver,
    PeriodicStateInput,
)
from .orbital_phase_adapter import (
    OrbitalPhaseState,
    OrbitalPlaneFrame,
    extract_orbital_phase_state,
)
from .orbital_phase_sidecar import (
    OrbitalPhaseSidecarHistory,
    run_orbital_phase_sidecar,
)
from .ring_cann import (
    CANNOutput,
    RingCANN,
    RingCANNConfig,
    periodic_spectral_derivative,
)
from .line_cann import (
    LineCANN,
    LineCANNConfig,
    LineCANNOutput,
    decode_line_activity,
    decode_line_activity_peak_fit,
    decode_line_activity_boundary_corrected,
)
from .coupled_ring_line_cann import (
    CoupledRingLineCANN,
    CoupledRingLineCANNConfig,
    CoupledRingLineCANNOutput,
)
from .plane_cann import PlaneCANN, PlaneCANNConfig, PlaneCANNOutput

__all__ = [
    "OrbitalPhaseSidecarHistory", "run_orbital_phase_sidecar",
    "OrbitalPhaseState", "OrbitalPlaneFrame", "extract_orbital_phase_state",
    "PassiveCANNObservation", "PassiveRingCANNObserver", "PeriodicStateInput",
    "CANNOutput", "RingCANN", "RingCANNConfig",
    "periodic_spectral_derivative",
    "LineCANN", "LineCANNConfig", "LineCANNOutput", "decode_line_activity",
    "decode_line_activity_peak_fit",
    "decode_line_activity_boundary_corrected",
    "CoupledRingLineCANN", "CoupledRingLineCANNConfig",
    "CoupledRingLineCANNOutput",
    "PlaneCANN", "PlaneCANNConfig", "PlaneCANNOutput",
]
