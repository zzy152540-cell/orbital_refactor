from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments.single_satellite_cann_comparison import (
    _preprocess_valid_infrared_azimuth_with_cann,
    _preprocess_valid_optical_with_plane_cann,
    _preprocess_valid_radar_with_line_cann,
)
from interfaces.data_objects import Observation


@dataclass(frozen=True)
class MultimodalCANNPreprocessorConfig:
    """Select the measurement-domain CANN adapters for one node stream."""

    infrared: bool = True
    radar: bool = True
    optical: bool = True
    infrared_method: str = "hybrid_ring_line_cann"


class MultimodalCANNPreprocessor:
    """Experimental three-modal adapter for single or cooperative pipelines.

    Instances are intentionally state-isolated by node.  The underlying CANNs
    are reconstructed for every complete stream, matching the existing
    single-satellite validation path exactly.  The class only transforms
    standard observations and never receives an EKF/Schmidt/CI object.
    """

    def __init__(self, config: MultimodalCANNPreprocessorConfig | None = None):
        self.config = config or MultimodalCANNPreprocessorConfig()
        self.process_count = 0

    def process(
        self, observations: Sequence[Observation], timestamps: np.ndarray,
    ) -> list[Observation]:
        self.process_count += 1
        result = list(observations)
        timeline = np.asarray(timestamps, dtype=float).reshape(-1)
        _validate_modal_streams(result, timeline, self.config)
        if self.config.optical:
            result = _preprocess_valid_optical_with_plane_cann(result, timeline)
        if self.config.infrared:
            result = _preprocess_valid_infrared_azimuth_with_cann(
                result, timeline, method=self.config.infrared_method,
            )
        if self.config.radar:
            result = _preprocess_valid_radar_with_line_cann(result, timeline)
        return result


def _validate_modal_streams(
    observations: Sequence[Observation], timestamps: np.ndarray,
    config: MultimodalCANNPreprocessorConfig,
) -> None:
    if timestamps.size == 0 or np.any(~np.isfinite(timestamps)):
        raise ValueError("CANN preprocessing requires a finite nonempty timeline.")
    enabled = {
        "optical": config.optical,
        "infrared": config.infrared,
        "radar": config.radar,
    }
    aliases = {"ir": "infrared", "rad": "radar", "opt": "optical"}
    grouped: dict[str, list[Observation]] = {name: [] for name in enabled}
    for observation in observations:
        name = aliases.get(observation.modality.lower(), observation.modality.lower())
        if name in grouped:
            grouped[name].append(observation)
    for name, active in enabled.items():
        if not active:
            continue
        stream = grouped[name]
        if len(stream) != timestamps.size:
            raise ValueError(
                f"Enabled {name} CANN requires one observation per timestamp."
            )
        observed_timestamps = np.asarray(
            [item.timestamp for item in stream], dtype=float,
        )
        if not np.allclose(observed_timestamps, timestamps, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} observations must follow the runtime timeline.")
