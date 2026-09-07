from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.navigation_shadow_quality import NavigationShadowQualityHistory
from interfaces.data_objects import ObservationMessage


@dataclass(frozen=True)
class ModalityShadowQuality:
    information_id: str
    timestamp: float
    observer_id: str
    target_id: str
    modality: str
    navigation_discrepancy: float
    navigation_quality: float
    frontend_quality: float
    shadow_quality: float
    feedback_quality: float
    valid: bool


def build_modality_shadow_quality(
    *, observations: list[ObservationMessage],
    navigation_quality_by_node: dict[str, NavigationShadowQualityHistory],
    minimum_quality: float = 0.05,
) -> list[ModalityShadowQuality]:
    """Project node navigation diagnostics onto modality-specific link quality.

    Optical and infrared depend on endpoint direction discrepancies. Radar
    range/range-rate depends on both direction and radial discrepancies. The
    returned feedback signal is delayed within each directed modality link.
    """
    if not np.isfinite(minimum_quality) or not 0.0 <= minimum_quality <= 1.0:
        raise ValueError("minimum_quality must lie in [0, 1].")
    ordered = sorted(observations, key=lambda item: (
        item.observer_id, item.target_id, item.modality, item.timestamp,
        item.information_id,
    ))
    previous_quality = {}
    results = []
    for observation in ordered:
        observer = _sample_navigation(
            navigation_quality_by_node, observation.observer_id,
            observation.timestamp,
        )
        target = _sample_navigation(
            navigation_quality_by_node, observation.target_id,
            observation.timestamp,
        )
        modality = observation.modality.upper()
        if modality in {"OPTICAL", "INFRARED", "AZ_EL"}:
            components = (observer[0], target[0])
        elif modality in {"RADAR", "RANGE", "RANGE_RATE"}:
            components = (observer[0], target[0], observer[1], target[1])
        else:
            raise ValueError(f"Unsupported shadow-quality modality: {modality}")
        discrepancy = float(np.sqrt(np.mean(np.square(components))))
        freshness = min(observer[2], target[2])
        navigation_quality = float(np.clip(
            np.exp(-0.5 * discrepancy**2) * freshness,
            minimum_quality, 1.0,
        ))
        frontend = float(observation.metadata.get(
            "frontend_quality_score", 1.0,
        ))
        if not np.isfinite(frontend) or not 0.0 <= frontend <= 1.0:
            raise ValueError("frontend_quality_score must lie in [0, 1].")
        valid = bool(observation.valid_flag and observer[3] and target[3])
        quality = float(np.clip(
            navigation_quality * frontend, minimum_quality, 1.0,
        )) if valid else minimum_quality
        link = (observation.observer_id, observation.target_id, modality)
        feedback = float(previous_quality.get(link, minimum_quality))
        previous_quality[link] = quality
        results.append(ModalityShadowQuality(
            information_id=observation.information_id,
            timestamp=float(observation.timestamp),
            observer_id=observation.observer_id,
            target_id=observation.target_id, modality=modality,
            navigation_discrepancy=discrepancy,
            navigation_quality=navigation_quality,
            frontend_quality=frontend, shadow_quality=quality,
            feedback_quality=feedback, valid=valid,
        ))
    return results


def _sample_navigation(histories, node_id, timestamp):
    if node_id not in histories:
        raise ValueError(f"Missing navigation quality for node {node_id}.")
    history = histories[node_id]
    matches = np.flatnonzero(np.isclose(
        history.timestamps, timestamp, rtol=0.0, atol=1e-9,
    ))
    if matches.size != 1:
        raise ValueError("Observation timestamp must match one navigation epoch.")
    index = int(matches[0])
    return (
        float(history.direction_discrepancy[index]),
        float(history.radial_discrepancy[index]),
        float(history.freshness_quality[index]),
        bool(history.valid[index] and not history.boundary_saturated[index]),
    )
