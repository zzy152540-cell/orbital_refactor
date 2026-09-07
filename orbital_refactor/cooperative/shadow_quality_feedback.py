from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from brain_inspired.modality_shadow_quality import ModalityShadowQuality
from interfaces.data_objects import ObservationMessage


@dataclass(frozen=True)
class ShadowQualityFeedbackConfig:
    enabled: bool = False
    minimum_feedback_quality: float = 0.05
    maximum_covariance_inflation: float = 4.0
    rejection_threshold: float | None = None

    def validate(self) -> None:
        if (not np.isfinite(self.minimum_feedback_quality)
                or not 0.0 < self.minimum_feedback_quality <= 1.0):
            raise ValueError("minimum_feedback_quality must lie in (0, 1].")
        if (not np.isfinite(self.maximum_covariance_inflation)
                or self.maximum_covariance_inflation < 1.0):
            raise ValueError("maximum_covariance_inflation must be at least one.")
        if self.rejection_threshold is not None and (
            not np.isfinite(self.rejection_threshold)
            or not 0.0 <= self.rejection_threshold <= 1.0
        ):
            raise ValueError("rejection_threshold must lie in [0, 1].")


def apply_shadow_quality_feedback(
    observations: list[ObservationMessage],
    quality_records: list[ModalityShadowQuality], *,
    config: ShadowQualityFeedbackConfig | None = None,
) -> list[ObservationMessage]:
    """Apply bounded covariance inflation using delayed shadow quality only."""
    settings = config or ShadowQualityFeedbackConfig()
    settings.validate()
    if not settings.enabled:
        return list(observations)
    quality_by_id = {record.information_id: record for record in quality_records}
    if len(quality_by_id) != len(quality_records):
        raise ValueError("Shadow-quality information IDs must be unique.")
    adjusted = []
    for observation in observations:
        try:
            record = quality_by_id[observation.information_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing shadow quality for {observation.information_id}."
            ) from exc
        quality = float(np.clip(
            record.feedback_quality, settings.minimum_feedback_quality, 1.0,
        ))
        inflation = float(min(
            settings.maximum_covariance_inflation, 1.0 / quality,
        ))
        rejected = bool(
            settings.rejection_threshold is not None
            and quality < settings.rejection_threshold
        )
        adjusted.append(replace(
            observation,
            covariance=np.asarray(observation.covariance, dtype=float) * inflation,
            valid_flag=bool(observation.valid_flag and not rejected),
            metadata={
                **observation.metadata,
                "shadow_feedback_enabled": True,
                "shadow_feedback_quality": quality,
                "shadow_covariance_inflation": inflation,
                "shadow_feedback_rejected": rejected,
                "shadow_feedback_delay_epochs": 1,
            },
        ))
    return adjusted
