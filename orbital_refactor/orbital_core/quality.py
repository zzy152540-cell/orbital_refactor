from __future__ import annotations

import numpy as np


def quality_score_from_covariance(
    covariance: np.ndarray,
    *,
    nis: float | None = None,
    expected_nis: float | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Return a bounded quality score without changing the legacy filter result.

    Covariance information is always used. NIS contributes only when both NIS and
    its expected value are supplied, so existing pipelines can adopt this helper
    incrementally.
    """
    trace = max(float(np.trace(covariance)), epsilon)
    covariance_score = 1.0 / (1.0 + trace)
    if nis is None or expected_nis is None or not np.isfinite(nis):
        return covariance_score
    consistency_score = 1.0 / (1.0 + abs(float(nis) - expected_nis) / max(expected_nis, epsilon))
    return float(np.clip(covariance_score * consistency_score, 0.0, 1.0))
