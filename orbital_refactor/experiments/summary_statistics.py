from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def interval_coverage(values, interval: tuple[float, float]) -> float:
    """Return the fraction of values inside a closed interval."""

    array = np.asarray(values)
    lower, upper = interval
    return float(np.mean((array >= lower) & (array <= upper)))


def mean_metric_dict(
    values: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Average every metric over mappings that contain that metric."""

    keys = sorted({key for value in values for key in value})
    return {
        key: float(np.mean([value[key] for value in values if key in value]))
        for key in keys
    }
