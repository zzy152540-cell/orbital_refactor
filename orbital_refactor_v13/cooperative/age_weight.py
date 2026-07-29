from __future__ import annotations

import numpy as np

Array = np.ndarray


def inflate_covariance_by_age(
    covariance: Array,
    age: float,
    penalty: float = 1e-2,
) -> Array:
    """Inflate covariance according to information age.

    P_age = P + lambda * age * I

    A larger information age indicates older information and therefore
    decreases its contribution during covariance intersection.
    """
    P = np.asarray(covariance, dtype=float)
    if age <= 0:
        return P.copy()

    return P + penalty * float(age) * np.eye(P.shape[0])
