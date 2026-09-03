from __future__ import annotations

from copy import deepcopy

import numpy as np


class RecoveryConfirmationPreprocessor:
    """Causal two-sample recovery confirmation without a CANN model."""

    def __init__(self, maximum_gap=20.0):
        self.maximum_gap = float(maximum_gap)

    def process(self, observations, timestamps):
        del timestamps
        result = deepcopy(list(observations))
        for modality in ("radar", "infrared", "optical"):
            stream = [item for item in result if item.modality.lower() == modality]
            self._process_stream(stream, modality)
        return result

    def _process_stream(self, stream, modality):
        last_trusted_time = None
        pending = None
        for item in stream:
            if not item.valid_flag:
                continue
            if last_trusted_time is None:
                last_trusted_time = float(item.timestamp)
                continue
            long_gap = float(item.timestamp) - last_trusted_time > self.maximum_gap
            if not long_gap and pending is None:
                last_trusted_time = float(item.timestamp)
                continue
            if pending is not None and _consistent(
                modality, pending.measurement, item.measurement,
                float(item.timestamp) - float(pending.timestamp),
            ):
                item.metadata = {
                    **item.metadata, "recovery_confirmation": "confirmed",
                }
                last_trusted_time = float(item.timestamp)
                pending = None
                continue
            pending = item
            item.valid_flag = False
            item.metadata = {
                **item.metadata, "recovery_confirmation": "pending",
            }


def _consistent(modality, previous, current, dt):
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if modality == "radar":
        expected_change = 0.5 * (previous[1] + current[1]) * max(dt, 1e-12)
        return bool(
            abs(current[0] - previous[0] - expected_change) <= 300.0
            and abs(current[1] - previous[1]) <= 0.5
        )
    if modality == "infrared":
        difference = (current - previous + np.pi) % (2.0 * np.pi) - np.pi
        return bool(np.all(np.abs(difference) <= np.deg2rad(1.0)))
    if modality == "optical":
        return bool(np.all(np.abs(current - previous) <= 0.1))
    raise ValueError(f"Unsupported recovery-confirmation modality: {modality}")
