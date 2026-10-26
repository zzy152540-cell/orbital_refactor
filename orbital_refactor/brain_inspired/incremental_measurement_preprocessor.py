"""Persistent per-epoch CANN preprocessing for multimodal measurements."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Sequence

import numpy as np

from interfaces.data_objects import Observation

from .cann_measurement_adapter import preprocess_observation
from .coupled_ring_line_cann import (
    CoupledRingLineCANN,
    CoupledRingLineCANNConfig,
)
from .line_cann import LineCANN, LineCANNConfig
from .plane_cann import PlaneCANN, PlaneCANNConfig
from .ring_cann import RingCANNConfig


def _realtime_infrared_config() -> CoupledRingLineCANNConfig:
    return CoupledRingLineCANNConfig(
        ring=RingCANNConfig(
            num_neurons=90,
            internal_dt=0.002,
            integration_mode="phase_shift",
        ),
        bias_anchor_mode="hybrid_dual",
        minimum_bias_baseline=120.0,
        anchor_agreement_scale=np.deg2rad(0.004),
        line=LineCANNConfig(
            minimum_value=np.deg2rad(-0.05),
            maximum_value=np.deg2rad(0.05),
            tuning_width=np.deg2rad(0.003),
            cue_gain=0.2,
        ),
    )


def _research_infrared_config() -> CoupledRingLineCANNConfig:
    config = _realtime_infrared_config()
    return replace(
        config,
        ring=RingCANNConfig(num_neurons=180, internal_dt=0.001),
    )


@dataclass(frozen=True)
class IncrementalMultimodalCANNConfig:
    infrared: bool = True
    radar: bool = True
    optical: bool = True
    radial_backup: bool = False
    profile: str = "realtime"
    infrared_coupled_config: CoupledRingLineCANNConfig | None = None
    recovery_gap_seconds: float = 2.0
    trusted_measurement_confidence: float = 0.8
    prefer_trusted_measurements: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.recovery_gap_seconds)
            or self.recovery_gap_seconds <= 0.0
        ):
            raise ValueError("recovery_gap_seconds must be positive and finite.")
        if not 0.0 <= self.trusted_measurement_confidence <= 1.0:
            raise ValueError(
                "trusted_measurement_confidence must lie in [0, 1]."
            )

    def resolved_infrared_config(self) -> CoupledRingLineCANNConfig:
        if self.profile not in {"realtime", "research"}:
            raise ValueError("CANN profile must be 'realtime' or 'research'.")
        config = self.infrared_coupled_config
        if config is None:
            config = (
                _realtime_infrared_config()
                if self.profile == "realtime"
                else _research_infrared_config()
            )
        config.validate()
        return config


@dataclass
class IncrementalCANNPerformance:
    call_count: int = 0
    observation_count: int = 0
    total_seconds: float = 0.0
    seconds_by_modality: dict[str, float] = field(default_factory=dict)


class IncrementalMultimodalCANNPreprocessor:
    """Advance each modality once per new epoch without replaying history."""

    def __init__(self, config: IncrementalMultimodalCANNConfig | None = None):
        self.config = config or IncrementalMultimodalCANNConfig()
        infrared_config = self.config.resolved_infrared_config()
        self._states = {
            "RADAR": _IncrementalRadar(self.config),
            "OPTICAL": _IncrementalOptical(self.config),
            "INFRARED": _IncrementalInfrared(infrared_config, self.config),
        }
        self._radial_backup_state = _IncrementalRadar(self.config)
        self.performance = IncrementalCANNPerformance()

    def process(
        self, observations: Sequence[Observation], timestamps: np.ndarray,
    ) -> list[Observation]:
        values = list(observations)
        timeline = np.asarray(timestamps, dtype=float).reshape(-1)
        if timeline.shape != (1,) or not np.isfinite(timeline[0]):
            raise ValueError("Incremental CANN requires exactly one finite epoch.")
        timestamp = float(timeline[0])
        if any(not np.isclose(item.timestamp, timestamp) for item in values):
            raise ValueError("Incremental CANN observations must match the epoch.")
        normalized = [_canonical_modality(item.modality) for item in values]
        radar_available = any(
            modality == "RADAR" and item.valid_flag
            for item, modality in zip(values, normalized)
        )
        enabled = {
            "RADAR": self.config.radar,
            "OPTICAL": self.config.optical,
            "INFRARED": self.config.infrared,
        }
        for modality in enabled:
            if sum(name == modality for name in normalized) > 1:
                raise ValueError(
                    f"Incremental CANN accepts at most one {modality} observation per epoch."
                )
        started = time.perf_counter()
        result = []
        for item, modality in zip(values, normalized):
            if modality not in enabled or not enabled[modality]:
                result.append(item)
                continue
            modality_started = time.perf_counter()
            processed = self._states[modality].process(item)
            if (
                self.config.radial_backup
                and not radar_available
                and _has_radial_backup(processed)
            ):
                radial_started = time.perf_counter()
                processed = self._process_radial_backup(processed)
                radial_elapsed = time.perf_counter() - radial_started
                self.performance.seconds_by_modality["RADIAL_BACKUP"] = (
                    self.performance.seconds_by_modality.get(
                        "RADIAL_BACKUP", 0.0,
                    ) + radial_elapsed
                )
            result.append(processed)
            elapsed = time.perf_counter() - modality_started
            self.performance.seconds_by_modality[modality] = (
                self.performance.seconds_by_modality.get(modality, 0.0) + elapsed
            )
        self.performance.call_count += 1
        self.performance.observation_count += len(values)
        self.performance.total_seconds += time.perf_counter() - started
        return result

    def _process_radial_backup(self, item: Observation) -> Observation:
        metadata = dict(item.metadata)
        covariance = np.diag([
            float(metadata["rangeProxyVarianceM2"]),
            float(metadata["rangeRateProxyVarianceM2ps2"]),
        ])
        source_id = str(metadata.get(
            "observation_id",
            f"{item.observer_id}:{item.target_id}:{item.timestamp:.12g}",
        ))
        radial = self._radial_backup_state.process(replace(
            item,
            modality="RADAR",
            source_type="RADIAL_BACKUP_CANN_INPUT",
            measurement=np.asarray([
                metadata["rangeProxyM"], metadata["rangeRateProxyMps"],
            ], dtype=float),
            covariance=covariance,
            frame="SPRI",
            metadata={
                **metadata,
                "observation_id": f"{source_id}:radial_backup",
            },
        ))
        radial_covariance = np.asarray(radial.covariance, dtype=float)
        metadata.update({
            "rangeProxyM": float(radial.measurement[0]),
            "rangeRateProxyMps": float(radial.measurement[1]),
            "rangeProxyVarianceM2": float(radial_covariance[0, 0]),
            "rangeRateProxyVarianceM2ps2": float(radial_covariance[1, 1]),
            "radialBackupCannProcessed": True,
            "radialBackupCannDiagnostics": dict(
                radial.metadata.get("cann_diagnostics", {})
            ),
            "radialBackupCannSourceMeasurementIds": tuple(
                radial.metadata.get("source_measurement_ids", ())
            ),
        })
        return replace(item, metadata=metadata)


def _canonical_modality(value: str) -> str:
    name = str(value).upper()
    return {"IR": "INFRARED", "RAD": "RADAR", "OPT": "OPTICAL"}.get(
        name, name,
    )


def _has_radial_backup(item: Observation) -> bool:
    metadata = item.metadata
    return bool(
        metadata.get("rangeProxySource")
        == "INDEPENDENT_RADIAL_BACKUP_SHADOW"
        and metadata.get("rangeRateProxySource")
        == "INDEPENDENT_RADIAL_BACKUP_SHADOW"
        and metadata.get("rangeProxyM") is not None
        and metadata.get("rangeRateProxyMps") is not None
        and metadata.get("rangeProxyVarianceM2") is not None
        and metadata.get("rangeRateProxyVarianceM2ps2") is not None
    )


def _processed(source, measurement, diagnostics, *, valid=None):
    result = preprocess_observation(
        source, measurement=np.asarray(measurement, dtype=float),
        diagnostics={"incremental": True, **diagnostics},
    )
    if valid is not None:
        result.valid_flag = bool(valid)
    return result


class _IncrementalRadar:
    def __init__(self, policy: IncrementalMultimodalCANNConfig):
        self.policy = policy
        self.range_cann = LineCANN(LineCANNConfig(
            num_neurons=501, minimum_value=0.0, maximum_value=50_000_000.0,
            tuning_width=400_000.0, cue_gain=0.8,
        ))
        self.rate_cann = LineCANN(LineCANNConfig(
            num_neurons=401, minimum_value=-20_000.0, maximum_value=20_000.0,
            tuning_width=400.0, cue_gain=1.0,
        ))
        self.initialized = False
        self.last_timestamp = None
        self.last_valid_timestamp = None
        self.last_trusted_timestamp = None
        self.last_trusted_measurement = None
        self.held_rate_acceleration = 0.0
        self.trusted_sample_count = 0
        self.pending_measurement = None
        self.pending_timestamp = None

    def process(self, item: Observation) -> Observation:
        measurement = np.asarray(item.measurement, dtype=float).reshape(2)
        timestamp = float(item.timestamp)
        if not self.initialized:
            if not item.valid_flag:
                return item
            self.range_cann.reset(measurement[0], timestamp=timestamp)
            self.rate_cann.reset(measurement[1], timestamp=timestamp)
            self.initialized = True
            self.last_timestamp = timestamp
            self.last_valid_timestamp = timestamp
            self.last_trusted_timestamp = timestamp
            self.last_trusted_measurement = measurement.copy()
            self.trusted_sample_count = 1
            return _processed(item, measurement, {
                "radar_preprocessor": "incremental_range_rate_dual_line_cann",
                "radar_initialized": True,
            })
        dt = _strict_dt(timestamp, self.last_timestamp)
        range_output = self.range_cann.step(
            self.rate_cann.output().decoded_value, dt,
        )
        rate_output = self.rate_cann.step(self.held_rate_acceleration, dt)
        self.last_timestamp = timestamp
        if not item.valid_flag:
            return item
        gap = timestamp - self.last_valid_timestamp
        range_innovation = measurement[0] - range_output.decoded_value
        rate_innovation = measurement[1] - rate_output.decoded_value
        long_gap = gap > self.policy.recovery_gap_seconds
        trusted_measurement = bool(
            self.policy.prefer_trusted_measurements
            and item.confidence >= self.policy.trusted_measurement_confidence
        )
        candidate_consistent = False
        if self.pending_measurement is not None:
            candidate_dt = max(timestamp - self.pending_timestamp, 1.0e-12)
            expected_range_change = 0.5 * (
                self.pending_measurement[1] + measurement[1]
            ) * candidate_dt
            candidate_consistent = bool(
                abs(measurement[0] - self.pending_measurement[0]
                    - expected_range_change) <= 300.0
                and abs(measurement[1] - self.pending_measurement[1]) <= 0.5
            )
        reanchored = bool(
            (long_gap and candidate_consistent)
            or (not long_gap and (
                range_output.saturated_at_boundary
                or rate_output.saturated_at_boundary
            ))
        )
        recovery_pending = False
        range_substituted = False
        rate_substituted = False
        filtered = measurement.copy()
        if long_gap and trusted_measurement:
            range_output = self.range_cann.reset(measurement[0], timestamp=timestamp)
            rate_output = self.rate_cann.reset(measurement[1], timestamp=timestamp)
            self.pending_measurement = None
            self.pending_timestamp = None
            reanchored = True
        elif long_gap and not candidate_consistent:
            self.pending_measurement = measurement.copy()
            self.pending_timestamp = timestamp
            recovery_pending = True
        elif reanchored:
            range_output = self.range_cann.reset(measurement[0], timestamp=timestamp)
            rate_output = self.rate_cann.reset(measurement[1], timestamp=timestamp)
            self.pending_measurement = None
            self.pending_timestamp = None
        else:
            warming_up = self.trusted_sample_count < 2
            range_substituted = bool(
                not warming_up and abs(range_innovation) > 300.0
            )
            rate_substituted = bool(
                not warming_up and abs(rate_innovation) > 0.5
            )
            if trusted_measurement and (range_substituted or rate_substituted):
                range_output = self.range_cann.reset(
                    measurement[0], timestamp=timestamp,
                )
                rate_output = self.rate_cann.reset(
                    measurement[1], timestamp=timestamp,
                )
                range_substituted = False
                rate_substituted = False
                reanchored = True
            filtered = np.array([
                range_output.decoded_value if range_substituted else measurement[0],
                rate_output.decoded_value if rate_substituted else measurement[1],
            ])
            if not range_substituted:
                range_output = self.range_cann.apply_value_cue(measurement[0])
            if not rate_substituted:
                rate_output = self.rate_cann.apply_value_cue(measurement[1])
        if not recovery_pending:
            if not range_substituted and not rate_substituted:
                self.held_rate_acceleration = (
                    measurement[1] - self.last_trusted_measurement[1]
                ) / max(timestamp - self.last_trusted_timestamp, 1.0e-12)
                self.last_trusted_measurement = measurement.copy()
                self.last_trusted_timestamp = timestamp
                self.trusted_sample_count += 1
            self.last_valid_timestamp = timestamp
        diagnostics = {
            "radar_preprocessor": "incremental_range_rate_dual_line_cann",
            "radar_reanchored": reanchored,
            "range_substituted": range_substituted,
            "range_rate_substituted": rate_substituted,
            "recovery_pending": recovery_pending,
            "range_bump_concentration": float(range_output.bump_concentration),
            "range_rate_bump_concentration": float(
                rate_output.bump_concentration
            ),
        }
        return _processed(
            item, filtered, diagnostics,
            valid=False if recovery_pending else None,
        )


class _IncrementalOptical:
    def __init__(self, policy: IncrementalMultimodalCANNConfig):
        self.policy = policy
        axis = LineCANNConfig(
            num_neurons=401, minimum_value=-2_000.0, maximum_value=2_000.0,
            tuning_width=40.0, cue_gain=1.0,
        )
        self.cann = PlaneCANN(PlaneCANNConfig(
            x_axis=axis, y_axis=axis, materialize_joint_activity=False,
        ))
        self.initialized = False
        self.last_timestamp = None
        self.last_trusted_timestamp = None
        self.last_trusted_measurement = None
        self.velocity = np.zeros(2)
        self.pending_measurement = None
        self.pending_timestamp = None

    def process(self, item: Observation) -> Observation:
        measurement = np.asarray(item.measurement, dtype=float).reshape(2)
        timestamp = float(item.timestamp)
        if not self.initialized:
            if not item.valid_flag:
                return item
            self.cann.reset(measurement, timestamp=timestamp)
            self.initialized = True
            self.last_timestamp = timestamp
            self.last_trusted_timestamp = timestamp
            self.last_trusted_measurement = measurement.copy()
            return _processed(item, measurement, {
                "optical_preprocessor": "incremental_fault_aware_plane_cann",
                "optical_initialized": True,
            })
        dt = _strict_dt(timestamp, self.last_timestamp)
        output = self.cann.step(self.velocity, dt)
        self.last_timestamp = timestamp
        if not item.valid_flag:
            return item
        gap = timestamp - self.last_trusted_timestamp
        innovation = measurement - output.decoded_position
        stable_motion = bool(np.all(np.abs(self.velocity) * dt < 0.1))
        inside_stable_field = bool(
            np.all(np.abs(measurement) <= 10.0)
            and np.all(np.abs(output.decoded_position) <= 10.0)
        )
        long_gap = gap > self.policy.recovery_gap_seconds
        trusted_measurement = bool(
            self.policy.prefer_trusted_measurements
            and item.confidence >= self.policy.trusted_measurement_confidence
        )
        candidate_consistent = bool(
            self.pending_measurement is not None
            and np.all(np.abs(measurement - self.pending_measurement) <= 0.1)
        )
        reanchored = bool(
            (long_gap and candidate_consistent)
            or (not long_gap and (
                output.saturated_at_boundary
                or not stable_motion or not inside_stable_field
            ))
        )
        recovery_pending = False
        substituted = np.zeros(2, dtype=bool)
        filtered = measurement.copy()
        if long_gap and trusted_measurement:
            output = self.cann.reset(measurement, timestamp=timestamp)
            self.velocity = np.zeros(2)
            self.pending_measurement = None
            self.pending_timestamp = None
            self.last_trusted_measurement = measurement.copy()
            self.last_trusted_timestamp = timestamp
            reanchored = True
        elif long_gap and not candidate_consistent:
            self.pending_measurement = measurement.copy()
            self.pending_timestamp = timestamp
            recovery_pending = True
        elif reanchored:
            output = self.cann.reset(measurement, timestamp=timestamp)
            if long_gap:
                self.velocity = (measurement - self.pending_measurement) / max(
                    timestamp - self.pending_timestamp, 1.0e-12,
                )
            else:
                self.velocity = np.zeros(2)
            self.pending_measurement = None
            self.pending_timestamp = None
            self.last_trusted_measurement = measurement.copy()
            self.last_trusted_timestamp = timestamp
        else:
            joint_anomaly = bool(np.all(np.abs(innovation) > 0.05))
            substituted[:] = joint_anomaly
            if trusted_measurement and joint_anomaly:
                output = self.cann.reset(measurement, timestamp=timestamp)
                substituted[:] = False
                self.velocity = np.zeros(2)
                self.last_trusted_measurement = measurement.copy()
                self.last_trusted_timestamp = timestamp
                reanchored = True
            filtered = np.where(substituted, output.decoded_position, measurement)
            output = self.cann.apply_position_cue(filtered, cue_gain=1.0)
        if not np.any(substituted) and not recovery_pending and not reanchored:
            self.velocity = (measurement - self.last_trusted_measurement) / max(
                gap, 1.0e-12,
            )
            self.last_trusted_measurement = measurement.copy()
            self.last_trusted_timestamp = timestamp
        diagnostics = {
            "optical_preprocessor": "incremental_fault_aware_plane_cann",
            "optical_reanchored": reanchored,
            "u_substituted": bool(substituted[0]),
            "v_substituted": bool(substituted[1]),
            "recovery_pending": recovery_pending,
            "plane_bump_concentration": float(output.bump_concentration),
        }
        return _processed(
            item, filtered, diagnostics,
            valid=False if recovery_pending else None,
        )


class _IncrementalInfrared:
    def __init__(
        self,
        config: CoupledRingLineCANNConfig,
        policy: IncrementalMultimodalCANNConfig,
    ):
        self.policy = policy
        self.azimuth = CoupledRingLineCANN(config)
        self.elevation = LineCANN(LineCANNConfig(
            num_neurons=361,
            minimum_value=-0.5 * np.pi,
            maximum_value=0.5 * np.pi,
            tuning_width=np.deg2rad(2.0),
            cue_gain=1.0,
        ))
        self.initialized = False
        self.last_timestamp = None
        self.azimuth_held_rate = 0.0
        self.azimuth_last_trusted_timestamp = None
        self.azimuth_last_trusted = None
        self.azimuth_pending = None
        self.azimuth_pending_timestamp = None
        self.elevation_held_rate = 0.0
        self.elevation_last_valid_timestamp = None
        self.elevation_last_trusted = None
        self.elevation_pending = None
        self.elevation_pending_timestamp = None

    def process(self, item: Observation) -> Observation:
        measurement = np.asarray(item.measurement, dtype=float).reshape(2)
        timestamp = float(item.timestamp)
        if not self.initialized:
            if not item.valid_flag:
                return item
            self.azimuth.initialize(phase=measurement[0], timestamp=timestamp)
            self.elevation.reset(measurement[1], timestamp=timestamp)
            self.initialized = True
            self.last_timestamp = timestamp
            self.azimuth_last_trusted_timestamp = timestamp
            self.azimuth_last_trusted = float(measurement[0])
            self.elevation_last_valid_timestamp = timestamp
            self.elevation_last_trusted = float(measurement[1])
            return _processed(item, measurement, {
                "azimuth_preprocessor": "incremental_hybrid_ring_line_cann",
                "infrared_initialized": True,
            })
        dt = _strict_dt(timestamp, self.last_timestamp)
        azimuth_output, azimuth_diagnostics = self._update_azimuth(
            timestamp, float(measurement[0]), bool(item.valid_flag), item.confidence,
        )
        elevation_output, elevation_diagnostics = self._update_elevation(
            timestamp, dt, float(measurement[1]), bool(item.valid_flag),
            item.confidence,
        )
        self.last_timestamp = timestamp
        if not item.valid_flag:
            return item
        diagnostics = {
            "azimuth_preprocessor": "incremental_hybrid_ring_line_cann",
            **azimuth_diagnostics,
            **elevation_diagnostics,
        }
        diagnostics["recovery_pending"] = bool(
            azimuth_diagnostics["azimuth_recovery_pending"]
            or elevation_diagnostics["elevation_recovery_pending"]
        )
        filtered = measurement.copy()
        if diagnostics["azimuth_substituted"]:
            filtered[0] = _wrapped_difference(azimuth_output.decoded_phase, 0.0)
        if diagnostics["elevation_substituted"]:
            filtered[1] = elevation_output.decoded_value
        return _processed(
            item, filtered, diagnostics,
            valid=False if diagnostics["recovery_pending"] else None,
        )

    def _update_azimuth(self, timestamp, hint, available, confidence):
        gap = timestamp - self.azimuth_last_trusted_timestamp
        reanchored = False
        recovery_pending = False
        long_gap = gap > self.policy.recovery_gap_seconds
        trusted_measurement = bool(
            self.policy.prefer_trusted_measurements
            and confidence >= self.policy.trusted_measurement_confidence
        )
        if available and long_gap and trusted_measurement:
            output = self.azimuth.initialize(phase=hint, timestamp=timestamp)
            self.azimuth_held_rate = 0.0
            self.azimuth_last_trusted_timestamp = timestamp
            self.azimuth_last_trusted = hint
            self.azimuth_pending = None
            self.azimuth_pending_timestamp = None
            reanchored = True
        elif available and long_gap:
            consistent = bool(
                self.azimuth_pending is not None
                and abs(_wrapped_difference(hint, self.azimuth_pending))
                <= np.deg2rad(1.0)
            )
            if consistent:
                output = self.azimuth.initialize(phase=hint, timestamp=timestamp)
                self.azimuth_held_rate = _wrapped_difference(
                    hint, self.azimuth_pending,
                ) / max(timestamp - self.azimuth_pending_timestamp, 1.0e-12)
                self.azimuth_last_trusted_timestamp = timestamp
                self.azimuth_last_trusted = hint
                self.azimuth_pending = None
                self.azimuth_pending_timestamp = None
                reanchored = True
            else:
                output = self.azimuth.update(
                    timestamp=timestamp,
                    measured_phase_rate=self.azimuth_held_rate,
                    phase_hint=None,
                    phase_hint_valid=False,
                )
                self.azimuth_pending = hint
                self.azimuth_pending_timestamp = timestamp
                recovery_pending = True
        else:
            output = self.azimuth.update(
                timestamp=timestamp,
                measured_phase_rate=self.azimuth_held_rate,
                phase_hint=hint if available else None,
                phase_hint_valid=available,
            )
            if available and output.cue_applied:
                self.azimuth_held_rate = _wrapped_difference(
                    hint, self.azimuth_last_trusted,
                ) / max(gap, 1.0e-12)
                self.azimuth_last_trusted_timestamp = timestamp
                self.azimuth_last_trusted = hint
        diagnostics = {
            "decoded_rate_bias": float(output.decoded_rate_bias),
            "bias_observation_count": int(output.bias_observation_count),
            "long_anchor_trusted": output.long_anchor_trusted,
            "azimuth_substituted": bool(
                available and not reanchored and not recovery_pending
                and not output.cue_applied
            ),
            "azimuth_reanchored": reanchored,
            "azimuth_recovery_pending": recovery_pending,
        }
        if (
            diagnostics["azimuth_substituted"]
            and trusted_measurement
        ):
            output = self.azimuth.initialize(phase=hint, timestamp=timestamp)
            self.azimuth_held_rate = 0.0
            self.azimuth_last_trusted_timestamp = timestamp
            self.azimuth_last_trusted = hint
            diagnostics["azimuth_substituted"] = False
            diagnostics["azimuth_reanchored"] = True
        return output, diagnostics

    def _update_elevation(self, timestamp, dt, hint, available, confidence):
        output = self.elevation.step(self.elevation_held_rate, dt)
        substituted = False
        reanchored = False
        recovery_pending = False
        if available:
            gap = timestamp - self.elevation_last_valid_timestamp
            innovation = hint - output.decoded_value
            long_gap = gap > self.policy.recovery_gap_seconds
            trusted_measurement = bool(
                self.policy.prefer_trusted_measurements
                and confidence >= self.policy.trusted_measurement_confidence
            )
            consistent = bool(
                self.elevation_pending is not None
                and abs(hint - self.elevation_pending) <= np.deg2rad(1.0)
            )
            reanchored = bool(
                (long_gap and consistent)
                or (not long_gap and output.saturated_at_boundary)
            )
            if long_gap and trusted_measurement:
                output = self.elevation.reset(hint, timestamp=timestamp)
                self.elevation_held_rate = 0.0
                self.elevation_last_trusted = hint
                self.elevation_pending = None
                self.elevation_pending_timestamp = None
                reanchored = True
            elif long_gap and not consistent:
                self.elevation_pending = hint
                self.elevation_pending_timestamp = timestamp
                recovery_pending = True
            elif reanchored:
                output = self.elevation.reset(hint, timestamp=timestamp)
                self.elevation_held_rate = (
                    (hint - self.elevation_pending)
                    / max(timestamp - self.elevation_pending_timestamp, 1.0e-12)
                    if long_gap else 0.0
                )
                self.elevation_last_trusted = hint
                self.elevation_pending = None
                self.elevation_pending_timestamp = None
            elif abs(innovation) > np.deg2rad(3.0):
                if trusted_measurement:
                    output = self.elevation.reset(hint, timestamp=timestamp)
                    self.elevation_held_rate = 0.0
                    self.elevation_last_trusted = hint
                    reanchored = True
                else:
                    substituted = True
            else:
                output = self.elevation.apply_value_cue(hint)
                self.elevation_held_rate = (
                    hint - self.elevation_last_trusted
                ) / max(gap, 1.0e-12)
                self.elevation_last_trusted = hint
            if not substituted and not recovery_pending:
                self.elevation_last_valid_timestamp = timestamp
        diagnostics = {
            "elevation_bump_concentration": float(output.bump_concentration),
            "elevation_bump_width": float(output.bump_width),
            "elevation_saturated_at_boundary": bool(
                output.saturated_at_boundary
            ),
            "elevation_reanchored": reanchored,
            "elevation_substituted": substituted,
            "elevation_recovery_pending": recovery_pending,
        }
        return output, diagnostics


def _strict_dt(timestamp, previous):
    dt = float(timestamp - previous)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Incremental CANN timestamps must be strictly increasing.")
    return dt


def _wrapped_difference(actual, expected):
    return float((actual - expected + np.pi) % (2.0 * np.pi) - np.pi)
