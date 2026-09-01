from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_phase_sidecar import run_orbital_phase_sidecar
from brain_inspired.cann_measurement_adapter import preprocess_observation
from brain_inspired.coupled_ring_line_cann import (
    CoupledRingLineCANN, CoupledRingLineCANNConfig,
)
from brain_inspired.line_cann import LineCANN, LineCANNConfig
from brain_inspired.plane_cann import PlaneCANN, PlaneCANNConfig
from cooperative.multi_sat_pipeline import build_module_inputs
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.constants import R_EARTH
from orbital_core.coordinates import state_history_eci_to_spri
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario
from experiments.cann_inter_satellite_azimuth import _cann_tracker, _difference


def run_single_satellite_cann_comparison(
    *, duration: float = 1800.0, dt: float = 2.0, seed: int = 0,
    outage_start: float = 600.0, outage_end: float = 1200.0,
    cue_interval_samples: int = 5,
    outage_modalities: tuple[str, ...] = ("opt", "ir", "rad"),
    outage_windows: dict[str, tuple[float, float]] | None = None,
    enable_cann: bool = True,
    adaptive_cann_preprocess_ir: bool = False,
    hybrid_cann_preprocess_ir: bool = False,
    radar_cann_preprocess: bool = False,
    radar_fault_mode: str | None = None,
    optical_cann_preprocess: bool = False,
    optical_fault_mode: str | None = None,
    infrared_fault_mode: str | None = None,
):
    if adaptive_cann_preprocess_ir and hybrid_cann_preprocess_ir:
        raise ValueError("Select at most one infrared CANN preprocessor.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    target = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(55.0),
        np.deg2rad(15.0), 0.0, np.deg2rad(8.0),
    )
    observer = keplerian_to_eci(
        R_EARTH + 702e3, 0.0012, np.deg2rad(54.5),
        np.deg2rad(14.5), 0.0, np.deg2rad(7.0),
    )
    scenario = generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci={"sat_01": observer},
    )
    observer_track = scenario.observer_trajectories["sat_01"]
    relative = scenario.relative_state_eci_by_node["sat_01"]
    spri = state_history_eci_to_spri(
        relative, observer_track.q_eci2pri_history,
    )
    requested_outages = {
        str(name).lower() for name in (
            outage_windows.keys() if outage_windows is not None
            else outage_modalities
        )
    }
    unknown = requested_outages - {"opt", "ir", "rad"}
    if unknown:
        raise ValueError(f"Unknown outage modalities: {sorted(unknown)}")
    if outage_windows is None:
        normalized_outage_windows = {
            name: (float(outage_start), float(outage_end))
            for name in requested_outages
        }
    else:
        normalized_outage_windows = {}
        for name, window in outage_windows.items():
            start, end = map(float, window)
            if end < start:
                raise ValueError(f"Invalid outage window for {name}: {window}")
            normalized_outage_windows[str(name).lower()] = (start, end)
    outage_window_by_modality = {
        name: (
            (timestamps >= normalized_outage_windows[name][0])
            & (timestamps <= normalized_outage_windows[name][1])
            if name in normalized_outage_windows
            else np.zeros_like(timestamps, dtype=bool)
        )
        for name in ("opt", "ir", "rad")
    }
    outage_window = np.logical_or.reduce(tuple(outage_window_by_modality.values()))
    available_by_modality = {
        name: ~outage_window_by_modality[name] for name in ("opt", "ir", "rad")
    }
    rng = np.random.default_rng(seed)
    observations = []
    observations += create_optical_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        covariance=np.diag([2e-4, 2e-4]) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["opt"],
    )
    if optical_fault_mode is not None:
        _inject_optical_faults(
            observations, timestamps, mode=optical_fault_mode,
        )
    optical_preprocess_diagnostics = None
    if optical_cann_preprocess:
        raw_optical = np.asarray([
            item.measurement.copy() for item in observations
            if item.modality.lower() == "optical"
        ])
        observations = _preprocess_valid_optical_with_plane_cann(
            observations, timestamps,
        )
        processed_optical = [
            item for item in observations if item.modality.lower() == "optical"
        ]
        processed_values = np.asarray([
            item.measurement for item in processed_optical
        ])
        optical_valid = np.asarray([
            item.valid_flag for item in processed_optical
        ], dtype=bool)
        delta = processed_values[optical_valid] - raw_optical[optical_valid]
        optical_preprocess_diagnostics = {
            "u_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "u_substituted", False,
                )) for item in processed_optical if item.valid_flag
            )),
            "v_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "v_substituted", False,
                )) for item in processed_optical if item.valid_flag
            )),
            "uv_rms_change": [
                float(np.sqrt(np.mean(delta[:, axis] ** 2))) if delta.size else 0.0
                for axis in range(2)
            ],
            "reanchor_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "optical_reanchored", False,
                )) for item in processed_optical if item.valid_flag
            )),
        }
    observations += create_infrared_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        covariance=np.diag(np.deg2rad([0.05, 0.05])) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["ir"],
    )
    if infrared_fault_mode is not None:
        _inject_infrared_faults(
            observations, timestamps, mode=infrared_fault_mode,
        )
    observations += create_radar_observations(
        timestamps=timestamps, relative_position_spri=spri[:, :3],
        relative_velocity_spri=spri[:, 3:],
        covariance=np.diag([30.0, 0.05]) ** 2,
        observer_id="sat_01", target_id="target", rng=rng,
        valid_flags=available_by_modality["rad"],
    )
    if radar_fault_mode is not None:
        _inject_radar_faults(
            observations, timestamps, mode=radar_fault_mode,
        )
    radar_preprocess_diagnostics = None
    if radar_cann_preprocess:
        raw_radar = np.asarray([
            item.measurement.copy() for item in observations
            if item.modality.lower() == "radar"
        ])
        observations = _preprocess_valid_radar_with_line_cann(
            observations, timestamps,
        )
        processed_radar = [
            item for item in observations if item.modality.lower() == "radar"
        ]
        processed_values = np.asarray([
            item.measurement for item in processed_radar
        ])
        radar_valid = np.asarray([
            item.valid_flag for item in processed_radar
        ], dtype=bool)
        delta = processed_values[radar_valid] - raw_radar[radar_valid]
        radar_preprocess_diagnostics = {
            "range_rms_change_m": float(np.sqrt(np.mean(delta[:, 0] ** 2))),
            "range_max_abs_change_m": float(np.max(np.abs(delta[:, 0]))),
            "range_rate_rms_change_mps": float(
                np.sqrt(np.mean(delta[:, 1] ** 2))
            ),
            "range_rate_max_abs_change_mps": float(
                np.max(np.abs(delta[:, 1]))
            ),
            "reanchor_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "radar_reanchored", False,
                )) for item in processed_radar if item.valid_flag
            )),
            "range_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "range_substituted", False,
                )) for item in processed_radar if item.valid_flag
            )),
            "range_rate_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "range_rate_substituted", False,
                )) for item in processed_radar if item.valid_flag
            )),
        }
    infrared_preprocess_diagnostics = None
    if adaptive_cann_preprocess_ir:
        observations = _preprocess_valid_infrared_azimuth_with_cann(
            observations, timestamps, method="bias_adaptive_cann",
        )
    elif hybrid_cann_preprocess_ir:
        observations = _preprocess_valid_infrared_azimuth_with_cann(
            observations, timestamps, method="hybrid_ring_line_cann",
        )
        processed_infrared = [
            item for item in observations
            if item.modality.lower() in {"ir", "infrared"}
        ]
        infrared_preprocess_diagnostics = {
            "azimuth_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "azimuth_substituted", False,
                )) for item in processed_infrared if item.valid_flag
            )),
            "elevation_substitution_count": int(sum(
                bool(item.metadata.get("cann_diagnostics", {}).get(
                    "elevation_substituted", False,
                )) for item in processed_infrared if item.valid_flag
            )),
        }
    module_input = build_module_inputs(
        scenario=scenario,
        observations_by_node={"sat_01": observations},
        initial_error_by_node={"sat_01": np.array([50., -40., 30., .05, -.04, .03])},
        ci_grid_points=31,
    )["sat_01"]
    if enable_cann:
        module_input.config["brain_inspired"] = {
            "cann": {
                "enabled": True,
                "cue_interval_samples": cue_interval_samples,
            },
        }
    history = StateAwarenessModule().run_history(module_input)
    estimate_eci = observer_track.state_history_eci + history.fused_state_history
    truth_eci = scenario.target_trajectory.state_history_eci
    # Use the same estimator-derived fixed frame as the integrated sidecar so
    # truth and decoded phase are compared in one coordinate convention.
    cann = history.cann_sidecar_history
    truth_phase = None
    phase_error = None
    if cann is not None:
        frame = OrbitalPlaneFrame.from_state_eci(estimate_eci[0])
        truth_phase = run_orbital_phase_sidecar(
            timestamps=timestamps, state_history_eci=truth_eci, frame=frame,
            source_id="target:truth",
        ).source_phase
        phase_error = (
            (cann.decoded_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi
        )
    position_error = np.linalg.norm(estimate_eci[:, :3] - truth_eci[:, :3], axis=1)
    velocity_error = np.linalg.norm(estimate_eci[:, 3:] - truth_eci[:, 3:], axis=1)
    def _window_rmse(values, mask):
        return float(np.sqrt(np.mean(values[mask] ** 2))) if np.any(mask) else None

    recovery_start = max(
        (end for _, end in normalized_outage_windows.values()), default=-np.inf,
    )
    recovery_window = timestamps > recovery_start
    summary = {
        "outage_modalities": sorted(requested_outages),
        "outage_windows": {
            name: [start, end]
            for name, (start, end) in sorted(normalized_outage_windows.items())
        },
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_rmse_outage_m": _window_rmse(position_error, outage_window),
        "position_rmse_by_outage_m": {
            name: _window_rmse(position_error, outage_window_by_modality[name])
            for name in sorted(requested_outages)
        },
        "position_rmse_recovery_m": _window_rmse(position_error, recovery_window),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "velocity_rmse_outage_mps": _window_rmse(velocity_error, outage_window),
        "velocity_rmse_by_outage_mps": {
            name: _window_rmse(velocity_error, outage_window_by_modality[name])
            for name in sorted(requested_outages)
        },
        "velocity_rmse_recovery_mps": _window_rmse(velocity_error, recovery_window),
        "cann_enabled": bool(enable_cann),
        "adaptive_cann_preprocess_ir": bool(adaptive_cann_preprocess_ir),
        "hybrid_cann_preprocess_ir": bool(hybrid_cann_preprocess_ir),
        "radar_cann_preprocess": bool(radar_cann_preprocess),
        "radar_fault_mode": radar_fault_mode,
        "optical_cann_preprocess": bool(optical_cann_preprocess),
        "optical_fault_mode": optical_fault_mode,
        "optical_preprocess_diagnostics": optical_preprocess_diagnostics,
        "infrared_fault_mode": infrared_fault_mode,
        "infrared_preprocess_diagnostics": infrared_preprocess_diagnostics,
        "radar_preprocess_diagnostics": radar_preprocess_diagnostics,
        "cann_phase_rmse_deg": (
            float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
            if phase_error is not None else None
        ),
        "cann_max_phase_error_deg": (
            float(np.rad2deg(np.max(np.abs(phase_error))))
            if phase_error is not None else None
        ),
        "cann_valid_fraction": float(np.mean(cann.valid)) if cann is not None else None,
        "optical_valid_count": int(np.count_nonzero(history.measurement_valid_history["opt"])),
        "infrared_valid_count": int(np.count_nonzero(history.measurement_valid_history["ir"])),
        "radar_valid_count": int(np.count_nonzero(history.measurement_valid_history["rad"])),
    }
    return {
        "timestamps": timestamps, "position_error_m": position_error,
        "velocity_error_mps": velocity_error, "truth_phase": truth_phase,
        "estimated_state_history_eci": estimate_eci,
        "truth_state_history_eci": truth_eci,
        "cann": cann,
        "available": np.logical_and.reduce(tuple(available_by_modality.values())),
        "available_by_modality": available_by_modality,
        "measurement_valid_by_modality": {
            name: np.asarray(flags, dtype=bool).copy()
            for name, flags in history.measurement_valid_history.items()
        },
        "summary": summary,
    }


def _preprocess_valid_infrared_azimuth_with_cann(
    observations, timestamps, *, method="bias_adaptive_cann",
):
    """Causally smooth valid IR azimuths without creating outage measurements."""
    infrared = [
        item for item in observations
        if item.modality.lower() in {"ir", "infrared"}
    ]
    if len(infrared) != len(timestamps):
        raise ValueError("Expected one infrared observation per timestamp.")
    hint = np.array([item.measurement[0] for item in infrared], dtype=float)
    elevation_hint = np.array(
        [item.measurement[1] for item in infrared], dtype=float,
    )
    available = np.array([item.valid_flag for item in infrared], dtype=bool)
    valid_indices = np.flatnonzero(available)
    if not valid_indices.size:
        return observations
    rate = np.zeros(len(timestamps), dtype=float)
    last_rate = 0.0
    previous = int(valid_indices[0])
    for index in range(previous + 1, len(timestamps)):
        if available[index]:
            elapsed = float(timestamps[index] - timestamps[previous])
            last_rate = float(_difference(hint[index], hint[previous]) / elapsed)
            previous = index
        rate[index - 1] = last_rate
    rate[-1] = last_rate
    diagnostics_by_index = [{} for _ in timestamps]
    if method == "bias_adaptive_cann":
        phase, _ = _cann_tracker(
            np.asarray(timestamps, dtype=float), hint[valid_indices[0]], rate,
            hint % (2.0 * np.pi), available, np.deg2rad(3.0),
            rate_bias_gain=0.1,
        )
    elif method == "hybrid_ring_line_cann":
        phase, diagnostics_by_index = _hybrid_ring_line_ir_azimuth(
            timestamps, hint, rate, available,
        )
        elevation, elevation_diagnostics = _line_cann_ir_elevation(
            timestamps, elevation_hint, available,
        )
        diagnostics_by_index = [
            {**azimuth, **el}
            for azimuth, el in zip(
                diagnostics_by_index, elevation_diagnostics,
            )
        ]
    else:
        raise ValueError(f"Unknown infrared CANN preprocessor: {method}")
    replacements = {}
    for index, item in enumerate(infrared):
        if not item.valid_flag:
            continue
        measurement = item.measurement.copy()
        measurement[0] = _difference(phase[index], 0.0)
        if method == "hybrid_ring_line_cann":
            measurement[1] = elevation[index]
        replacements[id(item)] = preprocess_observation(
            item, measurement=measurement,
            diagnostics={
                "azimuth_preprocessor": method,
                **diagnostics_by_index[index],
            },
        )
    return [replacements.get(id(item), item) for item in observations]


def _hybrid_ring_line_ir_azimuth(timestamps, hint, rate, available):
    del rate
    config = CoupledRingLineCANNConfig(
        bias_anchor_mode="hybrid_dual", minimum_bias_baseline=120.0,
        anchor_agreement_scale=np.deg2rad(0.004),
        line=LineCANNConfig(
            minimum_value=np.deg2rad(-0.05),
            maximum_value=np.deg2rad(0.05),
            tuning_width=np.deg2rad(0.003), cue_gain=0.2,
        ),
    )
    observer = CoupledRingLineCANN(config)
    first_valid = int(np.flatnonzero(available)[0])
    first = observer.initialize(
        phase=hint[first_valid], timestamp=float(timestamps[first_valid]),
    )
    phase = np.full(len(timestamps), hint[first_valid], dtype=float)
    diagnostics = [{} for _ in timestamps]
    phase[first_valid] = first.decoded_phase
    held_rate = 0.0
    last_trusted_timestamp = float(timestamps[first_valid])
    last_trusted_phase = float(hint[first_valid])
    for index in range(first_valid + 1, len(timestamps)):
        use = bool(available[index])
        gap = float(timestamps[index] - last_trusted_timestamp)
        reanchored = bool(use and gap > 20.0)
        if reanchored:
            output = observer.initialize(
                phase=float(hint[index]), timestamp=float(timestamps[index]),
            )
            held_rate = 0.0
            last_trusted_timestamp = float(timestamps[index])
            last_trusted_phase = float(hint[index])
        else:
            output = observer.update(
                timestamp=float(timestamps[index]),
                measured_phase_rate=held_rate,
                phase_hint=float(hint[index]) if use else None,
                phase_hint_valid=use,
            )
            if use and output.cue_applied:
                held_rate = float(_difference(
                    hint[index], last_trusted_phase,
                ) / max(gap, 1.0e-12))
                last_trusted_timestamp = float(timestamps[index])
                last_trusted_phase = float(hint[index])
        phase[index] = output.decoded_phase
        diagnostics[index] = {
            "decoded_rate_bias": float(output.decoded_rate_bias),
            "bias_observation_count": int(output.bias_observation_count),
            "long_anchor_trusted": output.long_anchor_trusted,
            "azimuth_substituted": bool(
                use and not reanchored and not output.cue_applied
            ),
            "azimuth_reanchored": reanchored,
        }
    return phase, diagnostics


def _line_cann_ir_elevation(timestamps, hint, available):
    valid_indices = np.flatnonzero(available)
    first_valid = int(valid_indices[0])
    observer = LineCANN(LineCANNConfig(
        num_neurons=361,
        minimum_value=-0.5 * np.pi, maximum_value=0.5 * np.pi,
        tuning_width=np.deg2rad(2.0), cue_gain=1.0,
    ))
    first = observer.reset(
        float(hint[first_valid]), timestamp=float(timestamps[first_valid]),
    )
    elevation = np.full(len(timestamps), hint[first_valid], dtype=float)
    elevation[first_valid] = first.decoded_value
    diagnostics = [{} for _ in timestamps]
    last_valid_timestamp = float(timestamps[first_valid])
    maximum_propagation_duration = 20.0
    innovation_gate = np.deg2rad(3.0)
    held_rate = 0.0
    last_trusted_hint = float(hint[first_valid])
    for index in range(first_valid + 1, len(timestamps)):
        dt = float(timestamps[index] - timestamps[index - 1])
        output = observer.step(held_rate, dt)
        substituted = False
        if available[index]:
            gap = float(timestamps[index] - last_valid_timestamp)
            innovation = float(hint[index] - output.decoded_value)
            reanchored = bool(
                gap > maximum_propagation_duration
                or output.saturated_at_boundary
            )
            if reanchored:
                output = observer.reset(
                    float(hint[index]), timestamp=float(timestamps[index]),
                )
                held_rate = 0.0
                last_trusted_hint = float(hint[index])
            elif abs(innovation) > innovation_gate:
                substituted = True
            else:
                output = observer.apply_value_cue(float(hint[index]))
                held_rate = float(
                    (hint[index] - last_trusted_hint) / max(gap, 1.0e-12)
                )
                last_trusted_hint = float(hint[index])
            if not substituted:
                last_valid_timestamp = float(timestamps[index])
        else:
            reanchored = False
        elevation[index] = output.decoded_value
        diagnostics[index] = {
            "elevation_bump_concentration": float(output.bump_concentration),
            "elevation_bump_width": float(output.bump_width),
            "elevation_saturated_at_boundary": bool(
                output.saturated_at_boundary
            ),
            "elevation_reanchored": reanchored,
            "elevation_substituted": substituted,
        }
    return elevation, diagnostics


def _inject_infrared_faults(observations, timestamps, *, mode):
    if mode != "impulsive":
        raise ValueError(f"Unknown infrared fault mode: {mode}")
    infrared = [
        item for item in observations
        if item.modality.lower() in {"ir", "infrared"}
    ]
    requested_times = (300.0, 450.0, 1350.0, 1500.0)
    signs = (1.0, -1.0, 1.0, -1.0)
    for fault_time, sign in zip(requested_times, signs):
        index = int(np.argmin(np.abs(np.asarray(timestamps) - fault_time)))
        if abs(float(timestamps[index]) - fault_time) > 0.5:
            continue
        item = infrared[index]
        if not item.valid_flag:
            continue
        item.measurement = item.measurement.copy()
        item.measurement += sign * np.deg2rad([5.0, 5.0])
        item.metadata = {
            **item.metadata, "injected_infrared_fault": "impulsive",
        }


def _preprocess_valid_radar_with_line_cann(observations, timestamps):
    radar = [item for item in observations if item.modality.lower() == "radar"]
    if len(radar) != len(timestamps):
        raise ValueError("Expected one radar observation per timestamp.")
    measurement = np.asarray([item.measurement for item in radar], dtype=float)
    available = np.asarray([item.valid_flag for item in radar], dtype=bool)
    valid_indices = np.flatnonzero(available)
    if not valid_indices.size:
        return observations
    filtered, diagnostics = _radar_range_rate_line_cann(
        timestamps, measurement, available,
    )
    replacements = {}
    for index, item in enumerate(radar):
        if not item.valid_flag:
            continue
        replacements[id(item)] = preprocess_observation(
            item, measurement=filtered[index],
            diagnostics={
                "radar_preprocessor": "range_rate_dual_line_cann",
                **diagnostics[index],
            },
        )
    return [replacements.get(id(item), item) for item in observations]


def _radar_range_rate_line_cann(timestamps, measurement, available):
    timestamps = np.asarray(timestamps, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    available = np.asarray(available, dtype=bool)
    if measurement.shape != (timestamps.size, 2):
        raise ValueError("Radar Line CANN expects range/range-rate pairs.")
    valid_indices = np.flatnonzero(available)
    if not valid_indices.size:
        return measurement.copy(), [{} for _ in timestamps]
    range_cann = LineCANN(LineCANNConfig(
        num_neurons=501, minimum_value=0.0, maximum_value=50_000_000.0,
        tuning_width=400_000.0, cue_gain=0.8,
    ))
    rate_cann = LineCANN(LineCANNConfig(
        num_neurons=401, minimum_value=-20_000.0, maximum_value=20_000.0,
        tuning_width=400.0, cue_gain=1.0,
    ))
    first_valid = int(valid_indices[0])
    range_output = range_cann.reset(
        measurement[first_valid, 0], timestamp=timestamps[first_valid],
    )
    rate_output = rate_cann.reset(
        measurement[first_valid, 1], timestamp=timestamps[first_valid],
    )
    filtered = measurement.copy()
    filtered[first_valid] = [
        range_output.decoded_value, rate_output.decoded_value,
    ]
    diagnostics = [{} for _ in timestamps]
    last_valid_timestamp = float(timestamps[first_valid])
    for index in range(first_valid + 1, timestamps.size):
        dt = float(timestamps[index] - timestamps[index - 1])
        range_output = range_cann.step(rate_output.decoded_value, dt)
        rate_output = rate_cann.step(0.0, dt)
        reanchored = False
        range_substituted = False
        rate_substituted = False
        if available[index]:
            gap = float(timestamps[index] - last_valid_timestamp)
            range_innovation = float(
                measurement[index, 0] - range_output.decoded_value
            )
            rate_innovation = float(
                measurement[index, 1] - rate_output.decoded_value
            )
            reanchored = bool(
                gap > 20.0 or range_output.saturated_at_boundary
                or rate_output.saturated_at_boundary
            )
            if reanchored:
                range_output = range_cann.reset(
                    measurement[index, 0], timestamp=timestamps[index],
                )
                rate_output = rate_cann.reset(
                    measurement[index, 1], timestamp=timestamps[index],
                )
            else:
                range_substituted = abs(range_innovation) > 300.0
                rate_substituted = abs(rate_innovation) > 0.5
                predicted_range = range_output.decoded_value
                predicted_rate = rate_output.decoded_value
                if not range_substituted:
                    range_output = range_cann.apply_value_cue(
                        measurement[index, 0],
                    )
                if not rate_substituted:
                    rate_output = rate_cann.apply_value_cue(
                        measurement[index, 1],
                    )
                filtered[index] = [
                    predicted_range if range_substituted
                    else measurement[index, 0],
                    predicted_rate if rate_substituted
                    else measurement[index, 1],
                ]
            last_valid_timestamp = float(timestamps[index])
        diagnostics[index] = {
            "radar_reanchored": reanchored,
            "range_substituted": range_substituted,
            "range_rate_substituted": rate_substituted,
            "range_bump_concentration": float(
                range_output.bump_concentration
            ),
            "range_rate_bump_concentration": float(
                rate_output.bump_concentration
            ),
        }
    return filtered, diagnostics


def _inject_radar_faults(observations, timestamps, *, mode):
    if mode != "impulsive":
        raise ValueError(f"Unknown radar fault mode: {mode}")
    radar = [item for item in observations if item.modality.lower() == "radar"]
    if len(radar) != len(timestamps):
        raise ValueError("Expected one radar observation per timestamp.")
    requested_times = (300.0, 450.0, 1350.0, 1500.0)
    signs = (1.0, -1.0, 1.0, -1.0)
    for fault_time, sign in zip(requested_times, signs):
        index = int(np.argmin(np.abs(np.asarray(timestamps) - fault_time)))
        if abs(float(timestamps[index]) - fault_time) > 0.5:
            continue
        item = radar[index]
        if not item.valid_flag:
            continue
        item.measurement = item.measurement.copy()
        item.measurement[0] += sign * 3_000.0
        item.measurement[1] += sign * 5.0
        item.metadata = {
            **item.metadata, "injected_radar_fault": "impulsive",
        }


def _preprocess_valid_optical_with_plane_cann(observations, timestamps):
    optical = [
        item for item in observations if item.modality.lower() == "optical"
    ]
    if len(optical) != len(timestamps):
        raise ValueError("Expected one optical observation per timestamp.")
    measurement = np.asarray([item.measurement for item in optical], dtype=float)
    available = np.asarray([item.valid_flag for item in optical], dtype=bool)
    if not np.any(available):
        return observations
    filtered, diagnostics = _optical_uv_plane_cann(
        timestamps, measurement, available,
    )
    replacements = {}
    for index, item in enumerate(optical):
        if not item.valid_flag:
            continue
        replacements[id(item)] = preprocess_observation(
            item, measurement=filtered[index],
            diagnostics={
                "optical_preprocessor": "fault_aware_plane_cann",
                **diagnostics[index],
            },
        )
    return [replacements.get(id(item), item) for item in observations]


def _optical_uv_plane_cann(timestamps, measurement, available):
    timestamps = np.asarray(timestamps, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    available = np.asarray(available, dtype=bool)
    axis = LineCANNConfig(
        num_neurons=401, minimum_value=-2_000.0, maximum_value=2_000.0,
        tuning_width=40.0, cue_gain=1.0,
    )
    observer = PlaneCANN(PlaneCANNConfig(x_axis=axis, y_axis=axis))
    first_valid = int(np.flatnonzero(available)[0])
    output = observer.reset(
        measurement[first_valid], timestamp=timestamps[first_valid],
    )
    filtered = measurement.copy()
    diagnostics = [{} for _ in timestamps]
    velocity = np.zeros(2, dtype=float)
    last_trusted_measurement = measurement[first_valid].copy()
    last_trusted_timestamp = float(timestamps[first_valid])
    for index in range(first_valid + 1, timestamps.size):
        dt = float(timestamps[index] - timestamps[index - 1])
        output = observer.step(velocity, dt)
        reanchored = False
        substituted = np.zeros(2, dtype=bool)
        if available[index]:
            gap = float(timestamps[index] - last_trusted_timestamp)
            innovation = measurement[index] - output.decoded_position
            stable_motion = bool(np.all(np.abs(velocity) * dt < 0.1))
            inside_stable_field = bool(
                np.all(np.abs(measurement[index]) <= 10.0)
                and np.all(np.abs(output.decoded_position) <= 10.0)
            )
            reanchored = bool(
                gap > 20.0 or output.saturated_at_boundary
                or not stable_motion or not inside_stable_field
            )
            if reanchored:
                output = observer.reset(
                    measurement[index], timestamp=timestamps[index],
                )
                filtered[index] = measurement[index]
            else:
                joint_anomaly = bool(np.all(np.abs(innovation) > 0.05))
                substituted[:] = joint_anomaly
                filtered[index] = np.where(
                    substituted, output.decoded_position, measurement[index],
                )
                cue = np.where(
                    substituted, output.decoded_position, measurement[index],
                )
                output = observer.apply_position_cue(cue, cue_gain=1.0)
            if not np.any(substituted):
                velocity = (
                    measurement[index] - last_trusted_measurement
                ) / max(gap, 1.0e-12)
                last_trusted_measurement = measurement[index].copy()
                last_trusted_timestamp = float(timestamps[index])
        diagnostics[index] = {
            "optical_reanchored": reanchored,
            "u_substituted": bool(substituted[0]),
            "v_substituted": bool(substituted[1]),
            "plane_bump_concentration": float(output.bump_concentration),
        }
    return filtered, diagnostics


def _inject_optical_faults(observations, timestamps, *, mode):
    if mode != "impulsive":
        raise ValueError(f"Unknown optical fault mode: {mode}")
    optical = [
        item for item in observations if item.modality.lower() == "optical"
    ]
    requested_times = (1250.0, 1350.0, 1500.0, 1650.0)
    signs = (1.0, -1.0, 1.0, -1.0)
    for fault_time, sign in zip(requested_times, signs):
        index = int(np.argmin(np.abs(np.asarray(timestamps) - fault_time)))
        if abs(float(timestamps[index]) - fault_time) > 0.5:
            continue
        item = optical[index]
        if not item.valid_flag:
            continue
        item.measurement = item.measurement.copy()
        item.measurement += sign * np.array([0.2, 0.4])
        item.metadata = {
            **item.metadata, "injected_optical_fault": "impulsive",
        }


def write_single_satellite_cann_results(result, output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8",
    )
    t = result["timestamps"]
    cann = result["cann"]
    phase_error = None
    if cann is not None:
        phase_error = (
            (cann.decoded_phase - result["truth_phase"] + np.pi) % (2*np.pi) - np.pi
        )
    panel_count = 3 if cann is not None else 2
    fig, axes = plt.subplots(panel_count, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, result["position_error_m"], label="Federated-CI position error")
    axes[0].set_ylabel("Position error (m)"); axes[0].grid(True); axes[0].legend()
    if cann is not None:
        axes[1].plot(t, np.rad2deg(phase_error), label="CANN phase error", color="tab:orange")
        axes[1].set_ylabel("Phase error (deg)"); axes[1].grid(True); axes[1].legend()
        axes[2].plot(t, cann.bump_concentration, label="Bump concentration")
        axes[2].plot(t, cann.bump_width, label="Bump width (rad)")
        axes[2].set_ylabel("CANN diagnostic"); axes[2].grid(True); axes[2].legend()
    else:
        axes[1].plot(t, result["velocity_error_mps"], label="Federated-CI velocity error",
                     color="tab:green")
        axes[1].set_ylabel("Velocity error (m/s)"); axes[1].grid(True); axes[1].legend()
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.fill_between(t, 0, 1, where=~result["available"], color="gray", alpha=.12,
                          transform=axis.get_xaxis_transform(), label=None)
    suffix = "with passive CANN" if cann is not None else "original baseline (no CANN)"
    fig.suptitle(f"Single-satellite three-modal Federated-CI: {suffix}")
    fig.tight_layout()
    figure_path = output / "overview.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    return {"summary": summary_path, "figure": figure_path}
