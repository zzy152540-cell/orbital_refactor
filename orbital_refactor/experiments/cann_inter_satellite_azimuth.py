from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_inspired.passive_phase_observer import PassiveRingCANNObserver, PeriodicStateInput
from brain_inspired.coupled_ring_line_cann import CoupledRingLineCANN
from orbital_core.coordinates import build_rtn_quaternion, state_eci_to_spri
from orbital_core.measurements import h_ir_spri, h_optical_spri
from scenarios.measurement_visibility import VisibilityConfig, evaluate_inter_satellite_visibility
from scenarios.walker_scenario import WalkerDeltaConfig, generate_walker_delta_scenario


@dataclass(frozen=True)
class InterSatelliteAzimuthResult:
    observer_id: str
    target_id: str
    timestamps: np.ndarray
    truth_azimuth: np.ndarray
    hint: np.ndarray
    hint_available: np.ndarray
    hold_phase: np.ndarray
    integrated_phase: np.ndarray
    complementary_phase: np.ndarray
    pll_phase: np.ndarray
    circular_kalman_phase: np.ndarray
    cann_phase: np.ndarray
    adaptive_cann_phase: np.ndarray
    coupled_ring_line_phase: np.ndarray
    coupled_ring_line_rate_bias: np.ndarray
    cann_concentration: np.ndarray
    adaptive_cann_concentration: np.ndarray
    rmse_deg_by_mode: dict[str, float]
    outage_rmse_deg_by_mode: dict[str, float]
    reacquisition_time_s_by_mode: dict[str, float | None]
    boundary_crossing_count: int
    outage_window: tuple[float, float]
    sensor_azimuth_offset_deg: float


def run_inter_satellite_azimuth_benchmark(
    *, duration: float = 600.0, dt: float = 2.0, seed: int = 0,
    outage_window: tuple[float, float] = (200.0, 400.0),
    rotate_sensor_frame_to_boundary: bool = False,
    pll_kp: float = 0.5,
    pll_ki: float = 0.05,
    kalman_phase_process_std_deg: float = 0.002,
    kalman_bias_process_std_deg_s: float = 0.0005,
    adaptive_cann_rate_bias_gain: float = 0.1,
) -> InterSatelliteAzimuthResult:
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    config = WalkerDeltaConfig(
        total_satellites=20, plane_count=10, phasing=1,
        semi_major_axis=6_978_137.0, eccentricity=0.001,
        inclination=np.deg2rad(53.0),
    )
    scenario = generate_walker_delta_scenario(timestamps=times, config=config)
    observer, target, truth, geometry_visible = _select_link(scenario, times)
    sensor_offset = 0.0
    if rotate_sensor_frame_to_boundary:
        unwrapped = np.unwrap(truth)
        sensor_offset = -0.5 * (float(np.min(unwrapped)) + float(np.max(unwrapped)))
        truth = (truth + sensor_offset) % (2.0 * np.pi)
    rng = np.random.default_rng(seed)
    truth_unwrapped = np.unwrap(truth)
    truth_rate = np.gradient(truth_unwrapped, times)
    measured_rate = (
        truth_rate + np.deg2rad(0.003)
        + rng.normal(0.0, np.deg2rad(0.01), times.size)
    )
    scheduled = np.arange(times.size) % 5 == 0
    available = geometry_visible & scheduled & ~(
        (times >= outage_window[0]) & (times <= outage_window[1])
    )
    hint = np.full(times.size, np.nan)
    ir_noise = rng.normal(0.0, np.deg2rad(0.05), times.size)
    optical_noise = rng.normal(0.0, np.deg2rad(0.02), times.size)
    hint[available] = _weighted_circular_mean(
        (truth[available] + ir_noise[available]) % (2.0 * np.pi),
        (truth[available] + optical_noise[available]) % (2.0 * np.pi),
        1.0 / np.deg2rad(0.05) ** 2, 1.0 / np.deg2rad(0.02) ** 2,
    )
    indices = np.flatnonzero(available)
    if indices.size >= 4:
        hint[indices[len(indices) // 3]] += np.deg2rad(5.0)
        hint[indices[-2]] -= np.deg2rad(5.0)
        hint %= 2.0 * np.pi

    initial = truth[0]
    integrated = (initial + _left_integral(measured_rate, dt)) % (2.0 * np.pi)
    hold = _measurement_hold(initial, hint, available)
    complementary = _gated_complementary(
        initial, measured_rate, dt, hint, available, np.deg2rad(3.0),
    )
    pll = _gated_pll(
        initial, measured_rate, dt, hint, available, np.deg2rad(3.0),
        kp=pll_kp, ki=pll_ki,
    )
    circular_kalman = _circular_kalman(
        initial, measured_rate, dt, hint, available, np.deg2rad(3.0),
        phase_process_std_deg=kalman_phase_process_std_deg,
        bias_process_std_deg_s=kalman_bias_process_std_deg_s,
    )
    cann_phase, concentration = _cann_tracker(
        times, initial, measured_rate, hint, available, np.deg2rad(3.0),
    )
    adaptive_cann_phase, adaptive_concentration = _cann_tracker(
        times, initial, measured_rate, hint, available, np.deg2rad(3.0),
        rate_bias_gain=adaptive_cann_rate_bias_gain,
    )
    coupled_phase, coupled_rate_bias = _coupled_ring_line_tracker(
        times, initial, measured_rate, hint, available,
    )
    phases = {
        "measurement_hold": hold, "ordinary_integration": integrated,
        "gated_complementary": complementary, "gated_cann": cann_phase,
        "bias_adaptive_cann": adaptive_cann_phase,
        "coupled_ring_line_cann": coupled_phase,
        "gated_pll": pll, "circular_kalman": circular_kalman,
    }
    errors = {name: _difference(value, truth) for name, value in phases.items()}
    outage = (times >= outage_window[0]) & (times <= outage_window[1])
    return InterSatelliteAzimuthResult(
        observer_id=observer, target_id=target, timestamps=times,
        truth_azimuth=truth, hint=hint, hint_available=available,
        hold_phase=hold, integrated_phase=integrated,
        complementary_phase=complementary, pll_phase=pll,
        circular_kalman_phase=circular_kalman, cann_phase=cann_phase,
        adaptive_cann_phase=adaptive_cann_phase,
        coupled_ring_line_phase=coupled_phase,
        coupled_ring_line_rate_bias=coupled_rate_bias,
        cann_concentration=concentration,
        adaptive_cann_concentration=adaptive_concentration,
        rmse_deg_by_mode={name: float(np.rad2deg(np.sqrt(np.mean(error**2))))
                          for name, error in errors.items()},
        outage_rmse_deg_by_mode={name: float(np.rad2deg(np.sqrt(np.mean(error[outage]**2))))
                                 for name, error in errors.items()},
        reacquisition_time_s_by_mode={name: _reacquisition_time(
            times, error, outage_window[1], np.deg2rad(0.5),
        ) for name, error in errors.items()},
        boundary_crossing_count=int(np.sum(np.abs(np.diff(truth)) > np.pi)),
        outage_window=(float(outage_window[0]), float(outage_window[1])),
        sensor_azimuth_offset_deg=float(np.rad2deg(sensor_offset)),
    )


def write_inter_satellite_azimuth_summary(result, output_path):
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "observer_id": result.observer_id, "target_id": result.target_id,
        "rmse_deg_by_mode": result.rmse_deg_by_mode,
        "outage_rmse_deg_by_mode": result.outage_rmse_deg_by_mode,
        "reacquisition_time_s_by_mode": result.reacquisition_time_s_by_mode,
        "available_hint_count": int(result.hint_available.sum()),
        "boundary_crossing_count": result.boundary_crossing_count,
        "outage_window": result.outage_window,
        "sensor_azimuth_offset_deg": result.sensor_azimuth_offset_deg,
    }, indent=2), encoding="utf-8")
    return output


def generate_inter_satellite_azimuth_figure(result, output_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phases = {"hold": result.hold_phase, "integration": result.integrated_phase,
              "complementary": result.complementary_phase,
              "PLL": result.pll_phase, "circular Kalman": result.circular_kalman_phase,
              "CANN": result.cann_phase,
              "adaptive CANN": result.adaptive_cann_phase,
              "Ring-Line CANN": result.coupled_ring_line_phase}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    truth = np.unwrap(result.truth_azimuth)
    axes[0, 0].plot(result.timestamps, np.rad2deg(truth), "k--", label="truth")
    for name, phase in phases.items():
        axes[0, 0].plot(result.timestamps, np.rad2deg(_align(phase, truth[0])), label=name)
        axes[0, 1].plot(result.timestamps, np.rad2deg(_difference(phase, result.truth_azimuth)), label=name)
    axes[0, 1].axvspan(*result.outage_window, color="gray", alpha=.15,
                       label="forced outage")
    visible = np.flatnonzero(result.hint_available)
    axes[1, 0].scatter(result.timestamps[visible], np.rad2deg(_difference(
        result.hint[visible], result.truth_azimuth[visible])), s=18)
    axes[1, 1].plot(result.timestamps, result.cann_concentration)
    axes[0, 0].set(title="RTN inter-satellite azimuth", ylabel="unwrapped azimuth (deg)")
    axes[0, 1].set(title="Tracking error", ylabel="error (deg)")
    axes[1, 0].set(title="Fused IR/optical cue error", ylabel="cue error (deg)")
    axes[1, 1].set(title="CANN bump concentration", ylabel="concentration")
    for axis in axes.flat:
        axis.set_xlabel("time (s)"); axis.grid(alpha=.3)
    axes[0, 0].legend(); axes[0, 1].legend()
    figure.suptitle(f"Walker link {result.observer_id} -> {result.target_id}", fontsize=14)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170); plt.close(figure); return output


def _select_link(scenario, times):
    limits = VisibilityConfig(maximum_range=7_000e3)
    best = None
    for observer in scenario.node_ids:
        for target in scenario.node_ids:
            if observer == target: continue
            azimuth, visible = [], []
            for index in range(times.size):
                source = scenario.truth_state_history_by_node[observer][index]
                destination = scenario.truth_state_history_by_node[target][index]
                relative = destination - source
                quaternion = build_rtn_quaternion(source)
                spri = state_eci_to_spri(relative, quaternion)
                ir = h_ir_spri(relative, quaternion)[0]
                front = spri[2] > 1.0
                if front:
                    optical = h_optical_spri(relative, quaternion)
                    bearing = np.arctan2(optical[1], optical[0])
                    front = abs(_difference(bearing, ir)) < 1e-10
                physical = evaluate_inter_satellite_visibility(source, destination, limits).visible
                azimuth.append(ir % (2.0 * np.pi)); visible.append(front and physical)
            azimuth, visible = np.asarray(azimuth), np.asarray(visible)
            crossings = int(np.sum(np.abs(np.diff(azimuth)) > np.pi))
            sufficiently_visible = int(visible.sum() >= 50)
            score = (
                int(crossings > 0 and sufficiently_visible),
                int(visible.sum()), crossings,
                float(np.ptp(np.unwrap(azimuth))),
            )
            if best is None or score > best[0]: best = (score, observer, target, azimuth, visible)
    if best is None or best[0][1] < 5:
        raise RuntimeError("No usable Walker optical/IR link found.")
    return best[1], best[2], best[3], best[4]


def _cann_tracker(
    times, initial, rate, hint, available, gate, *, cue_gain=0.05,
    rate_bias_gain=0.0, maximum_rate_bias_deg_s=0.05,
):
    """Track circular phase with an optional innovation-driven rate-bias loop.

    The outer loop is deliberately local to the CANN preprocessor.  A gated
    phase innovation is converted to an average rate correction over the time
    since the previous accepted cue.  It never modifies estimator state or
    covariance.
    """
    observer = PassiveRingCANNObserver(); first = observer.initialize(phase=initial, timestamp=times[0])
    phase, quality = [first.decoded_phase], [first.bump_concentration]
    rate_bias = 0.0
    last_accepted_cue_time = float(times[0])
    last_accepted_cue_phase = float(initial)
    integrated_measured_rate = 0.0
    maximum_rate_bias = np.deg2rad(maximum_rate_bias_deg_s)
    for index in range(1, times.size):
        interval = float(times[index] - times[index - 1])
        integrated_measured_rate += rate[index - 1] * interval
        step_rate = rate[index - 1] + rate_bias
        predicted = (phase[-1] + step_rate * interval) % (2*np.pi)
        use = bool(available[index] and abs(_difference(hint[index], predicted)) <= gate)
        output = observer.update(PeriodicStateInput(timestamp=times[index], phase_rate=step_rate,
            phase_hint=hint[index] if use else None, phase_hint_valid=use, cue_gain=cue_gain))
        if use and rate_bias_gain > 0.0:
            cue_interval = float(times[index] - last_accepted_cue_time)
            cue_prediction = (
                last_accepted_cue_phase + integrated_measured_rate
                + rate_bias * cue_interval
            ) % (2.0 * np.pi)
            innovation = _difference(hint[index], cue_prediction)
            rate_bias = np.clip(
                rate_bias + rate_bias_gain * innovation / cue_interval,
                -maximum_rate_bias, maximum_rate_bias,
            )
            last_accepted_cue_time = float(times[index])
            last_accepted_cue_phase = float(hint[index])
            integrated_measured_rate = 0.0
        phase.append(output.decoded_phase); quality.append(output.bump_concentration)
    return np.asarray(phase), np.asarray(quality)


def _coupled_ring_line_tracker(
    times, initial, rate, hint, available, config=None,
):
    observer = CoupledRingLineCANN() if config is None else CoupledRingLineCANN(config)
    first = observer.initialize(phase=initial, timestamp=times[0])
    phase = [first.decoded_phase]
    rate_bias = [first.decoded_rate_bias]
    for index in range(1, times.size):
        use = bool(available[index])
        output = observer.update(
            timestamp=times[index], measured_phase_rate=rate[index - 1],
            phase_hint=hint[index] if use else None,
            phase_hint_valid=use,
        )
        phase.append(output.decoded_phase)
        rate_bias.append(output.decoded_rate_bias)
    return np.asarray(phase), np.asarray(rate_bias)


def _gated_complementary(initial, rate, dt, hint, available, gate):
    phase=np.empty(rate.size); phase[0]=initial
    for i in range(1,rate.size):
        phase[i]=(phase[i-1]+rate[i-1]*dt)%(2*np.pi); innovation=_difference(hint[i],phase[i])
        if available[i] and abs(innovation)<=gate: phase[i]=(phase[i]+innovation)%(2*np.pi)
    return phase


def _gated_pll(initial, rate, dt, hint, available, gate, kp=0.5, ki=0.05):
    """Second-order circular PLL with gated phase innovations."""
    phase = np.empty(rate.size); phase[0] = initial
    rate_bias = 0.0
    for index in range(1, rate.size):
        phase[index] = (
            phase[index - 1] + (rate[index - 1] + rate_bias) * dt
        ) % (2.0 * np.pi)
        if available[index]:
            innovation = _difference(hint[index], phase[index])
            if abs(innovation) <= gate:
                phase[index] = (phase[index] + kp * innovation) % (2.0 * np.pi)
                rate_bias += ki * innovation / dt
    return phase


def _circular_kalman(
    initial, rate, dt, hint, available, gate, *,
    phase_process_std_deg=0.002, bias_process_std_deg_s=0.0005,
):
    """Two-state phase/rate-bias Kalman filter with wrapped innovations."""
    phase = np.empty(rate.size); phase[0] = initial
    state = np.array([initial, 0.0])
    covariance = np.diag(np.deg2rad([0.1, 0.01])) ** 2
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process = np.diag([
        np.deg2rad(phase_process_std_deg) ** 2,
        np.deg2rad(bias_process_std_deg_s) ** 2,
    ])
    measurement_variance = 1.0 / (
        1.0 / np.deg2rad(0.05) ** 2 + 1.0 / np.deg2rad(0.02) ** 2
    )
    observation = np.array([1.0, 0.0])
    for index in range(1, rate.size):
        state = transition @ state + np.array([rate[index - 1] * dt, 0.0])
        state[0] %= 2.0 * np.pi
        covariance = transition @ covariance @ transition.T + process
        if available[index]:
            innovation = float(_difference(hint[index], state[0]))
            innovation_variance = float(observation @ covariance @ observation + measurement_variance)
            if abs(innovation) <= gate:
                gain = covariance @ observation / innovation_variance
                state = state + gain * innovation
                state[0] %= 2.0 * np.pi
                covariance = (
                    np.eye(2) - np.outer(gain, observation)
                ) @ covariance
        phase[index] = state[0]
    return phase


def _measurement_hold(initial, hint, available):
    phase=np.empty(hint.size); phase[0]=initial
    for i in range(1,hint.size): phase[i]=hint[i] if available[i] else phase[i-1]
    return phase


def _weighted_circular_mean(first, second, w1, w2):
    return np.angle(w1*np.exp(1j*first)+w2*np.exp(1j*second))%(2*np.pi)
def _left_integral(rate, dt): return np.concatenate(([0.0],np.cumsum(rate[:-1]*dt)))
def _difference(actual, expected): return (np.asarray(actual)-np.asarray(expected)+np.pi)%(2*np.pi)-np.pi
def _align(phase, reference):
    value=np.unwrap(phase); return value-np.round((value[0]-reference)/(2*np.pi))*2*np.pi
def _reacquisition_time(times, error, outage_end, threshold):
    candidates=np.flatnonzero((times>outage_end)&(np.abs(error)<=threshold))
    return None if not candidates.size else float(times[candidates[0]]-outage_end)
