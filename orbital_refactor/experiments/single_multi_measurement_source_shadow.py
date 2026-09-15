"""Paired analytic versus shared-raw measurement-source shadow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MultimodalMeasurementSourceConfig,
    MultimodalSensorSimulationConfig,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from adapters.frontend_covariance_calibration import (
    calibration_table_from_error_budget as radar_optical_calibration_table,
)
from adapters.infrared_covariance_calibration import (
    calibration_table_from_error_budget as infrared_calibration_table,
)
from experiments.infrared_error_budget_audit import (
    DEFAULT_PROFILES as DEFAULT_INFRARED_PROFILES,
    run_infrared_error_budget_audit,
)
from experiments.radar_optical_error_budget_audit import (
    DEFAULT_OPTICAL_PROFILES,
    DEFAULT_RADAR_PROFILES,
    run_radar_optical_error_budget_audit,
)
from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)
from experiments.walker_raw_sensor_comparison import (
    _mean_modality_nis,
    _position_rmse,
    _run,
    _walker_case,
    replace_multimodal_messages_with_shared_frontend,
)


@dataclass(frozen=True)
class MeasurementSourceArm:
    scope: str
    source: str
    position_rmse_m: float
    valid_radar_count: int
    valid_infrared_count: int
    valid_optical_count: int
    radar_mean_nis: float | None
    infrared_mean_nis: float | None
    optical_mean_nis: float | None


@dataclass(frozen=True)
class MeasurementSourceShadowReport:
    duration_seconds: float
    dt_seconds: float
    seed: int
    single_observer_raan_deg: float
    single_all_modalities_observable: bool
    empirical_calibration_modalities: tuple[str, ...]
    shared_truth_and_initialization: bool
    production_default_unchanged: bool
    raw_frontend_nis_alignment_ready: bool
    interpretation: str
    arms: tuple[MeasurementSourceArm, ...]


def run_single_multi_measurement_source_shadow(
    *, duration=20.0, dt=2.0, seed=0,
    sensors=None,
    covariance_calibration_by_modality=None,
):
    sensors = sensors or MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            width=128, height=128, focal_length_x_pixels=40.0,
            focal_length_y_pixels=40.0, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
        ),
        infrared=InfraredCameraConfig(
            width=128, height=128, focal_length_x_pixels=300.0,
            focal_length_y_pixels=300.0, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
        ),
        radar=RadarRangeDopplerConfig(
            width=129, height=129, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
        ),
    )
    analytic_single = run_single_satellite_cann_comparison(
        duration=duration, dt=dt, seed=seed, enable_cann=False,
        observer_raan_deg=20.0,
    )
    raw_single = run_single_satellite_cann_comparison(
        duration=duration, dt=dt, seed=seed, enable_cann=False,
        observer_raan_deg=20.0,
        measurement_source_config=MultimodalMeasurementSourceConfig(
            source="raw_frontend", sensors=sensors,
            covariance_calibration_by_modality=(
                covariance_calibration_by_modality or {}
            ),
        ),
    )
    case = _walker_case(seed=seed, duration=duration, dt=dt)
    raw_messages = replace_multimodal_messages_with_shared_frontend(
        case["observations"], timestamps=case["timestamps"],
        truth_state_history_by_node=case["truth"], config=sensors,
        random_seed=3_000_000 + int(seed),
        covariance_calibration_by_modality=(
            covariance_calibration_by_modality
        ),
    )
    analytic_history = _run(case, case["observations"])
    raw_history = _run(case, raw_messages)
    arms = (
        _single_arm("analytic", analytic_single),
        _single_arm("raw_frontend", raw_single),
        _walker_arm("analytic", analytic_history, case["observations"], case),
        _walker_arm("raw_frontend", raw_history, raw_messages, case),
    )
    raw_walker = arms[-1]
    single_arms = tuple(arm for arm in arms if arm.scope == "single")
    all_single_modalities_observable = all(
        min(
            arm.valid_radar_count,
            arm.valid_infrared_count,
            arm.valid_optical_count,
        ) > 0
        for arm in single_arms
    )
    raw_nis = (
        raw_walker.radar_mean_nis,
        raw_walker.infrared_mean_nis,
        raw_walker.optical_mean_nis,
    )
    nis_alignment_ready = all(
        value is not None and 0.5 <= value <= 5.0 for value in raw_nis
    )
    return MeasurementSourceShadowReport(
        duration_seconds=float(duration), dt_seconds=float(dt), seed=int(seed),
        single_observer_raan_deg=20.0,
        single_all_modalities_observable=all_single_modalities_observable,
        empirical_calibration_modalities=tuple(sorted(
            str(name).upper()
            for name in (covariance_calibration_by_modality or {})
        )),
        shared_truth_and_initialization=True, production_default_unchanged=True,
        raw_frontend_nis_alignment_ready=nis_alignment_ready,
        interpretation=(
            "This is an implementation-source shadow, not a sensor-performance "
            "claim. Analytic and raw arms share scenario truth and initialization "
            "but represent different noise-generation mechanisms. The raw source "
            "must not replace the analytic default until its empirical residual "
            "covariance is aligned with the filter covariance."
        ), arms=arms,
    )


def build_moderate_calibrated_measurement_source(
    *, calibration_seed=701, calibration_samples=300,
):
    """Build the previously validated moderate error/calibration work point."""
    optical_profile = next(
        item for item in DEFAULT_OPTICAL_PROFILES
        if item.name == "moderate_combined"
    )
    radar_profile = next(
        item for item in DEFAULT_RADAR_PROFILES
        if item.name == "moderate_combined"
    )
    infrared_profile = next(
        item for item in DEFAULT_INFRARED_PROFILES
        if item.name == "moderate_combined"
    )
    radar_optical_report = run_radar_optical_error_budget_audit(
        seed=calibration_seed, monte_carlo_samples=calibration_samples,
        optical_profiles=(optical_profile,), radar_profiles=(radar_profile,),
    )
    infrared_report = run_infrared_error_budget_audit(
        seed=calibration_seed + 1, monte_carlo_samples=calibration_samples,
        focal_lengths_pixels=(600.0,), peak_snrs=(1000.0,),
        profiles=(infrared_profile,),
    )
    tables = {
        modality: replace(radar_optical_calibration_table(
            radar_optical_report, modality=modality,
            profile="moderate_combined",
        ), apply_bias_correction=True)
        for modality in ("RADAR", "OPTICAL")
    }
    tables["INFRARED"] = replace(
        infrared_calibration_table(
            infrared_report, profile="moderate_combined",
        ),
        apply_bias_correction=True,
    )
    sensors = MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(
            width=128, height=128, focal_length_x_pixels=400.0,
            focal_length_y_pixels=400.0, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
            photon_noise_enabled=optical_profile.photon_noise_enabled,
            stray_light_drift_sigma=optical_profile.stray_light_drift_sigma,
            pointing_jitter_sigma_pixels=optical_profile.pointing_jitter_sigma_pixels,
            radial_distortion_k1_per_pixel2=(
                optical_profile.radial_distortion_k1_per_pixel2
            ),
            fixed_pixel_bias_x=optical_profile.fixed_pixel_bias_x,
            fixed_pixel_bias_y=optical_profile.fixed_pixel_bias_y,
        ),
        infrared=InfraredCameraConfig(
            width=128, height=128, focal_length_x_pixels=600.0,
            focal_length_y_pixels=600.0, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
            photon_noise_enabled=infrared_profile.photon_noise_enabled,
            thermal_background_drift_sigma=(
                infrared_profile.thermal_background_drift_sigma
            ),
            pointing_jitter_sigma_pixels=(
                infrared_profile.pointing_jitter_sigma_pixels
            ),
            radial_distortion_k1_per_pixel2=(
                infrared_profile.radial_distortion_k1_per_pixel2
            ),
            fixed_pixel_bias_x=infrared_profile.fixed_pixel_bias_x,
            fixed_pixel_bias_y=infrared_profile.fixed_pixel_bias_y,
        ),
        radar=RadarRangeDopplerConfig(
            width=129, height=129, source_peak=1000.0,
            background=10.0, read_noise_sigma=1.0,
            background_drift_sigma=radar_profile.background_drift_sigma,
            echo_amplitude_sigma_fraction=(
                radar_profile.echo_amplitude_sigma_fraction
            ),
            range_bias_m=radar_profile.range_bias_m,
            range_rate_bias_mps=radar_profile.range_rate_bias_mps,
            range_jitter_sigma_m=radar_profile.range_jitter_sigma_m,
            range_rate_jitter_sigma_mps=radar_profile.range_rate_jitter_sigma_mps,
        ),
    )
    return sensors, tables


def save_single_multi_measurement_source_shadow(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _single_arm(source, result):
    summary = result["summary"]
    return MeasurementSourceArm(
        scope="single", source=source,
        position_rmse_m=float(summary["position_rmse_m"]),
        valid_radar_count=int(summary["radar_valid_count"]),
        valid_infrared_count=int(summary["infrared_valid_count"]),
        valid_optical_count=int(summary["optical_valid_count"]),
        radar_mean_nis=None, infrared_mean_nis=None, optical_mean_nis=None,
    )


def _walker_arm(source, history, messages, case):
    counts = {
        modality: sum(
            item.modality.upper() == modality and item.valid_flag
            for item in messages
        ) for modality in ("RADAR", "INFRARED", "OPTICAL")
    }
    return MeasurementSourceArm(
        scope="walker", source=source,
        position_rmse_m=_position_rmse(history, case["truth"]),
        valid_radar_count=counts["RADAR"],
        valid_infrared_count=counts["INFRARED"],
        valid_optical_count=counts["OPTICAL"],
        radar_mean_nis=_mean_modality_nis(history, messages, "RADAR"),
        infrared_mean_nis=_mean_modality_nis(history, messages, "INFRARED"),
        optical_mean_nis=_mean_modality_nis(history, messages, "OPTICAL"),
    )
