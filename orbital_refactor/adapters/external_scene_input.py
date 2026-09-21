"""Adapter for main-display scene inputs based on structured orbital elements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from orbital_core.dynamics import rk4_step_absolute
from orbital_core.orbit_elements import keplerian_to_eci


UTC = timezone.utc


@dataclass(frozen=True)
class ExternalSatelliteDefinition:
    node_id: str
    asset_id: str
    satellite_id: str
    norad_id: int
    epoch_ms: int
    orbital_elements_eci: tuple[float, float, float, float, float, float]
    tle_line1: str
    tle_line2: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TleConsistencyDiagnostic:
    node_id: str
    available: bool
    inclination_difference_deg: float | None = None
    raan_difference_deg: float | None = None
    eccentricity_difference: float | None = None
    argument_of_perigee_difference_deg: float | None = None
    mean_anomaly_difference_deg: float | None = None


@dataclass(frozen=True)
class ExternalSceneInitialConditions:
    scene_id: str
    name: str
    start_time_ms: int
    end_time_ms: int
    step_seconds: float
    timestamps: np.ndarray
    satellites: tuple[ExternalSatelliteDefinition, ...]
    initial_state_by_node: Mapping[str, np.ndarray]
    tle_diagnostics: tuple[TleConsistencyDiagnostic, ...]


def load_external_scene_input(path: str | Path) -> ExternalSceneInitialConditions:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return adapt_external_scene_input(payload)


def adapt_external_scene_input(
    payload: Mapping[str, Any], *, propagation_step_seconds: float = 60.0,
) -> ExternalSceneInitialConditions:
    """Convert external structured Kepler elements to scene-start ECI states.

    Naive input datetimes and YYDDD.DDD orbital epochs are interpreted as UTC.
    Structured orbital elements are authoritative; TLE lines are retained only
    for audit diagnostics in this first integration version.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("External scene input must be a JSON object.")
    scene_id = _required_text(payload, "sceneId")
    name = _required_text(payload, "name")
    start = _parse_utc_datetime(_required_text(payload, "startTime"))
    end = _parse_utc_datetime(_required_text(payload, "endTime"))
    if end <= start:
        raise ValueError("Scene endTime must be later than startTime.")
    duration_seconds = (end - start).total_seconds()
    if not float(duration_seconds).is_integer():
        raise ValueError("Scene duration must align to whole seconds.")
    satellites_payload = payload.get("satellites")
    if not isinstance(satellites_payload, list) or not satellites_payload:
        raise ValueError("External scene input requires a nonempty satellites array.")
    if int(payload.get("satelliteCount", -1)) != len(satellites_payload):
        raise ValueError("satelliteCount does not match satellites length.")
    if not np.isfinite(propagation_step_seconds) or propagation_step_seconds <= 0.0:
        raise ValueError("propagation_step_seconds must be finite and positive.")

    satellites = tuple(_parse_satellite(item) for item in satellites_payload)
    node_ids = tuple(item.node_id for item in satellites)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Satellite asset names must be unique within a scene.")
    start_time_ms = _epoch_milliseconds(start)
    initial_states = {}
    diagnostics = []
    for satellite in satellites:
        state_at_epoch = keplerian_to_eci(*satellite.orbital_elements_eci)
        elapsed = (start_time_ms - satellite.epoch_ms) / 1000.0
        if elapsed < 0.0:
            raise ValueError(
                f"Scene start precedes orbital epoch for {satellite.node_id}."
            )
        initial_states[satellite.node_id] = _propagate_to_scene_start(
            state_at_epoch, elapsed, maximum_step=propagation_step_seconds,
        )
        diagnostics.append(_tle_diagnostic(satellite))
    timestamps = np.arange(0.0, duration_seconds + 1.0, 1.0)
    timestamps.setflags(write=False)
    for state in initial_states.values():
        state.setflags(write=False)
    return ExternalSceneInitialConditions(
        scene_id=scene_id, name=name,
        start_time_ms=start_time_ms, end_time_ms=_epoch_milliseconds(end),
        step_seconds=1.0, timestamps=timestamps,
        satellites=satellites, initial_state_by_node=initial_states,
        tle_diagnostics=tuple(diagnostics),
    )


def _parse_satellite(payload: Mapping[str, Any]) -> ExternalSatelliteDefinition:
    if not isinstance(payload, Mapping):
        raise TypeError("Every satellite entry must be a JSON object.")
    node_id = _required_text(payload, "assetName")
    asset_id = _required_text(payload, "assetId")
    satellite_id = _required_text(payload, "satelliteId")
    norad = int(_required_text(payload, "norad"))
    if norad < 0:
        raise ValueError("NORAD ID cannot be negative.")
    semi_major_axis = _finite_float(payload, "semiMajorAxis") * 1000.0
    eccentricity = _finite_float(payload, "eccentricity")
    inclination = np.deg2rad(_finite_float(payload, "inclination"))
    raan = np.deg2rad(_finite_float(payload, "raan"))
    argument = np.deg2rad(_finite_float(payload, "argPerigee"))
    mean_anomaly = np.deg2rad(_finite_float(payload, "meanAnomaly"))
    true_anomaly = _mean_to_true_anomaly(mean_anomaly, eccentricity)
    epoch = _parse_tle_epoch(_required_text(payload, "epoch"))
    tle_line1 = str(payload.get("tleLine1", "")).strip()
    tle_line2 = str(payload.get("tleLine2", "")).strip()
    metadata = {
        name: payload.get(name) for name in (
            "assetName", "name", "cnName", "intlId", "payloadType", "team",
            "country", "orbitType", "serviceStatus", "dataSource",
        )
    }
    return ExternalSatelliteDefinition(
        node_id=node_id, asset_id=asset_id, satellite_id=satellite_id,
        norad_id=norad, epoch_ms=_epoch_milliseconds(epoch),
        orbital_elements_eci=(
            float(semi_major_axis), float(eccentricity), float(inclination),
            float(raan), float(argument), float(true_anomaly),
        ),
        tle_line1=tle_line1, tle_line2=tle_line2, metadata=metadata,
    )


def _mean_to_true_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("Only elliptic orbital elements are supported.")
    mean = float(np.mod(mean_anomaly, 2.0 * np.pi))
    eccentric = mean if eccentricity < 0.8 else np.pi
    for _ in range(30):
        residual = eccentric - eccentricity * np.sin(eccentric) - mean
        delta = residual / (1.0 - eccentricity * np.cos(eccentric))
        eccentric -= delta
        if abs(delta) < 1e-13:
            break
    else:
        raise RuntimeError("Kepler equation did not converge.")
    numerator = np.sqrt(1.0 + eccentricity) * np.sin(0.5 * eccentric)
    denominator = np.sqrt(1.0 - eccentricity) * np.cos(0.5 * eccentric)
    return float(np.mod(2.0 * np.arctan2(numerator, denominator), 2.0 * np.pi))


def _propagate_to_scene_start(state, elapsed_seconds, *, maximum_step):
    result = np.asarray(state, dtype=float).reshape(6).copy()
    remaining = float(elapsed_seconds)
    while remaining > 0.0:
        step = min(float(maximum_step), remaining)
        result = rk4_step_absolute(result, step)
        remaining -= step
    return result


def _tle_diagnostic(satellite: ExternalSatelliteDefinition) -> TleConsistencyDiagnostic:
    fields = satellite.tle_line2.split()
    if len(fields) < 8 or fields[0] != "2":
        return TleConsistencyDiagnostic(satellite.node_id, False)
    try:
        tle_inclination = float(fields[2])
        tle_raan = float(fields[3])
        tle_eccentricity = float(f"0.{fields[4]}")
        tle_argument = float(fields[5])
        tle_mean_anomaly = float(fields[6])
    except ValueError:
        return TleConsistencyDiagnostic(satellite.node_id, False)
    a, e, inclination, raan, argument, true_anomaly = satellite.orbital_elements_eci
    del a
    structured_mean = _true_to_mean_anomaly(true_anomaly, e)
    return TleConsistencyDiagnostic(
        node_id=satellite.node_id, available=True,
        inclination_difference_deg=float(np.rad2deg(inclination) - tle_inclination),
        raan_difference_deg=_wrapped_degrees(np.rad2deg(raan) - tle_raan),
        eccentricity_difference=float(e - tle_eccentricity),
        argument_of_perigee_difference_deg=_wrapped_degrees(
            np.rad2deg(argument) - tle_argument
        ),
        mean_anomaly_difference_deg=_wrapped_degrees(
            np.rad2deg(structured_mean) - tle_mean_anomaly
        ),
    )


def _true_to_mean_anomaly(true_anomaly, eccentricity):
    eccentric = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )
    return float(np.mod(eccentric - eccentricity * np.sin(eccentric), 2.0 * np.pi))


def _wrapped_degrees(value):
    return float((value + 180.0) % 360.0 - 180.0)


def _parse_tle_epoch(value: str) -> datetime:
    try:
        year_short = int(value[:2])
        day_of_year = float(value[2:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid YYDDD.DDD orbital epoch: {value!r}") from exc
    year = 1900 + year_short if year_short >= 57 else 2000 + year_short
    if not 1.0 <= day_of_year < 367.0:
        raise ValueError(f"Invalid day of year in orbital epoch: {value!r}")
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1.0)


def _parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid scene UTC datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _epoch_milliseconds(value: datetime) -> int:
    return int(round(value.timestamp() * 1000.0))


def _required_text(payload, name):
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"External scene field {name} must be a nonempty string.")
    return value.strip()


def _finite_float(payload, name):
    try:
        value = float(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"External satellite field {name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"External satellite field {name} must be finite.")
    return value
