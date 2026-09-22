"""Build the external 31-satellite scene input from project trajectory files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from orbital_core.constants import MU_EARTH, R_EARTH
from orbital_core.orbit_elements import eci_to_keplerian


def build_project_scene_input(
    functional_initialization_path: str | Path,
    background_trajectory_path: str | Path,
    functional_trajectory_path: str | Path,
) -> dict[str, Any]:
    """Create the main-display input schema from 6+25 supplied trajectories."""

    initialization = _read_json(functional_initialization_path)
    background = _read_json(background_trajectory_path)
    functional_frames = _read_json(functional_trajectory_path)
    if not isinstance(initialization, list) or not isinstance(functional_frames, list):
        raise ValueError("Functional initialization and trajectory roots must be arrays.")
    background_satellites = background.get("satellites", [])
    if len(initialization) != 6 or len(background_satellites) != 25:
        raise ValueError("Expected 6 functional and 25 background satellites.")
    if not functional_frames or len(functional_frames[0].get("satelliteList", [])) != 6:
        raise ValueError("Functional trajectory must contain six satellites per frame.")

    first_functional = {
        item["satObj"]: item for item in functional_frames[0]["satelliteList"]
    }
    functional_tle = {}
    for entry in initialization:
        tle = _find_tle(entry)
        name = _find_functional_name(entry, first_functional)
        functional_tle[name] = tle

    start = _parse_time(functional_frames[0]["time"])
    end = _parse_time(functional_frames[-1]["time"])
    satellites = []
    for item in functional_frames[0]["satelliteList"]:
        satellites.append(_satellite_payload(
            name=str(item["satObj"]), norad=int(item["satId"]),
            state=_position_state(item["position"]),
            tle=functional_tle[str(item["satObj"])], epoch=start,
            role="FUNCTIONAL",
        ))
    for item in background_satellites:
        points = item.get("points", [])
        if not points:
            raise ValueError(f"Background satellite {item.get('name')} has no points.")
        point_time = _parse_time(points[0]["time"])
        if point_time != start:
            raise ValueError("All trajectory histories must share the same start time.")
        satellites.append(_satellite_payload(
            name=str(item["name"]), norad=int(item["norad"]),
            state=_position_state(points[0]),
            tle=(str(item.get("tleLine1", "")), str(item.get("tleLine2", ""))),
            epoch=start, role="BACKGROUND",
        ))
    if len({item["assetName"] for item in satellites}) != 31:
        raise ValueError("The combined project scene must contain 31 unique satellites.")
    return {
        "source": "space-perception",
        "pushedAt": _format_time(datetime.now(timezone.utc)),
        "sceneId": str(background.get("sceneId", functional_frames[0].get("run_id", ""))),
        "name": "31-satellite state-estimation integration scene",
        "type": "STATE_ESTIMATION_INTEGRATION",
        "status": "READY",
        "startTime": _format_time(start, utc_suffix=False),
        "endTime": _format_time(end, utc_suffix=False),
        "remark": "Generated from supplied 6 functional and 25 background trajectories",
        "description": "Structured osculating elements are derived from supplied J2000 states; TLE is retained for audit.",
        "satelliteCount": len(satellites),
        "satellites": satellites,
    }


def save_project_scene_input(payload: Mapping[str, Any], destination: str | Path) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Scene input already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _satellite_payload(*, name, norad, state, tle, epoch, role):
    a, e, inc, raan, argument, true_anomaly = eci_to_keplerian(state)
    mean_anomaly = _true_to_mean(true_anomaly, e)
    mean_motion = np.sqrt(MU_EARTH / a**3) * 86400.0 / (2.0 * np.pi)
    period_minutes = 1440.0 / mean_motion
    return {
        "assetId": f"asset-{norad}", "assetName": name,
        "satelliteId": f"satellite-{norad}", "norad": str(norad),
        "name": name, "cnName": name, "intlId": "",
        "payloadType": role, "team": "RED" if name.startswith("R") else "BLUE",
        "country": "", "orbitType": "LEO", "serviceStatus": "IN_ORBIT",
        "dataSource": "J2000_TRAJECTORY_DERIVED",
        "semiMajorAxis": f"{a / 1000.0:.12f}",
        "eccentricity": f"{e:.15f}",
        "inclination": f"{np.rad2deg(inc):.12f}",
        "raan": f"{np.rad2deg(raan):.12f}",
        "argPerigee": f"{np.rad2deg(argument):.12f}",
        "meanAnomaly": f"{np.rad2deg(mean_anomaly):.12f}",
        "period": f"{period_minutes:.9f}",
        "apogee": f"{(a * (1.0 + e) - R_EARTH) / 1000.0:.9f}",
        "perigee": f"{(a * (1.0 - e) - R_EARTH) / 1000.0:.9f}",
        "orbitAltitude": f"{(a - R_EARTH) / 1000.0:.9f}",
        "epoch": _tle_epoch(epoch), "meanMotion": f"{mean_motion:.12f}",
        "bstar": "0", "tleLine1": tle[0], "tleLine2": tle[1],
    }


def _true_to_mean(true_anomaly, eccentricity):
    eccentric = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(true_anomaly / 2.0),
        np.sqrt(1.0 + eccentricity) * np.cos(true_anomaly / 2.0),
    )
    return float(np.mod(eccentric - eccentricity * np.sin(eccentric), 2.0 * np.pi))


def _position_state(value):
    return np.array([value[name] for name in ("x", "y", "z", "vx", "vy", "vz")], dtype=float)


def _find_tle(entry):
    for value in entry.values():
        if isinstance(value, Mapping) and "line1" in value and "line2" in value:
            return str(value["line1"]), str(value["line2"])
    raise ValueError("Functional initialization entry is missing TLE lines.")


def _find_functional_name(entry, known):
    for value in entry.values():
        if isinstance(value, str) and value in known:
            return value
    raise ValueError("Cannot associate functional initialization with trajectory.")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _parse_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _format_time(value, *, utc_suffix=True):
    text = value.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    return text.replace("+00:00", "Z") if utc_suffix else text.replace("+00:00", "")


def _tle_epoch(value):
    start = datetime(value.year, 1, 1, tzinfo=timezone.utc)
    day = 1.0 + (value - start).total_seconds() / 86400.0
    return f"{value.year % 100:02d}{day:012.8f}"
