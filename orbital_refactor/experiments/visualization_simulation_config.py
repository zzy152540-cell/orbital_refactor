"""Strict JSON configuration for the visualization simulation launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from experiments.v15_dynamic_visualization import COMMUNICATION_DEGRADATION_PROFILES


CONFIG_SCHEMA_VERSION = "v1.0"
CONFIG_SECTIONS = {
    "constellation": (
        "total_satellites", "plane_count", "phasing", "altitude_km",
        "inclination_deg",
    ),
    "simulation": (
        "duration_s", "dt_s", "seed", "enable_absolute_navigation_dropout",
    ),
    "dynamics": ("process_noise_acceleration",),
    "measurements": (
        "radar_enabled", "infrared_enabled", "optical_enabled",
        "radar_range_sigma_m", "radar_range_rate_sigma_mps",
        "infrared_angle_sigma_deg", "optical_image_sigma",
        "absolute_navigation_sigma_m",
    ),
    "visibility": ("maximum_range_km",),
    "filter": ("replay_history_window_s", "max_pinned_age_s"),
    "communication": ("communication_profile",),
    "topology": ("enable_link_suspension",),
    "cann": ("cann_enabled", "cann_node_limit"),
    "output": ("output", "open_after_run", "auto_increment_output"),
}


@dataclass(frozen=True)
class VisualizationSimulationConfig:
    total_satellites: int = 20
    plane_count: int = 10
    phasing: int = 1
    altitude_km: float = 700.0
    inclination_deg: float = 53.0
    duration_s: float = 120.0
    dt_s: float = 2.0
    maximum_range_km: float = 6000.0
    seed: int = 0
    enable_absolute_navigation_dropout: bool = True
    process_noise_acceleration: float = 1e-8
    radar_enabled: bool = True
    infrared_enabled: bool = True
    optical_enabled: bool = True
    radar_range_sigma_m: float = 2.0
    radar_range_rate_sigma_mps: float = 0.05
    infrared_angle_sigma_deg: float = 0.05
    optical_image_sigma: float = 1e-3
    absolute_navigation_sigma_m: float = 3.0
    replay_history_window_s: float = 10.0
    max_pinned_age_s: float = 10.0
    communication_profile: str = "mild"
    enable_link_suspension: bool = True
    cann_enabled: bool = True
    cann_node_limit: int = 3
    output: str = "results/visualization_recordings/walker20_configured"
    open_after_run: bool = True
    auto_increment_output: bool = True

    def __post_init__(self) -> None:
        if self.total_satellites < 2:
            raise ValueError("total_satellites must be at least two.")
        if self.plane_count < 1 or self.total_satellites % self.plane_count:
            raise ValueError("plane_count must divide total_satellites.")
        if not 0 <= self.phasing < self.plane_count:
            raise ValueError("phasing must be in [0, plane_count).")
        positive = (
            self.altitude_km, self.duration_s, self.dt_s, self.maximum_range_km,
            self.process_noise_acceleration, self.radar_range_sigma_m,
            self.radar_range_rate_sigma_mps, self.infrared_angle_sigma_deg,
            self.optical_image_sigma, self.absolute_navigation_sigma_m,
            self.replay_history_window_s, self.max_pinned_age_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Physical scales and time settings must be positive.")
        if not np.isfinite(self.inclination_deg) or not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180].")
        if self.communication_profile not in COMMUNICATION_DEGRADATION_PROFILES:
            raise ValueError("Unknown communication_profile.")
        if not any((self.radar_enabled, self.infrared_enabled, self.optical_enabled)):
            raise ValueError("At least one relative measurement modality is required.")
        if not self.output.strip():
            raise ValueError("output cannot be empty.")
        if self.cann_node_limit < 0:
            raise ValueError("cann_node_limit cannot be negative.")

    @property
    def generator_arguments(self):
        modalities = tuple(
            name for name, enabled in (
                ("RADAR", self.radar_enabled),
                ("INFRARED", self.infrared_enabled),
                ("OPTICAL", self.optical_enabled),
            ) if enabled
        )
        return {
            "duration": float(self.duration_s), "dt": float(self.dt_s),
            "seed": int(self.seed), "include_cann": bool(self.cann_enabled),
            "communication_profile": self.communication_profile,
            "total_satellites": int(self.total_satellites),
            "plane_count": int(self.plane_count), "phasing": int(self.phasing),
            "altitude": float(self.altitude_km) * 1000.0,
            "inclination": np.deg2rad(float(self.inclination_deg)),
            "maximum_range": float(self.maximum_range_km) * 1000.0,
            "relative_modalities": modalities,
            "range_sigma": float(self.radar_range_sigma_m),
            "range_rate_sigma": float(self.radar_range_rate_sigma_mps),
            "az_el_sigma": np.deg2rad(float(self.infrared_angle_sigma_deg)),
            "optical_sigma": float(self.optical_image_sigma),
            "absolute_sigma": float(self.absolute_navigation_sigma_m),
            "process_noise_acceleration": float(self.process_noise_acceleration),
            "replay_history_window": float(self.replay_history_window_s),
            "max_pinned_age": float(self.max_pinned_age_s),
            "enable_link_suspension": bool(self.enable_link_suspension),
            "enable_absolute_navigation_dropout": bool(
                self.enable_absolute_navigation_dropout
            ),
            "cann_node_limit": int(self.cann_node_limit),
        }


def load_visualization_simulation_config(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return visualization_simulation_config_from_dict(payload)


def visualization_simulation_config_from_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError("Visualization configuration must be a JSON object.")
    payload = dict(payload)
    version = payload.pop("schema_version", None)
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {CONFIG_SCHEMA_VERSION}.")
    allowed = set(VisualizationSimulationConfig.__dataclass_fields__)
    section_names = set(CONFIG_SECTIONS) | {"integrity"}
    if not set(payload).issubset(allowed) and set(payload) & section_names:
        unknown_sections = set(payload) - section_names
        if unknown_sections:
            raise ValueError(
                f"Unknown visualization configuration sections: {sorted(unknown_sections)}"
            )
        flattened = {}
        for section, field_names in CONFIG_SECTIONS.items():
            values = payload.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section {section} must be an object.")
            unknown = set(values) - set(field_names)
            if unknown:
                raise ValueError(f"Unknown fields in {section}: {sorted(unknown)}")
            flattened.update(values)
        integrity = payload.get("integrity", {"policy": "validated_baseline"})
        if integrity != {"policy": "validated_baseline"}:
            raise ValueError("Only the validated_baseline integrity policy is supported.")
        payload = flattened
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown visualization configuration fields: {sorted(unknown)}")
    return VisualizationSimulationConfig(**payload)


def save_visualization_simulation_config(config, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(config)
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        **{
            section: {field: values[field] for field in field_names}
            for section, field_names in CONFIG_SECTIONS.items()
        },
        "integrity": {"policy": "validated_baseline"},
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


def next_available_recording_path(requested: str | Path) -> Path:
    path = Path(requested)
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique recording directory name.")
