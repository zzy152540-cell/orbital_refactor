"""Strict JSON configuration for the visualization simulation launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np

from experiments.v15_dynamic_visualization import COMMUNICATION_DEGRADATION_PROFILES


CONFIG_SCHEMA_VERSION = "v1.0"
CONFIG_SECTIONS = {
    "constellation": (
        "scenario_type", "total_satellites", "plane_count", "phasing", "altitude_km",
        "inclination_deg",
        "friendly_total_satellites", "friendly_plane_count", "friendly_phasing",
        "friendly_altitude_km", "friendly_inclination_deg",
        "hostile_total_satellites", "hostile_plane_count", "hostile_phasing",
        "hostile_altitude_km", "hostile_inclination_deg",
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
    "topology": ("topology_audit_max_duration_s", "enable_link_suspension"),
    "cann": ("cann_enabled", "cann_node_limit"),
    "output": ("output", "open_after_run", "auto_increment_output"),
}


@dataclass(frozen=True)
class VisualizationSimulationConfig:
    scenario_type: str = "WALKER_COOPERATIVE"
    total_satellites: int = 20
    plane_count: int = 10
    phasing: int = 1
    altitude_km: float = 700.0
    inclination_deg: float = 53.0
    friendly_total_satellites: int = 15
    friendly_plane_count: int = 3
    friendly_phasing: int = 1
    friendly_altitude_km: float = 700.0
    friendly_inclination_deg: float = 53.0
    hostile_total_satellites: int = 15
    hostile_plane_count: int = 3
    hostile_phasing: int = 1
    hostile_altitude_km: float = 705.0
    hostile_inclination_deg: float = 53.0
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
    topology_audit_max_duration_s: float = 300.0
    enable_link_suspension: bool = True
    cann_enabled: bool = True
    cann_node_limit: int = 3
    output: str = "results/visualization_recordings/walker20_configured"
    open_after_run: bool = True
    auto_increment_output: bool = True

    def __post_init__(self) -> None:
        if self.scenario_type not in {
            "WALKER_COOPERATIVE", "BILATERAL_CONFIGURABLE", "BILATERAL_15X15",
        }:
            raise ValueError("Unknown visualization scenario_type.")
        if self.total_satellites < 2:
            raise ValueError("total_satellites must be at least two.")
        if self.plane_count < 1 or self.total_satellites % self.plane_count:
            raise ValueError("plane_count must divide total_satellites.")
        if not 0 <= self.phasing < self.plane_count:
            raise ValueError("phasing must be in [0, plane_count).")
        for role in ("friendly", "hostile"):
            total = int(getattr(self, f"{role}_total_satellites"))
            planes = int(getattr(self, f"{role}_plane_count"))
            phasing = int(getattr(self, f"{role}_phasing"))
            if total < 2:
                raise ValueError(
                    f"{role}_total_satellites must be at least two."
                )
            if planes < 1 or total % planes:
                raise ValueError(
                    f"{role}_plane_count must divide {role}_total_satellites."
                )
            if not 0 <= phasing < planes:
                raise ValueError(
                    f"{role}_phasing must be in [0, {role}_plane_count)."
                )
        positive = (
            self.altitude_km, self.friendly_altitude_km,
            self.hostile_altitude_km, self.duration_s, self.dt_s,
            self.maximum_range_km,
            self.process_noise_acceleration, self.radar_range_sigma_m,
            self.radar_range_rate_sigma_mps, self.infrared_angle_sigma_deg,
            self.optical_image_sigma, self.absolute_navigation_sigma_m,
            self.replay_history_window_s, self.max_pinned_age_s,
            self.topology_audit_max_duration_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Physical scales and time settings must be positive.")
        if not np.isfinite(self.inclination_deg) or not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180].")
        for role in ("friendly", "hostile"):
            inclination = float(getattr(self, f"{role}_inclination_deg"))
            if not np.isfinite(inclination) or not 0.0 <= inclination <= 180.0:
                raise ValueError(
                    f"{role}_inclination_deg must be in [0, 180]."
                )
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
            "topology_audit_max_duration": float(
                self.topology_audit_max_duration_s
            ),
            "enable_link_suspension": bool(self.enable_link_suspension),
            "enable_absolute_navigation_dropout": bool(
                self.enable_absolute_navigation_dropout
            ),
            "cann_node_limit": int(self.cann_node_limit),
        }

    @property
    def bilateral_generator_arguments(self):
        frozen = self.scenario_type == "BILATERAL_15X15"
        return {
            "duration": float(self.duration_s),
            "dt": float(self.dt_s),
            "seed": int(self.seed),
            "include_cann": bool(self.cann_enabled),
            "cann_node_limit": int(self.cann_node_limit),
            "retain_raw_frames": True,
            # The ordinary bilateral scene uses the same three physical
            # modalities as the nominal acceptance controls.  A synthetic
            # radial backup is enabled only by an explicit robustness preset.
            "radial_backup_profile": "none",
            "enable_local_unified": True,
            "enable_consensus": bool(frozen),
            "friendly_total_satellites": (
                15 if frozen else int(self.friendly_total_satellites)
            ),
            "friendly_plane_count": (
                3 if frozen else int(self.friendly_plane_count)
            ),
            "friendly_phasing": 1 if frozen else int(self.friendly_phasing),
            "friendly_altitude_m": 1000.0 * (
                700.0 if frozen else float(self.friendly_altitude_km)
            ),
            "friendly_inclination_rad": np.deg2rad(
                53.0 if frozen else float(self.friendly_inclination_deg)
            ),
            "hostile_total_satellites": (
                15 if frozen else int(self.hostile_total_satellites)
            ),
            "hostile_plane_count": (
                3 if frozen else int(self.hostile_plane_count)
            ),
            "hostile_phasing": 1 if frozen else int(self.hostile_phasing),
            "hostile_altitude_m": 1000.0 * (
                705.0 if frozen else float(self.hostile_altitude_km)
            ),
            "hostile_inclination_rad": np.deg2rad(
                53.0 if frozen else float(self.hostile_inclination_deg)
            ),
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


def next_available_recording_path(
    requested: str | Path, *, timestamp: datetime | None = None,
) -> Path:
    path = Path(requested)
    if not path.exists():
        return path
    base_name = re.sub(
        r"(?:(?:_\d{3})+|_\d{8}_\d{6}(?:_\d{3})?)$", "", path.name,
    )
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{base_name}_{stamp}")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = path.with_name(f"{base_name}_{stamp}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique recording directory name.")
