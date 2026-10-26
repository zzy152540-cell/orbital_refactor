import pytest
from datetime import datetime
from pathlib import Path

import experiments.run_v15_visualization_simulation as simulation_runner
from experiments.visualization_simulation_config import (
    load_visualization_simulation_config, next_available_recording_path,
    visualization_simulation_config_from_dict,
)


def test_visualization_simulation_config_loads_and_converts_units():
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "total_satellites": 10,
        "plane_count": 5, "duration_s": 20.0, "dt_s": 2.0,
        "altitude_km": 800.0, "inclination_deg": 60.0,
        "maximum_range_km": 5000.0, "communication_profile": "mild",
        "topology_audit_max_duration_s": 90.0,
        "output": "results/example", "open_after_run": False,
    })
    arguments = config.generator_arguments
    assert arguments["total_satellites"] == 10
    assert arguments["altitude"] == 800e3
    assert arguments["maximum_range"] == 5000e3
    assert arguments["topology_audit_max_duration"] == 90.0
    assert not config.open_after_run


def test_visualization_simulation_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown visualization"):
        visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "unsupported": True,
        })


def test_visualization_simulation_config_rejects_invalid_walker_partition():
    with pytest.raises(ValueError, match="must divide"):
        visualization_simulation_config_from_dict({
            "schema_version": "v1.0", "total_satellites": 10,
            "plane_count": 3,
        })


def test_configured_runner_forwards_validated_arguments_without_opening(
    monkeypatch,
):
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "duration_s": 8.0,
        "output": "results/configured-test", "open_after_run": False,
    })
    captured = {}

    def fake_generate(output, **arguments):
        captured.update(arguments)
        return output

    monkeypatch.setattr(
        simulation_runner, "generate_dynamic_walker_recording", fake_generate,
    )
    monkeypatch.setattr(
        simulation_runner, "load_visualization_simulation_config",
        lambda _path: config,
    )
    output = simulation_runner.run_configured_visualization_simulation("unused.json")
    assert output.as_posix() == "results/configured-test"
    assert captured["duration"] == 8.0
    assert captured["total_satellites"] == 20


def test_repository_example_configuration_is_valid():
    config = load_visualization_simulation_config(
        "configs/visualization_walker20.json"
    )
    assert config.total_satellites == 20
    bilateral = load_visualization_simulation_config(
        "configs/visualization_bilateral30.json"
    )
    assert bilateral.scenario_type == "BILATERAL_15X15"
    assert bilateral.duration_s == 20.0
    custom = load_visualization_simulation_config(
        "configs/visualization_bilateral_custom.json"
    )
    assert custom.scenario_type == "BILATERAL_CONFIGURABLE"
    assert custom.friendly_total_satellites == 8
    assert custom.hostile_total_satellites == 6


def test_bilateral_configuration_exposes_only_relevant_generator_arguments():
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0",
        "scenario_type": "BILATERAL_15X15",
        "duration_s": 20.0,
        "dt_s": 1.0,
        "seed": 3,
        "cann_enabled": True,
        "cann_node_limit": 4,
        "output": "results/bilateral-configured",
    })
    arguments = config.bilateral_generator_arguments
    assert arguments["duration"] == 20.0
    assert arguments["dt"] == 1.0
    assert arguments["seed"] == 3
    assert arguments["include_cann"] is True
    assert arguments["cann_node_limit"] == 4
    assert arguments["retain_raw_frames"] is True
    assert arguments["radial_backup_profile"] == "none"
    assert arguments["enable_local_unified"] is True
    assert arguments["enable_consensus"] is True
    assert arguments["friendly_total_satellites"] == 15
    assert arguments["friendly_plane_count"] == 3
    assert arguments["hostile_total_satellites"] == 15
    assert arguments["hostile_plane_count"] == 3
    assert arguments["maximum_observation_range_m"] == 6_000_000.0
    assert arguments["process_noise_acceleration"] == 1e-4
    assert arguments["tracking_modalities"] == (
        "RADAR", "INFRARED", "OPTICAL",
    )


def test_configurable_bilateral_walker_fleets_are_forwarded():
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0",
        "scenario_type": "BILATERAL_CONFIGURABLE",
        "friendly_total_satellites": 8,
        "friendly_plane_count": 2,
        "friendly_phasing": 1,
        "friendly_altitude_km": 720.0,
        "friendly_inclination_deg": 55.0,
        "hostile_total_satellites": 6,
        "hostile_plane_count": 3,
        "hostile_phasing": 2,
        "hostile_altitude_km": 760.0,
        "hostile_inclination_deg": 57.0,
        "maximum_range_km": 4500.0,
        "bilateral_process_noise_acceleration": 2e-4,
        "radar_enabled": True,
        "infrared_enabled": False,
        "optical_enabled": True,
        "estimation_architecture": "SINGLE_OBSERVER",
        "tracking_observer_index": 3,
        "tracking_target_index": 4,
        "dropout_modality": "RADAR",
        "dropout_start_s": 20.0,
        "dropout_duration_s": 30.0,
    })
    arguments = config.bilateral_generator_arguments

    assert arguments["friendly_total_satellites"] == 8
    assert arguments["friendly_plane_count"] == 2
    assert arguments["friendly_altitude_m"] == 720_000.0
    assert arguments["hostile_total_satellites"] == 6
    assert arguments["hostile_plane_count"] == 3
    assert arguments["hostile_altitude_m"] == 760_000.0
    assert arguments["radial_backup_profile"] == "none"
    assert arguments["enable_local_unified"] is True
    assert arguments["enable_consensus"] is False
    assert arguments["maximum_observation_range_m"] == 4_500_000.0
    assert arguments["process_noise_acceleration"] == 2e-4
    assert arguments["tracking_modalities"] == ("RADAR", "OPTICAL")
    assert arguments["enable_consensus"] is False
    assert arguments["tracking_observer_indices"] == (3,)
    assert arguments["tracking_target_indices"] == (4,)
    assert arguments["show_communication_topology"] is False
    assert arguments["dropout_modality"] == "RADAR"
    assert arguments["dropout_start"] == 20.0
    assert arguments["dropout_end"] == pytest.approx(50.0)
    assert arguments["dropout_end"] < 50.0
    assert arguments["configured_dropout_duration_s"] == 30.0


def test_configurable_bilateral_cooperative_architecture_enables_consensus():
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0",
        "scenario_type": "BILATERAL_CONFIGURABLE",
        "estimation_architecture": "MULTI_COOPERATIVE",
    })
    arguments = config.bilateral_generator_arguments

    assert arguments["enable_consensus"] is True
    assert arguments["tracking_observer_indices"] is None
    assert arguments["tracking_target_indices"] is None
    assert arguments["show_communication_topology"] is True


def test_configured_runner_routes_bilateral_scenario(monkeypatch):
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0",
        "scenario_type": "BILATERAL_15X15",
        "duration_s": 6.0,
        "output": "results/bilateral-configured-test",
        "open_after_run": False,
    })
    captured = {}

    def fake_generate(output, **arguments):
        captured.update(arguments)
        return output

    monkeypatch.setattr(
        simulation_runner, "generate_bilateral_visualization_recording",
        fake_generate,
    )
    monkeypatch.setattr(
        simulation_runner, "load_visualization_simulation_config",
        lambda _path: config,
    )
    output = simulation_runner.run_configured_visualization_simulation("unused.json")
    assert output.as_posix() == "results/bilateral-configured-test"
    assert captured["duration"] == 6.0
    assert captured["retain_raw_frames"] is True


def test_next_available_recording_path_uses_timestamp_and_strips_legacy_suffixes(
    monkeypatch,
):
    occupied = {"run_001_002", "run_20260921_153045"}
    monkeypatch.setattr(Path, "exists", lambda self: self.name in occupied)
    fixed = datetime(2026, 9, 21, 15, 30, 45)
    assert next_available_recording_path(
        "results/run_001_002", timestamp=fixed,
    ).as_posix() == (
        "results/run_20260921_153045_001"
    )
