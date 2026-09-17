"""Small stable facade for cross-project file-bundle integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interfaces.interface_contracts import MODULE_CONFIG_SCHEMA_VERSION, Modality
from interfaces.module_serialization import (
    MODULE_BUNDLE_SCHEMA_VERSION,
    load_module_input,
    save_module_output,
)
from interfaces.raw_sensor_serialization import RAW_SENSOR_FRAME_SCHEMA_VERSION
from interfaces.state_awareness_module import StateAwarenessModule


PUBLIC_API_VERSION = "v1.0"

__all__ = ["PUBLIC_API_VERSION", "get_public_api_info", "run_module_bundle"]


def get_public_api_info() -> dict[str, Any]:
    """Describe the externally supported contract without importing internals."""

    return {
        "public_api_version": PUBLIC_API_VERSION,
        "module_config_schema_version": MODULE_CONFIG_SCHEMA_VERSION,
        "module_bundle_schema_version": MODULE_BUNDLE_SCHEMA_VERSION,
        "raw_sensor_frame_schema_version": RAW_SENSOR_FRAME_SCHEMA_VERSION,
        "supported_filter_architectures": ["federated_ci", "centralized"],
        "supported_modalities": [item.value for item in Modality],
        "request_type": "ModuleInput",
        "response_type": "ModuleOutput",
        "persistence_format": "JSON+NPZ (allow_pickle=False)",
    }


def run_module_bundle(
    request_directory: str | Path,
    response_directory: str | Path,
) -> Path:
    """Load one request bundle, execute the stable module and save its response."""

    module_input = load_module_input(request_directory)
    output = StateAwarenessModule().run(module_input)
    return save_module_output(output, response_directory)
