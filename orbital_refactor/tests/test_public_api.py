import numpy as np

from examples.run_standard_interface import build_demo_input
from interfaces.interface_contracts import (
    InterfaceErrorCode,
    InterfaceValidationError,
)
from interfaces.module_serialization import (
    load_module_output,
    save_module_input,
)
from interfaces.public_api import get_public_api_info, run_module_bundle
from interfaces.state_awareness_module import StateAwarenessModule


def test_public_api_reports_versions_and_supported_contract():
    info = get_public_api_info()
    assert info["public_api_version"] == "v1.0"
    assert info["module_config_schema_version"] == "v1.0"
    assert info["module_bundle_schema_version"] == "v1.0"
    assert info["supported_filter_architectures"] == ["federated_ci", "centralized"]
    assert {"RADAR", "INFRARED", "OPTICAL"}.issubset(info["supported_modalities"])


def test_interface_error_has_stable_json_response():
    error = InterfaceValidationError(
        InterfaceErrorCode.INVALID_CONFIG, "missing value", field="config.runtime"
    )
    assert error.to_dict() == {
        "error_type": "InterfaceValidationError",
        "code": "INVALID_CONFIG",
        "field": "config.runtime",
        "message": "missing value",
    }


def test_external_bundle_entry_matches_direct_module_execution(tmp_path):
    module_input = build_demo_input()
    expected = StateAwarenessModule().run(module_input)
    request = save_module_input(module_input, tmp_path / "request")
    response = run_module_bundle(request, tmp_path / "response")
    actual = load_module_output(response)

    np.testing.assert_array_equal(
        actual.state_output.position_estimate, expected.state_output.position_estimate
    )
    np.testing.assert_array_equal(actual.state_output.covariance, expected.state_output.covariance)
