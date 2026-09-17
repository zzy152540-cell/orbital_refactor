import json

import numpy as np
import pytest

from examples.run_standard_interface import build_demo_input
from interfaces.module_serialization import (
    MODULE_BUNDLE_MANIFEST,
    load_module_input,
    load_module_output,
    save_module_input,
    save_module_output,
)
from interfaces.state_awareness_module import StateAwarenessModule


def test_module_input_round_trip_preserves_estimator_result(tmp_path):
    original = build_demo_input()
    original.config["integration_metadata"] = {
        "producer": "windows-test",
        "labels": ("portable", "pickle-free"),
        "numeric": np.array([1.0, 2.0]),
    }
    destination = save_module_input(original, tmp_path / "request")
    restored = load_module_input(destination)

    assert restored.initial_state.target_id == original.initial_state.target_id
    assert restored.config["integration_metadata"]["labels"] == (
        "portable", "pickle-free",
    )
    np.testing.assert_array_equal(
        restored.config["integration_metadata"]["numeric"], np.array([1.0, 2.0])
    )
    expected = StateAwarenessModule().run(original)
    actual = StateAwarenessModule().run(restored)
    np.testing.assert_allclose(
        actual.state_output.position_estimate, expected.state_output.position_estimate
    )
    np.testing.assert_allclose(actual.state_output.covariance, expected.state_output.covariance)


def test_module_output_round_trip_is_pickle_free(tmp_path):
    output = StateAwarenessModule().run(build_demo_input())
    destination = save_module_output(output, tmp_path / "response")
    restored = load_module_output(destination)

    np.testing.assert_array_equal(
        restored.state_output.position_estimate, output.state_output.position_estimate
    )
    np.testing.assert_array_equal(restored.state_output.covariance, output.state_output.covariance)
    assert restored.fusion_status.modality_weights == output.fusion_status.modality_weights
    assert restored.runtime_status.status == output.runtime_status.status


def test_module_bundle_rejects_overwrite_schema_and_type_mismatch(tmp_path):
    destination = save_module_input(build_demo_input(), tmp_path / "request")
    with pytest.raises(FileExistsError):
        save_module_input(build_demo_input(), destination)

    manifest_path = destination / MODULE_BUNDLE_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["bundle_type"] = "ModuleOutput"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle type"):
        load_module_input(destination)

    payload["bundle_type"] = "ModuleInput"
    payload["schema_version"] = "v2.0"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_module_input(destination)
