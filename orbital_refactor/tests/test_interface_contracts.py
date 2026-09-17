import numpy as np
import pytest

from examples.run_standard_interface import build_demo_input
from interfaces.data_objects import Observation
from interfaces.interface_contracts import (
    InterfaceErrorCode,
    InterfaceValidationError,
    Modality,
    ReferenceFrame,
    canonical_modality,
    validate_module_input,
    validate_observation,
)
from interfaces.state_awareness_module import StateAwarenessModule


def _radar(**overrides):
    values = dict(
        timestamp=0.0, observer_id="sat_a", target_id="sat_b",
        modality="RADAR", source_type="SIMULATED_RADAR",
        measurement=np.array([1_000.0, 0.2]),
        covariance=np.diag([4.0, 0.01]), confidence=1.0,
        frame="SPRI", valid_flag=True,
        metadata={"measurement_type": "RANGE_RANGE_RATE"},
    )
    values.update(overrides)
    return Observation(**values)


def test_public_enums_remain_string_compatible():
    assert Modality.RADAR == "RADAR"
    assert ReferenceFrame.SPRI == "SPRI"
    assert canonical_modality("ir") == "INFRARED"


def test_infrared_log_extent_contract_accepts_three_components():
    validate_observation(Observation(
        timestamp=0.0, observer_id="sat_a", target_id="sat_b",
        modality="INFRARED", source_type="SIMULATED_INFRARED",
        measurement=np.array([0.1, -0.2, -9.0]),
        covariance=np.diag([1e-6, 1e-6, 0.01]), confidence=1.0,
        frame="SPRI", valid_flag=True,
        metadata={
            "measurement_type": "AZIMUTH_ELEVATION_LOG_ANGULAR_EXTENT"
        },
    ))


def test_standard_interface_demo_passes_contract_and_preserves_result():
    module_input = build_demo_input()
    validate_module_input(module_input)
    output = StateAwarenessModule().run(module_input)
    assert output.state_output.valid_flag
    assert output.state_output.covariance.shape == (6, 6)


def test_observation_rejects_dimension_covariance_and_frame_errors():
    with pytest.raises(InterfaceValidationError) as dimension_error:
        validate_observation(_radar(measurement=np.ones(3), covariance=np.eye(3)))
    assert dimension_error.value.code == InterfaceErrorCode.INVALID_MEASUREMENT_SHAPE

    with pytest.raises(InterfaceValidationError) as covariance_error:
        validate_observation(_radar(covariance=np.array([[1.0, 2.0], [0.0, 1.0]])))
    assert covariance_error.value.code == InterfaceErrorCode.INVALID_COVARIANCE

    with pytest.raises(InterfaceValidationError) as frame_error:
        validate_observation(_radar(frame="CAMERA_NORMALIZED"))
    assert frame_error.value.code == InterfaceErrorCode.INCOMPATIBLE_FRAME


def test_optional_schema_version_rejects_incompatible_contract():
    module_input = build_demo_input()
    module_input.config["schema_version"] = "v2.0"
    with pytest.raises(InterfaceValidationError) as error:
        validate_module_input(module_input)
    assert error.value.code == InterfaceErrorCode.INVALID_CONFIG


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda value: value.config.pop("runtime"), "config.timestamps"),
        (
            lambda value: value.config["runtime"].update(
                chief_state_history_eci=np.zeros((2, 6))
            ),
            "config.chief_state_history_eci",
        ),
        (
            lambda value: value.config["filter"].update(architecture="unknown"),
            "config.filter.architecture",
        ),
        (
            lambda value: value.config["filter"].update(ci_grid_points=1),
            "config.filter.ci_grid_points",
        ),
    ],
)
def test_public_config_contract_reports_machine_readable_errors(mutate, field):
    module_input = build_demo_input()
    mutate(module_input)
    with pytest.raises(InterfaceValidationError) as error:
        validate_module_input(module_input)
    assert error.value.code == InterfaceErrorCode.INVALID_CONFIG
    assert error.value.field == field


def test_module_input_rejects_target_and_runtime_timestamp_mismatches():
    wrong_target = build_demo_input()
    wrong_target.sensor_measurements[0].target_id = "another_target"
    with pytest.raises(InterfaceValidationError) as error:
        validate_module_input(wrong_target)
    assert error.value.code == InterfaceErrorCode.TARGET_MISMATCH

    wrong_time = build_demo_input()
    wrong_time.sensor_measurements[0].timestamp = 0.5
    with pytest.raises(InterfaceValidationError) as error:
        validate_module_input(wrong_time)
    assert error.value.code == InterfaceErrorCode.INVALID_TIMESTAMP


def test_module_input_rejects_duplicate_modality_sample():
    module_input = build_demo_input()
    module_input.sensor_measurements.append(_radar(target_id="target_01"))
    with pytest.raises(InterfaceValidationError) as error:
        validate_module_input(module_input)
    assert error.value.code == InterfaceErrorCode.INVALID_CONFIG
    assert error.value.field == "sensor_measurements"
