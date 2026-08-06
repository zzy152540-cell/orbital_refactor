import numpy as np

from adapters.synthetic_measurement_adapter import (
    create_nn_state_observations,
    create_optical_observations,
    visibility_flags,
)


def test_optical_and_nn_observations_have_expected_metadata():
    timestamps = np.array([0.0, 1.0])
    position = np.array([[1.0, 2.0, 10.0], [2.0, 2.0, -1.0]])
    optical = create_optical_observations(
        timestamps=timestamps,
        relative_position_spri=position,
        covariance=np.eye(2) * 1e-8,
        observer_id="sat",
        target_id="target",
        rng=np.random.default_rng(1),
    )
    assert optical[0].valid_flag
    assert not optical[1].valid_flag
    assert optical[0].metadata["measurement_type"] == "NORMALIZED_IMAGE_COORDINATES"

    states = np.column_stack((position, np.zeros((2, 3))))
    nn = create_nn_state_observations(
        timestamps=timestamps,
        relative_state_eci=states,
        covariance=np.eye(6),
        observer_id="sat",
        target_id="target",
        rng=np.random.default_rng(2),
    )
    assert nn[0].source_type == "LEARNING"
    assert nn[0].frame == "ECI"


def test_visibility_flags_range_and_camera_constraints():
    position = np.array([[0.1, 0.1, 1.0], [10.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    flags = visibility_flags(
        relative_position_spri=position,
        min_range=0.5,
        max_range=20.0,
        require_positive_z=True,
        field_of_view_limit=1.0,
    )
    np.testing.assert_array_equal(flags, [True, False, False])
