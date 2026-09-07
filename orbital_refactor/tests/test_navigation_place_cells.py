import numpy as np

from brain_inspired.navigation_place_cells import NavigationPlaceCellEncoder
from brain_inspired.navigation_place_cells import (
    build_navigation_place_cell_histories,
)
import pytest


def test_place_cells_are_normalized_and_decode_center():
    output = NavigationPlaceCellEncoder().encode(
        phase=0.0, radial_position=0.0, along_track_position=0.0,
    )
    assert output.valid
    assert np.isclose(np.sum(output.activity), 1.0)
    assert abs(output.decoded_phase) < 1e-12
    assert abs(output.decoded_radial_position) < 1e-12
    assert abs(output.decoded_along_track_position) < 1e-12


def test_place_cells_are_periodic_in_phase():
    encoder = NavigationPlaceCellEncoder()
    first = encoder.encode(
        phase=0.2, radial_position=20.0, along_track_position=-30.0,
    )
    wrapped = encoder.encode(
        phase=0.2 + 2.0 * np.pi,
        radial_position=20.0, along_track_position=-30.0,
    )
    assert np.allclose(first.activity, wrapped.activity)


def test_place_cell_activity_changes_with_rt_position():
    encoder = NavigationPlaceCellEncoder()
    center = encoder.encode(
        phase=1.0, radial_position=0.0, along_track_position=0.0,
    )
    shifted = encoder.encode(
        phase=1.0, radial_position=100.0, along_track_position=-100.0,
    )
    assert not np.allclose(center.activity, shifted.activity)


def test_empty_place_cell_history_input_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        build_navigation_place_cell_histories(navigation_by_node={})
