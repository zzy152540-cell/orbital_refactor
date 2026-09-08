import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellConfig,
    HierarchicalNavigationPlaceCellEncoder,
)
from brain_inspired.navigation_place_cells import NavigationPlaceCellConfig


def test_hierarchical_encoder_improves_large_offset_reconstruction():
    encoder = HierarchicalNavigationPlaceCellEncoder()
    output = encoder.encode(
        phase=0.7, radial_position=15_000.0,
        along_track_position=-12_000.0,
    )
    coarse_error = np.linalg.norm(
        output.coarse_decoded_rt - np.array([15_000.0, -12_000.0])
    )
    hierarchical_error = np.linalg.norm(np.array([
        output.decoded_radial_position - 15_000.0,
        output.decoded_along_track_position + 12_000.0,
    ]))
    assert output.valid
    assert output.fine_scale_active
    assert hierarchical_error < coarse_error
    assert output.activity.size == encoder.config.cell_count == 1272


def test_hysteresis_prevents_immediate_scale_reentry():
    coarse = NavigationPlaceCellConfig(
        radial_centers_m=(-100.0, 0.0, 100.0),
        along_track_centers_m=(-100.0, 0.0, 100.0),
        radial_sigma_m=50.0, along_track_sigma_m=50.0,
    )
    fine = NavigationPlaceCellConfig(
        radial_centers_m=(-20.0, 0.0, 20.0),
        along_track_centers_m=(-20.0, 0.0, 20.0),
        radial_sigma_m=10.0, along_track_sigma_m=10.0,
    )
    encoder = HierarchicalNavigationPlaceCellEncoder(
        HierarchicalNavigationPlaceCellConfig(
            coarse=coarse, fine=fine,
            fine_enter_fraction=0.5, fine_exit_fraction=0.8,
        )
    )
    outside = encoder.encode(
        phase=0.0, radial_position=1000.0, along_track_position=0.0,
    )
    middle = encoder.encode(
        phase=0.1, radial_position=112.0, along_track_position=0.0,
    )
    inside = encoder.encode(
        phase=0.2, radial_position=95.0, along_track_position=0.0,
    )
    assert not outside.fine_scale_active
    assert not middle.fine_scale_active
    assert inside.fine_scale_active
    assert inside.scale_transition


def test_hierarchical_phase_is_periodic():
    first_encoder = HierarchicalNavigationPlaceCellEncoder()
    second_encoder = HierarchicalNavigationPlaceCellEncoder()
    first = first_encoder.encode(
        phase=0.4, radial_position=5000.0, along_track_position=-5000.0,
    )
    wrapped = second_encoder.encode(
        phase=0.4 + 2.0 * np.pi,
        radial_position=5000.0, along_track_position=-5000.0,
    )
    assert np.allclose(first.activity, wrapped.activity)
