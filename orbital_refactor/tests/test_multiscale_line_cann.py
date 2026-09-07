import numpy as np
import pytest

from brain_inspired.line_cann import LineCANNConfig
from brain_inspired.multiscale_line_cann import (
    MultiScaleLineCANN,
    MultiScaleLineCANNConfig,
)


def test_multiscale_line_represents_large_value_with_local_residual():
    cann = MultiScaleLineCANN()
    output = cann.initialize(15_123.0)
    assert output.valid
    assert output.decoded_value == pytest.approx(15_123.0, abs=1e-6)
    assert abs(output.fine_residual) <= 625.0
    assert not output.saturated_at_boundary


def test_multiscale_line_recenters_fine_scale_continuously():
    cann = MultiScaleLineCANN()
    first = cann.initialize(15_600.0)
    second = cann.step(20.0, 2.0)
    assert second.coarse_cell_changed
    assert second.decoded_value == pytest.approx(first.decoded_value + 40.0, abs=1e-6)
    assert abs(second.fine_residual) <= 625.0


def test_multiscale_line_reports_outer_boundary_saturation():
    cann = MultiScaleLineCANN()
    output = cann.initialize(60_000.0)
    assert output.saturated_at_boundary


def test_multiscale_config_requires_fine_cell_overlap():
    with pytest.raises(ValueError, match="half one coarse-cell"):
        MultiScaleLineCANNConfig(
            coarse=LineCANNConfig(
                num_neurons=11, minimum_value=-10.0,
                maximum_value=10.0, tuning_width=1.0,
            ),
            fine=LineCANNConfig(
                num_neurons=11, minimum_value=-0.5,
                maximum_value=0.5, tuning_width=0.1,
            ),
        ).validate()
