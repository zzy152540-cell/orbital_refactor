import numpy as np
from experiments.cann_inter_satellite_azimuth import _weighted_circular_mean, _difference

def test_weighted_circular_mean_handles_wrap_boundary():
    value=_weighted_circular_mean(np.deg2rad(359),np.deg2rad(1),1,1)
    assert abs(_difference(value,0.0))<1e-12
