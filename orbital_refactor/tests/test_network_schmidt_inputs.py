import numpy as np

from cooperative.network_schmidt_inputs import route_relative_observations
from cooperative.topology import NetworkTopology
from interfaces.data_objects import ObservationMessage


def test_relative_router_drops_invalid_observations():
    invalid = ObservationMessage(
        message_id="invalid", observer_id="a", target_id="b",
        timestamp=0.0, modality="RADAR", measurement=np.zeros(2),
        covariance=np.eye(2), valid_flag=False,
    )
    routed = route_relative_observations(
        [invalid], times=np.array([0.0]),
        topology=NetworkTopology({"a": ["b"], "b": ["a"]}),
        observation_usage="observer_only",
    )
    assert routed == {0.0: {}}
