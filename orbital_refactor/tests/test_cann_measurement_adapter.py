import numpy as np
import pytest

from brain_inspired.cann_measurement_adapter import (
    CANNMeasurementProposal, preprocess_observation,
)
from interfaces.data_objects import Observation


def _source():
    return Observation(
        timestamp=2.0, observer_id="sat", target_id="target",
        modality="INFRARED", source_type="TRADITIONAL",
        measurement=np.array([0.2, 0.1]), covariance=np.eye(2) * 0.01,
        confidence=1.0, frame="SPRI", valid_flag=True,
        metadata={"measurement_type": "AZ_EL", "observation_id": "ir:2"},
    )


def test_preprocessor_exposes_only_standard_observation_boundary():
    result = preprocess_observation(
        _source(), measurement=np.array([0.19, 0.1]),
        diagnostics={"concentration": 0.8},
    )
    assert result.source_type == "CANN_PREPROCESSED"
    assert result.metadata["source_measurement_ids"] == ("ir:2",)
    assert result.metadata["measurement_type"] == "AZ_EL"
    np.testing.assert_array_equal(result.covariance, _source().covariance)
    assert not hasattr(result, "state_estimate")


def test_propagated_proposal_requires_lineage_and_valid_covariance():
    proposal = CANNMeasurementProposal(
        timestamp=4.0, observer_id="sat", target_id="target",
        modality="INFRARED", measurement=np.array([0.2, 0.1]),
        covariance=np.eye(2), confidence=0.5, frame="SPRI",
        valid_flag=True, mode="PROPAGATED", source_measurement_ids=(),
        propagation_duration=2.0,
    )
    with pytest.raises(ValueError, match="lineage"):
        proposal.to_observation()
