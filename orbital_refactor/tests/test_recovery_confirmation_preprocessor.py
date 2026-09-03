import numpy as np

from experiments.recovery_confirmation_preprocessor import (
    RecoveryConfirmationPreprocessor,
)
from interfaces.data_objects import Observation


def _radar(timestamp, value, valid=True):
    return Observation(
        timestamp=timestamp, observer_id="sat", target_id="target",
        modality="RADAR", source_type="TRADITIONAL",
        measurement=np.asarray(value, dtype=float), covariance=np.eye(2),
        confidence=1.0, frame="SPRI", valid_flag=valid,
    )


def test_confirmation_rejects_two_bad_recovery_samples_then_accepts_pair():
    stream = [
        _radar(0.0, [1000.0, 1.0]),
        _radar(2.0, [1002.0, 1.0]),
        _radar(30.0, [4000.0, 6.0]),
        _radar(32.0, [-2000.0, -4.0]),
        _radar(34.0, [1034.0, 1.0]),
        _radar(36.0, [1036.0, 1.0]),
    ]
    output = RecoveryConfirmationPreprocessor().process(
        stream, np.arange(len(stream)),
    )
    assert [item.valid_flag for item in output] == [True, True, False, False, False, True]
    assert all(item.valid_flag for item in stream)
    assert output[-1].metadata["recovery_confirmation"] == "confirmed"
