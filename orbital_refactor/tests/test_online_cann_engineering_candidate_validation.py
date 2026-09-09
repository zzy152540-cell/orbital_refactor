import numpy as np

from experiments.online_cann_engineering_candidate_validation import (
    run_online_cann_engineering_candidate_validation,
)


def test_engineering_candidate_preserves_filter_contract():
    results = run_online_cann_engineering_candidate_validation(
        seed=4, episode_epochs=2, dt=0.2, include_walker=False,
    )
    assert len(results) == 4
    assert all(item.base_filter_identical for item in results)
    assert all(item.base_tensor_prefix_identical for item in results)
    assert all(item.reference_runtime_seconds > 0.0 for item in results)
    assert all(item.candidate_runtime_seconds > 0.0 for item in results)
    assert all(np.isfinite(item.maximum_cann_feature_difference)
               for item in results)
