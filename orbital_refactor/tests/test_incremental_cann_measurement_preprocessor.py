import numpy as np

from brain_inspired.incremental_measurement_preprocessor import (
    IncrementalMultimodalCANNConfig,
    IncrementalMultimodalCANNPreprocessor,
)
from experiments.single_satellite_cann_comparison import (
    _preprocess_valid_infrared_azimuth_with_cann,
    _preprocess_valid_optical_with_plane_cann,
    _preprocess_valid_radar_with_line_cann,
)
from interfaces.data_objects import Observation


def _observations(
    modality, measurements, timestamps, *, valid=None, confidence=1.0,
):
    flags = np.ones(len(timestamps), dtype=bool) if valid is None else valid
    return [
        Observation(
            timestamp=float(timestamp), observer_id="observer-1",
            target_id="target-1", modality=modality,
            source_type="TEST", measurement=np.asarray(value, dtype=float),
            covariance=np.eye(2), confidence=float(confidence), frame="SPRI",
            valid_flag=bool(flag), metadata={"observation_id": f"{modality}-{index}"},
        )
        for index, (timestamp, value, flag) in enumerate(
            zip(timestamps, measurements, flags)
        )
    ]


def _run_incrementally(processor, observations):
    return [
        processor.process([item], np.asarray([item.timestamp]))[0]
        for item in observations
    ]


def test_incremental_radar_matches_existing_causal_batch_path():
    timestamps = np.arange(5.0)
    measurements = np.array([
        [1_000_000.0, 100.0],
        [1_000_100.0, 100.0],
        [1_000_200.0, 100.0],
        [1_003_300.0, 105.0],
        [1_000_400.0, 100.0],
    ])
    source = _observations("RADAR", measurements, timestamps)
    expected = _preprocess_valid_radar_with_line_cann(source, timestamps)
    actual = _run_incrementally(
        IncrementalMultimodalCANNPreprocessor(
            IncrementalMultimodalCANNConfig(
                infrared=False,
                optical=False,
                prefer_trusted_measurements=False,
            ),
        ),
        source,
    )

    np.testing.assert_allclose(
        [item.measurement for item in actual],
        [item.measurement for item in expected],
        atol=1e-8,
    )
    assert [item.valid_flag for item in actual] == [
        item.valid_flag for item in expected
    ]
    assert [
        item.metadata.get("cann_diagnostics", {}).get("range_substituted", False)
        for item in actual
    ] == [
        item.metadata.get("cann_diagnostics", {}).get("range_substituted", False)
        for item in expected
    ]


def test_incremental_optical_matches_existing_causal_batch_path():
    timestamps = np.arange(5.0)
    measurements = np.array([
        [0.00, 0.00], [0.01, 0.01], [0.02, 0.02],
        [0.30, 0.40], [0.04, 0.04],
    ])
    source = _observations("OPTICAL", measurements, timestamps)
    expected = _preprocess_valid_optical_with_plane_cann(source, timestamps)
    actual = _run_incrementally(
        IncrementalMultimodalCANNPreprocessor(
            IncrementalMultimodalCANNConfig(
                infrared=False,
                radar=False,
                prefer_trusted_measurements=False,
            ),
        ),
        source,
    )

    np.testing.assert_allclose(
        [item.measurement for item in actual],
        [item.measurement for item in expected],
        atol=1e-10,
    )
    assert [
        item.metadata.get("cann_diagnostics", {}).get("u_substituted", False)
        for item in actual
    ] == [
        item.metadata.get("cann_diagnostics", {}).get("u_substituted", False)
        for item in expected
    ]


def test_incremental_infrared_matches_existing_hybrid_batch_path():
    timestamps = np.arange(5.0)
    measurements = np.deg2rad(np.array([
        [10.0, 2.0], [10.1, 2.1], [10.2, 2.2],
        [15.0, 8.0], [10.4, 2.4],
    ]))
    source = _observations("INFRARED", measurements, timestamps)
    config = IncrementalMultimodalCANNConfig(
        radar=False, optical=False, prefer_trusted_measurements=False,
    )
    expected = _preprocess_valid_infrared_azimuth_with_cann(
        source, timestamps, method="hybrid_ring_line_cann",
        coupled_config=config.resolved_infrared_config(),
    )
    actual = _run_incrementally(
        IncrementalMultimodalCANNPreprocessor(config), source,
    )

    np.testing.assert_allclose(
        [item.measurement for item in actual],
        [item.measurement for item in expected],
        atol=1e-9,
    )
    assert [item.valid_flag for item in actual] == [
        item.valid_flag for item in expected
    ]


def test_incremental_processor_accepts_sparse_modalities_and_records_cost():
    processor = IncrementalMultimodalCANNPreprocessor()
    first = (
        _observations("RADAR", [[1_000_000.0, 100.0]], [0.0])
        + _observations("INFRARED", [np.deg2rad([10.0, 2.0])], [0.0])
    )
    second = _observations("RADAR", [[1_000_100.0, 100.0]], [1.0])

    first_result = processor.process(first, np.asarray([0.0]))
    second_result = processor.process(second, np.asarray([1.0]))

    assert len(first_result) == 2 and len(second_result) == 1
    assert processor.performance.call_count == 2
    assert processor.performance.observation_count == 3
    assert processor.performance.total_seconds > 0.0
    assert set(processor.performance.seconds_by_modality) == {"RADAR", "INFRARED"}


def test_realtime_profile_reduces_ring_integration_resolution_explicitly():
    realtime = IncrementalMultimodalCANNConfig(profile="realtime")
    research = IncrementalMultimodalCANNConfig(profile="research")

    real_ring = realtime.resolved_infrared_config().ring
    research_ring = research.resolved_infrared_config().ring

    assert (real_ring.num_neurons, real_ring.internal_dt) == (90, 0.002)
    assert (research_ring.num_neurons, research_ring.internal_dt) == (180, 0.001)


def test_high_confidence_radar_reanchors_immediately_after_short_dropout():
    processor = IncrementalMultimodalCANNPreprocessor(
        IncrementalMultimodalCANNConfig(infrared=False, optical=False),
    )
    source = _observations(
        "RADAR",
        [[1_000_000.0, 100.0], [1_000_100.0, 100.0], [1_000_500.0, 100.0]],
        [0.0, 1.0, 5.0],
    )

    actual = _run_incrementally(processor, source)
    recovered = actual[-1]
    diagnostics = recovered.metadata["cann_diagnostics"]

    assert recovered.valid_flag
    np.testing.assert_allclose(recovered.measurement, source[-1].measurement)
    assert diagnostics["radar_reanchored"]
    assert not diagnostics["recovery_pending"]
    assert not diagnostics["range_substituted"]


def test_low_confidence_radar_outlier_can_still_be_substituted_by_cann():
    processor = IncrementalMultimodalCANNPreprocessor(
        IncrementalMultimodalCANNConfig(infrared=False, optical=False),
    )
    source = _observations(
        "RADAR",
        [
            [1_000_000.0, 100.0],
            [1_000_100.0, 100.0],
            [1_000_200.0, 100.0],
            [1_010_300.0, 105.0],
        ],
        [0.0, 1.0, 2.0, 3.0],
        confidence=0.5,
    )

    actual = _run_incrementally(processor, source)
    diagnostics = actual[-1].metadata["cann_diagnostics"]

    assert diagnostics["range_substituted"]
    assert diagnostics["range_rate_substituted"]
    assert not np.array_equal(actual[-1].measurement, source[-1].measurement)
