from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from experiments.multimodal_cann_preprocessor import (
    MultimodalCANNPreprocessor, MultimodalCANNPreprocessorConfig,
)
from interfaces.data_objects import Observation, ObservationMessage


def preprocess_network_observation_messages(messages, *, timestamps, config=None):
    """Apply isolated CANNs per directed edge without creating messages."""
    messages = list(messages)
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    already_processed = [
        item for item in messages if item.metadata.get("cann_preprocessed", False)
    ]
    by_edge = defaultdict(list)
    for message in messages:
        if message.metadata.get("cann_preprocessed", False):
            continue
        by_edge[(message.observer_id, message.target_id)].append(message)
    output = list(already_processed)
    for edge, edge_messages in by_edge.items():
        output.extend(_process_edge(edge, edge_messages, times, config=config))
    return sorted(output, key=lambda item: (item.timestamp, item.information_id))


def _process_edge(edge, messages, timestamps, *, config=None):
    supported = {item.modality.lower() for item in messages} & {
        "radar", "infrared", "optical",
    }
    passthrough = [item for item in messages if item.modality.lower() not in supported]
    if not supported:
        return list(messages)
    dense, original_by_key = _dense_observations(edge, messages, timestamps, supported)
    selected = config or MultimodalCANNPreprocessorConfig()
    processor = MultimodalCANNPreprocessor(MultimodalCANNPreprocessorConfig(
        radar=selected.radar and "radar" in supported,
        infrared=selected.infrared and "infrared" in supported,
        optical=selected.optical and "optical" in supported,
        infrared_method=selected.infrared_method,
        infrared_coupled_config=selected.infrared_coupled_config,
    ))
    processed = processor.process(dense, timestamps)
    result = list(passthrough)
    for observation in processed:
        key = (observation.modality.lower(), float(observation.timestamp))
        original = original_by_key.get(key)
        if original is None:
            continue
        result.append(replace(
            original,
            measurement=np.asarray(observation.measurement, dtype=float).copy(),
            covariance=np.asarray(observation.covariance, dtype=float).copy(),
            confidence=float(observation.confidence),
            valid_flag=bool(observation.valid_flag),
            metadata={
                **original.metadata,
                **observation.metadata,
                "cann_preprocessed": True,
            },
        ))
    return result


def _dense_observations(edge, messages, timestamps, modalities):
    observer_id, target_id = edge
    source = {}
    for item in messages:
        key = (item.modality.lower(), float(item.timestamp))
        if key in source:
            raise ValueError(
                "CANN preprocessing requires at most one observation per "
                f"directed edge, modality, and timestamp; duplicate {edge + key}."
            )
        source[key] = item
    original_by_key = dict(source)
    dense = []
    for modality in sorted(modalities):
        examples = [item for item in messages if item.modality.lower() == modality]
        example = examples[0]
        for timestamp in timestamps:
            message = source.get((modality, float(timestamp)))
            dense.append(Observation(
                timestamp=float(timestamp), observer_id=observer_id,
                target_id=target_id, modality=example.modality,
                source_type="NETWORK_MESSAGE",
                measurement=(
                    np.zeros_like(example.measurement)
                    if message is None else np.asarray(message.measurement).copy()
                ),
                covariance=(
                    np.asarray(example.covariance).copy()
                    if message is None else np.asarray(message.covariance).copy()
                ),
                confidence=(1.0 if message is None else message.confidence),
                frame=example.frame,
                valid_flag=(False if message is None else message.valid_flag),
                metadata=(
                    {"network_placeholder": True}
                    if message is None else {
                        **message.metadata,
                        "observation_id": message.information_id,
                    }
                ),
            ))
    return dense, original_by_key
