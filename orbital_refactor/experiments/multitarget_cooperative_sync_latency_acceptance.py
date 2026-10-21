"""Integrated P09 latency test with the real target Consensus-CI path.

The timing boundary starts after every source node has produced its target
report batch.  It ends after a separate receiver process has received all
batches for the epoch, reconstructed ``TargetNodeReport`` objects, and run the
project's ``DistributedTargetConsensus`` implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import socket
from time import perf_counter, perf_counter_ns, sleep

import numpy as np

from cooperative.topology import ring_topology
from tracking import DistributedTargetConsensus, TargetNodeReport


@dataclass(frozen=True)
class CooperativeSyncLatencyRecord:
    frame_index: int
    source_node_count: int
    target_count: int
    datagram_count: int
    payload_bytes: int
    report_ready_to_consensus_applied_ms: float
    serialization_ms: float
    socket_send_ms: float
    receiver_collection_ms: float
    deserialization_ms: float
    report_reconstruction_ms: float
    consensus_ci_ms: float
    receiver_processing_ms: float
    received_message_count: int
    estimate_count: int
    rejected_message_count: int
    pending_message_count: int


@dataclass(frozen=True)
class CooperativeSyncLatencyReport:
    schema_version: str
    transport: str
    topology: str
    friendly_observer_count: int
    hostile_target_count: int
    frame_count: int
    sample_interval_seconds: float
    realtime_pacing: bool
    threshold_ms: float
    consensus_grid_points: int
    timing_clock: str
    local_estimation_included: bool
    consensus_ci_included: bool
    visualization_included: bool
    file_output_included: bool
    sender_receiver_process_separated: bool
    logical_node_batches_per_frame: int
    development_prescan: bool
    elapsed_wall_seconds: float
    protocol_clean: bool
    threshold_met: bool
    mean_sync_ms: float
    p50_sync_ms: float
    p95_sync_ms: float
    p99_sync_ms: float
    maximum_sync_ms: float
    mean_serialization_ms: float
    mean_socket_send_ms: float
    mean_receiver_collection_ms: float
    mean_deserialization_ms: float
    mean_report_reconstruction_ms: float
    mean_consensus_ci_ms: float
    mean_receiver_processing_ms: float
    mean_payload_bytes_per_frame: float
    records: tuple[CooperativeSyncLatencyRecord, ...]


def run_multitarget_cooperative_sync_latency_acceptance(
    *,
    frame_count: int = 120,
    friendly_observer_count: int = 15,
    hostile_target_count: int = 15,
    sample_interval_seconds: float = 1.0,
    threshold_ms: float = 100.0,
    consensus_grid_points: int = 31,
    realtime_pacing: bool = False,
    timeout_seconds: float = 30.0,
) -> CooperativeSyncLatencyReport:
    """Measure report-ready through real ring-topology Consensus-CI output."""

    if int(frame_count) <= 0:
        raise ValueError("frame_count must be positive.")
    if int(friendly_observer_count) < 3:
        raise ValueError("A ring test requires at least three observer nodes.")
    if int(hostile_target_count) <= 0:
        raise ValueError("hostile_target_count must be positive.")
    if float(sample_interval_seconds) <= 0.0:
        raise ValueError("sample_interval_seconds must be positive.")
    if float(threshold_ms) <= 0.0 or float(timeout_seconds) <= 0.0:
        raise ValueError("threshold_ms and timeout_seconds must be positive.")
    if int(consensus_grid_points) < 2:
        raise ValueError("consensus_grid_points must be at least two.")

    node_ids = tuple(
        f"friendly_{index + 1:02d}"
        for index in range(int(friendly_observer_count))
    )
    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    receiver = context.Process(
        target=_cooperative_receiver_main,
        args=(
            ready_queue,
            result_queue,
            int(frame_count),
            node_ids,
            int(hostile_target_count),
            int(consensus_grid_points),
            float(timeout_seconds),
        ),
    )
    receiver.start()
    sender = None
    records: list[CooperativeSyncLatencyRecord] = []
    protocol_clean = True
    started = perf_counter()
    try:
        ready = ready_queue.get(timeout=float(timeout_seconds))
        if ready.get("error"):
            raise RuntimeError(ready["error"])
        destination = ("127.0.0.1", int(ready["port"]))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        next_deadline = perf_counter()
        for frame_index in range(int(frame_count)):
            if realtime_pacing and frame_index:
                next_deadline += float(sample_interval_seconds)
                remaining = next_deadline - perf_counter()
                if remaining > 0.0:
                    sleep(remaining)

            messages = tuple(
                _node_batch_message(
                    frame_index=frame_index,
                    source_index=source_index,
                    observer_count=int(friendly_observer_count),
                    target_count=int(hostile_target_count),
                    sample_interval_seconds=float(sample_interval_seconds),
                )
                for source_index in range(int(friendly_observer_count))
            )
            reports_ready_ns = perf_counter_ns()
            serialization_ns = 0
            socket_send_ns = 0
            payload_bytes = 0
            for message in messages:
                stage_started_ns = perf_counter_ns()
                payload = _encode_datagram(message)
                serialization_ns += perf_counter_ns() - stage_started_ns
                stage_started_ns = perf_counter_ns()
                sent = sender.sendto(payload, destination)
                socket_send_ns += perf_counter_ns() - stage_started_ns
                payload_bytes += len(payload)
                if sent != len(payload):
                    protocol_clean = False

            received = result_queue.get(timeout=float(timeout_seconds))
            if received.get("error"):
                raise RuntimeError(received["error"])
            if int(received["frame_index"]) != frame_index:
                protocol_clean = False
                raise RuntimeError("Receiver returned an unexpected frame index.")
            records.append(CooperativeSyncLatencyRecord(
                frame_index=frame_index,
                source_node_count=int(received["source_node_count"]),
                target_count=int(received["target_count"]),
                datagram_count=int(received["datagram_count"]),
                payload_bytes=payload_bytes,
                report_ready_to_consensus_applied_ms=_ms(
                    int(received["applied_ns"]) - reports_ready_ns
                ),
                serialization_ms=_ms(serialization_ns),
                socket_send_ms=_ms(socket_send_ns),
                receiver_collection_ms=_ms(
                    int(received["last_received_ns"])
                    - int(received["first_received_ns"])
                ),
                deserialization_ms=_ms(int(received["deserialization_ns"])),
                report_reconstruction_ms=_ms(
                    int(received["report_reconstruction_ns"])
                ),
                consensus_ci_ms=_ms(int(received["consensus_ci_ns"])),
                receiver_processing_ms=_ms(
                    int(received["applied_ns"])
                    - int(received["first_received_ns"])
                ),
                received_message_count=int(received["received_message_count"]),
                estimate_count=int(received["estimate_count"]),
                rejected_message_count=int(received["rejected_message_count"]),
                pending_message_count=int(received["pending_message_count"]),
            ))
    except Empty as exc:
        protocol_clean = False
        raise RuntimeError(
            "Timed out waiting for the cooperative UDP receiver process."
        ) from exc
    finally:
        if sender is not None:
            sender.close()
        receiver.join(timeout=float(timeout_seconds))
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(timeout=2.0)
            protocol_clean = False
        ready_queue.close()
        result_queue.close()

    values = np.asarray([
        item.report_ready_to_consensus_applied_ms for item in records
    ], dtype=float)
    expected_estimates = int(friendly_observer_count) * int(hostile_target_count)
    expected_received = 2 * expected_estimates
    clean = bool(
        protocol_clean
        and receiver.exitcode == 0
        and len(records) == int(frame_count)
        and all(item.source_node_count == int(friendly_observer_count) for item in records)
        and all(item.target_count == int(hostile_target_count) for item in records)
        and all(item.datagram_count == int(friendly_observer_count) for item in records)
        and all(item.estimate_count == expected_estimates for item in records)
        and all(item.received_message_count == expected_received for item in records)
        and all(item.rejected_message_count == 0 for item in records)
        and all(item.pending_message_count == 0 for item in records)
    )
    return CooperativeSyncLatencyReport(
        schema_version="multitarget-cooperative-sync-latency.v1",
        transport=(
            "SAME_MACHINE_CROSS_PROCESS_IPV4_UDP_"
            f"{int(friendly_observer_count)}_LOGICAL_NODE_RING_TARGET_REPORT_JSON"
        ),
        topology="FIXED_UNDIRECTED_RING_ONE_HOP",
        friendly_observer_count=int(friendly_observer_count),
        hostile_target_count=int(hostile_target_count),
        frame_count=len(records),
        sample_interval_seconds=float(sample_interval_seconds),
        realtime_pacing=bool(realtime_pacing),
        threshold_ms=float(threshold_ms),
        consensus_grid_points=int(consensus_grid_points),
        timing_clock="time.perf_counter_ns_monotonic_process_shared",
        local_estimation_included=False,
        consensus_ci_included=True,
        visualization_included=False,
        file_output_included=False,
        sender_receiver_process_separated=True,
        logical_node_batches_per_frame=int(friendly_observer_count),
        development_prescan=True,
        elapsed_wall_seconds=float(perf_counter() - started),
        protocol_clean=clean,
        threshold_met=bool(clean and float(np.mean(values)) <= float(threshold_ms)),
        mean_sync_ms=float(np.mean(values)),
        p50_sync_ms=float(np.percentile(values, 50)),
        p95_sync_ms=float(np.percentile(values, 95)),
        p99_sync_ms=float(np.percentile(values, 99)),
        maximum_sync_ms=float(np.max(values)),
        mean_serialization_ms=_mean(records, "serialization_ms"),
        mean_socket_send_ms=_mean(records, "socket_send_ms"),
        mean_receiver_collection_ms=_mean(records, "receiver_collection_ms"),
        mean_deserialization_ms=_mean(records, "deserialization_ms"),
        mean_report_reconstruction_ms=_mean(records, "report_reconstruction_ms"),
        mean_consensus_ci_ms=_mean(records, "consensus_ci_ms"),
        mean_receiver_processing_ms=_mean(records, "receiver_processing_ms"),
        mean_payload_bytes_per_frame=_mean(records, "payload_bytes"),
        records=tuple(records),
    )


def save_multitarget_cooperative_sync_latency_report(
    report: CooperativeSyncLatencyReport,
    output_directory: str | Path,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = asdict(report)
    summary.pop("records")
    summary["records_file"] = "cooperative_sync_latency.csv"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output / "cooperative_sync_latency.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(asdict(report.records[0]).keys())
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in report.records)
    return output


def _cooperative_receiver_main(
    ready_queue,
    result_queue,
    frame_count,
    node_ids,
    target_count,
    consensus_grid_points,
    timeout,
):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.settimeout(float(timeout))
    consensus = DistributedTargetConsensus(
        scene_id="acceptance-30sat",
        topology=ring_topology(node_ids),
        grid_points=int(consensus_grid_points),
    )
    try:
        receiver.bind(("127.0.0.1", 0))
        ready_queue.put({"port": int(receiver.getsockname()[1])})
        expected_sources = set(node_ids)
        for expected_frame in range(int(frame_count)):
            messages_by_source = {}
            first_received_ns = None
            last_received_ns = None
            deserialization_ns = 0
            reconstruction_ns = 0
            for _ in node_ids:
                data, _source = receiver.recvfrom(65_535)
                received_ns = perf_counter_ns()
                if first_received_ns is None:
                    first_received_ns = received_ns
                last_received_ns = received_ns
                stage_started_ns = perf_counter_ns()
                message = _decode_datagram(data)
                deserialization_ns += perf_counter_ns() - stage_started_ns
                if int(message["frameIndex"]) != expected_frame:
                    raise ValueError("Received an unexpected frame index.")
                source_id = str(message["sourceNodeId"])
                if source_id in messages_by_source:
                    raise ValueError("Received a duplicate source batch.")
                messages_by_source[source_id] = message
            if set(messages_by_source) != expected_sources:
                raise ValueError("Source node batches are incomplete.")

            stage_started_ns = perf_counter_ns()
            reports = tuple(
                _target_node_report(item)
                for source_id in node_ids
                for item in messages_by_source[source_id]["reports"]
            )
            reconstruction_ns += perf_counter_ns() - stage_started_ns
            if len(reports) != len(node_ids) * int(target_count):
                raise ValueError("Target report coverage is incomplete.")

            timestamps = {
                float(message["timestamp"])
                for message in messages_by_source.values()
            }
            if len(timestamps) != 1:
                raise ValueError("Source node batches use inconsistent timestamps.")
            epoch_timestamp = timestamps.pop()
            consensus_started_ns = perf_counter_ns()
            result = consensus.step(
                timestamp=epoch_timestamp,
                local_reports=reports,
                active_source_ids=node_ids,
            )
            applied_ns = perf_counter_ns()
            estimate_count = sum(
                len(estimates) for estimates in result.estimates_by_node.values()
            )
            received_message_count = sum(
                len(message_ids)
                for message_ids in result.received_message_ids_by_node.values()
            )
            rejected_message_count = sum(
                len(message_ids)
                for message_ids in result.rejected_message_ids_by_node.values()
            )
            result_queue.put({
                "frame_index": expected_frame,
                "source_node_count": len(messages_by_source),
                "target_count": int(target_count),
                "datagram_count": len(messages_by_source),
                "first_received_ns": int(first_received_ns),
                "last_received_ns": int(last_received_ns),
                "deserialization_ns": int(deserialization_ns),
                "report_reconstruction_ns": int(reconstruction_ns),
                "consensus_ci_ns": int(applied_ns - consensus_started_ns),
                "applied_ns": int(applied_ns),
                "received_message_count": int(received_message_count),
                "estimate_count": int(estimate_count),
                "rejected_message_count": int(rejected_message_count),
                "pending_message_count": int(result.pending_message_count),
            })
    except Exception as exc:  # pragma: no cover - parent surfaces child errors
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        if receiver.getsockname()[1] == 0:
            ready_queue.put(payload)
        else:
            result_queue.put(payload)
    finally:
        receiver.close()


def _node_batch_message(
    *, frame_index, source_index, observer_count, target_count,
    sample_interval_seconds,
):
    source_id = f"friendly_{source_index + 1:02d}"
    timestamp = float(frame_index) * float(sample_interval_seconds)
    return {
        "schemaVersion": "TARGET-STATE-SYNC-COOPERATIVE-1.0",
        "messageType": "TARGET_NODE_REPORT_BATCH",
        "sceneId": "acceptance-30sat",
        "sourceNodeId": source_id,
        "frameIndex": int(frame_index),
        "timestamp": timestamp,
        "observerCount": int(observer_count),
        "reports": [
            {
                "observerId": source_id,
                "targetId": f"hostile_{target_index + 1:02d}",
                "trackId": f"track_hostile_{target_index + 1:02d}",
                "timestamp": timestamp,
                "stateEci": [
                    7_000_000.0 + 100_000.0 * target_index + source_index,
                    7_500.0 * timestamp + 2.0 * source_index,
                    -1_500.0 * target_index,
                    -10.0 + 0.01 * source_index,
                    7_500.0 + 0.02 * target_index,
                    2.0 - 0.01 * target_index,
                ],
                "covarianceEci": np.diag([
                    25.0 + source_index,
                    25.0 + source_index,
                    25.0 + source_index,
                    0.1,
                    0.1,
                    0.1,
                ]).reshape(-1).tolist(),
                "qualityScore": 0.9,
                "validFlag": True,
                "usedMeasurementIds": [
                    f"e{frame_index}:{source_id}:t{target_index + 1:02d}"
                ],
            }
            for target_index in range(int(target_count))
        ],
    }


def _target_node_report(item) -> TargetNodeReport:
    return TargetNodeReport(
        observer_id=str(item["observerId"]),
        target_id=str(item["targetId"]),
        track_id=str(item["trackId"]),
        timestamp=float(item["timestamp"]),
        state_eci=np.asarray(item["stateEci"], dtype=float),
        covariance_eci=np.asarray(
            item["covarianceEci"], dtype=float
        ).reshape(6, 6),
        quality_score=float(item["qualityScore"]),
        valid_flag=bool(item["validFlag"]),
        used_measurement_ids=tuple(item["usedMeasurementIds"]),
    )


def _encode_datagram(message) -> bytes:
    payload = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(payload) > 65_507:
        raise ValueError("One node batch exceeds the IPv4 UDP payload limit.")
    return payload


def _decode_datagram(payload: bytes):
    message = json.loads(payload.decode("utf-8"))
    if message.get("schemaVersion") != "TARGET-STATE-SYNC-COOPERATIVE-1.0":
        raise ValueError("Unexpected cooperative synchronization schema.")
    return message


def _ms(nanoseconds: int) -> float:
    return float(nanoseconds) / 1_000_000.0


def _mean(records, field: str) -> float:
    return float(np.mean([float(getattr(item, field)) for item in records]))
