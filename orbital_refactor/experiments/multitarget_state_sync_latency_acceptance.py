"""P09 multi-target state synchronization latency development prescan."""

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


@dataclass(frozen=True)
class MultiTargetStateSyncRecord:
    frame_index: int
    target_count: int
    payload_bytes: int
    construction_to_applied_ms: float
    serialization_ms: float
    socket_send_ms: float
    send_to_receive_ms: float
    deserialization_ms: float
    identity_match_ms: float
    state_application_ms: float
    receive_to_applied_ms: float


@dataclass(frozen=True)
class MultiTargetStateSyncReport:
    schema_version: str
    transport: str
    friendly_observer_count: int
    hostile_target_count: int
    frame_count: int
    sample_interval_seconds: float
    realtime_pacing: bool
    threshold_ms: float
    timing_clock: str
    algorithm_processing_included: bool
    visualization_included: bool
    file_output_included: bool
    sender_receiver_process_separated: bool
    development_prescan: bool
    elapsed_wall_seconds: float
    protocol_clean: bool
    threshold_met: bool
    mean_construction_to_applied_ms: float
    p50_construction_to_applied_ms: float
    p95_construction_to_applied_ms: float
    p99_construction_to_applied_ms: float
    maximum_construction_to_applied_ms: float
    mean_serialization_ms: float
    mean_socket_send_ms: float
    mean_send_to_receive_ms: float
    mean_deserialization_ms: float
    mean_identity_match_ms: float
    mean_state_application_ms: float
    mean_receive_to_applied_ms: float
    mean_payload_bytes: float
    records: tuple[MultiTargetStateSyncRecord, ...]


def run_multitarget_state_sync_latency_acceptance(
    *,
    frame_count: int = 120,
    friendly_observer_count: int = 15,
    hostile_target_count: int = 15,
    sample_interval_seconds: float = 1.0,
    threshold_ms: float = 100.0,
    realtime_pacing: bool = False,
    timeout_seconds: float = 10.0,
) -> MultiTargetStateSyncReport:
    """Measure a 15-target report from construction through receiver application.

    The estimator is deliberately outside the timing boundary.  A real IPv4 UDP
    socket and a separate receiver process exercise compact target-report JSON
    serialization, transport, deserialization, target identity matching, and
    receiver state application.  Fifteen friendly nodes rotate over a fixed ring
    link, while every batch contains all fifteen hostile-target posteriors.
    """

    if int(frame_count) <= 0:
        raise ValueError("frame_count must be positive.")
    if int(friendly_observer_count) <= 0 or int(hostile_target_count) <= 0:
        raise ValueError("Observer and target counts must be positive.")
    if float(sample_interval_seconds) <= 0.0:
        raise ValueError("sample_interval_seconds must be positive.")
    if float(threshold_ms) <= 0.0 or float(timeout_seconds) <= 0.0:
        raise ValueError("threshold_ms and timeout_seconds must be positive.")

    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    receiver = context.Process(
        target=_receiver_main,
        args=(
            ready_queue,
            result_queue,
            int(frame_count),
            int(hostile_target_count),
            float(timeout_seconds),
        ),
    )
    receiver.start()
    sender = None
    records = []
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
            message = _target_report_message(
                frame_index=frame_index,
                observer_count=int(friendly_observer_count),
                target_count=int(hostile_target_count),
                sample_interval_seconds=float(sample_interval_seconds),
            )
            constructed_ns = perf_counter_ns()
            serialization_started_ns = perf_counter_ns()
            payload = _encode_target_report_datagram(message)
            serialization_done_ns = perf_counter_ns()
            send_started_ns = perf_counter_ns()
            sent = sender.sendto(payload, destination)
            send_done_ns = perf_counter_ns()
            if sent != len(payload):
                protocol_clean = False
            received = result_queue.get(timeout=float(timeout_seconds))
            if received.get("error"):
                raise RuntimeError(received["error"])
            if int(received["frame_index"]) != frame_index:
                protocol_clean = False
                raise RuntimeError("Receiver returned an unexpected frame index.")
            records.append(MultiTargetStateSyncRecord(
                frame_index=frame_index,
                target_count=int(received["target_count"]),
                payload_bytes=len(payload),
                construction_to_applied_ms=_ms(
                    int(received["applied_ns"]) - constructed_ns
                ),
                serialization_ms=_ms(
                    serialization_done_ns - serialization_started_ns
                ),
                socket_send_ms=_ms(send_done_ns - send_started_ns),
                send_to_receive_ms=_ms(
                    int(received["received_ns"]) - send_done_ns
                ),
                deserialization_ms=_ms(
                    int(received["decoded_ns"]) - int(received["received_ns"])
                ),
                identity_match_ms=_ms(
                    int(received["matched_ns"]) - int(received["decoded_ns"])
                ),
                state_application_ms=_ms(
                    int(received["applied_ns"]) - int(received["matched_ns"])
                ),
                receive_to_applied_ms=_ms(
                    int(received["applied_ns"]) - int(received["received_ns"])
                ),
            ))
    except Empty as exc:
        protocol_clean = False
        raise RuntimeError("Timed out waiting for the UDP receiver process.") from exc
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
        item.construction_to_applied_ms for item in records
    ], dtype=float)
    return MultiTargetStateSyncReport(
        schema_version="multitarget-state-sync-latency.v1",
        transport="SAME_MACHINE_CROSS_PROCESS_IPV4_UDP_TARGET_REPORT_JSON",
        friendly_observer_count=int(friendly_observer_count),
        hostile_target_count=int(hostile_target_count),
        frame_count=len(records),
        sample_interval_seconds=float(sample_interval_seconds),
        realtime_pacing=bool(realtime_pacing),
        threshold_ms=float(threshold_ms),
        timing_clock="time.perf_counter_ns_monotonic_process_shared",
        algorithm_processing_included=False,
        visualization_included=False,
        file_output_included=False,
        sender_receiver_process_separated=True,
        development_prescan=True,
        elapsed_wall_seconds=float(perf_counter() - started),
        protocol_clean=bool(
            protocol_clean
            and receiver.exitcode == 0
            and len(records) == int(frame_count)
            and all(item.target_count == int(hostile_target_count) for item in records)
        ),
        threshold_met=bool(float(np.mean(values)) <= float(threshold_ms)),
        mean_construction_to_applied_ms=float(np.mean(values)),
        p50_construction_to_applied_ms=float(np.percentile(values, 50)),
        p95_construction_to_applied_ms=float(np.percentile(values, 95)),
        p99_construction_to_applied_ms=float(np.percentile(values, 99)),
        maximum_construction_to_applied_ms=float(np.max(values)),
        mean_serialization_ms=_mean(records, "serialization_ms"),
        mean_socket_send_ms=_mean(records, "socket_send_ms"),
        mean_send_to_receive_ms=_mean(records, "send_to_receive_ms"),
        mean_deserialization_ms=_mean(records, "deserialization_ms"),
        mean_identity_match_ms=_mean(records, "identity_match_ms"),
        mean_state_application_ms=_mean(records, "state_application_ms"),
        mean_receive_to_applied_ms=_mean(records, "receive_to_applied_ms"),
        mean_payload_bytes=_mean(records, "payload_bytes"),
        records=tuple(records),
    )


def save_multitarget_state_sync_latency_report(
    report: MultiTargetStateSyncReport,
    output_directory: str | Path,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = asdict(report)
    summary.pop("records")
    summary["records_file"] = "state_sync_latency.csv"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output / "state_sync_latency.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(asdict(report.records[0]).keys())
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in report.records)
    return output


def _receiver_main(ready_queue, result_queue, frame_count, target_count, timeout):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.settimeout(float(timeout))
    state_by_receiver_target = {}
    identity_by_source_target = {}
    try:
        receiver.bind(("127.0.0.1", 0))
        ready_queue.put({"port": int(receiver.getsockname()[1])})
        for _ in range(int(frame_count)):
            data, _source = receiver.recvfrom(65_535)
            received_ns = perf_counter_ns()
            message = _decode_target_report_datagram(data)
            decoded_ns = perf_counter_ns()
            reports = message["reports"]
            if len(reports) != int(target_count):
                raise ValueError("Received target count does not match the scenario.")
            source_id = str(message["sourceNodeId"])
            receiver_id = str(message["destinationNodeId"])
            if source_id == receiver_id:
                raise ValueError("Source and destination nodes must differ.")
            matched = []
            for item in reports:
                target_id = str(item["targetId"])
                track_id = str(item["trackId"])
                identity_key = (source_id, target_id)
                previous = identity_by_source_target.setdefault(
                    identity_key, track_id
                )
                if previous != track_id:
                    raise ValueError("Target identity changed across frames.")
                matched.append((target_id, item))
            if len({target_id for target_id, _item in matched}) != int(target_count):
                raise ValueError("Target identities are incomplete or duplicated.")
            matched_ns = perf_counter_ns()
            for target_id, item in matched:
                state = tuple(float(value) for value in item["stateEci"])
                covariance = tuple(float(value) for value in item["covarianceEci"])
                if len(state) != 6 or len(covariance) != 36:
                    raise ValueError("Target state or covariance has invalid shape.")
                state_by_receiver_target[(receiver_id, target_id)] = (
                    state, covariance, float(item["qualityScore"])
                )
            applied_ns = perf_counter_ns()
            result_queue.put({
                "frame_index": int(message["frameIndex"]),
                "target_count": len(matched),
                "received_ns": received_ns,
                "decoded_ns": decoded_ns,
                "matched_ns": matched_ns,
                "applied_ns": applied_ns,
            })
    except Exception as exc:  # pragma: no cover - parent surfaces subprocess errors
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        if receiver.getsockname()[1] == 0:
            ready_queue.put(payload)
        else:
            result_queue.put(payload)
    finally:
        receiver.close()


def _target_report_message(
    *, frame_index, observer_count, target_count, sample_interval_seconds,
):
    source_index = int(frame_index) % int(observer_count)
    destination_index = (source_index + 1) % int(observer_count)
    timestamp = float(frame_index) * float(sample_interval_seconds)
    return {
        "schemaVersion": "TARGET-STATE-SYNC-1.0",
        "messageType": "TARGET_ESTIMATE_BATCH",
        "sceneId": "acceptance-30sat",
        "sourceNodeId": f"friendly_{source_index + 1:02d}",
        "destinationNodeId": f"friendly_{destination_index + 1:02d}",
        "frameIndex": int(frame_index),
        "timestamp": timestamp,
        "frame": "J2000",
        "reports": [
            {
                "targetId": f"hostile_{index + 1:02d}",
                "trackId": f"track_hostile_{index + 1:02d}",
                "stateEci": [
                    7_000_000.0 + 1_000.0 * index + frame_index,
                    2_000.0 * index - 0.5 * frame_index,
                    -1_500.0 * index + 0.25 * frame_index,
                    -10.0 + 0.01 * index,
                    7_500.0 + 0.02 * index,
                    2.0 - 0.01 * index,
                ],
                "covarianceEci": _covariance_values(index),
                "qualityScore": 0.9 - 0.01 * (index % 5),
                "validFlag": True,
            }
            for index in range(int(target_count))
        ],
    }


def _encode_target_report_datagram(message):
    if message.get("schemaVersion") != "TARGET-STATE-SYNC-1.0":
        raise ValueError("Unexpected target-report schemaVersion.")
    if message.get("messageType") != "TARGET_ESTIMATE_BATCH":
        raise ValueError("Unexpected target-report messageType.")
    encoded = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 65_506:
        raise ValueError("Target-report UDP payload exceeds the datagram limit.")
    return encoded


def _decode_target_report_datagram(data):
    try:
        message = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid target-report UDP JSON.") from exc
    required = {
        "schemaVersion", "messageType", "sceneId", "sourceNodeId",
        "destinationNodeId", "frameIndex", "timestamp", "frame", "reports",
    }
    if not isinstance(message, dict) or required - set(message):
        raise ValueError("Target-report UDP message is incomplete.")
    if message["schemaVersion"] != "TARGET-STATE-SYNC-1.0":
        raise ValueError("Unexpected target-report schemaVersion.")
    if message["messageType"] != "TARGET_ESTIMATE_BATCH":
        raise ValueError("Unexpected target-report messageType.")
    if not isinstance(message["reports"], list):
        raise ValueError("Target-report batch must contain a report list.")
    return message


def _covariance_values(index):
    diagonal = (
        4.0 + 0.1 * index,
        4.5 + 0.1 * index,
        5.0 + 0.1 * index,
        0.01,
        0.01,
        0.015,
    )
    return [
        float(diagonal[row]) if row == column else 0.0
        for row in range(6)
        for column in range(6)
    ]


def _ms(nanoseconds):
    return max(0.0, float(nanoseconds) / 1_000_000.0)


def _mean(records, field):
    return float(np.mean([float(getattr(item, field)) for item in records]))
