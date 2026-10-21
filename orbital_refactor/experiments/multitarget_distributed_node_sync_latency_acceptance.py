"""P09 timing with one independent process per logical satellite node."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import socket
from time import perf_counter, perf_counter_ns, sleep

import numpy as np

from cooperative.topology import ring_topology
from tracking import TargetNodeReport, fuse_target_reports

from experiments.multitarget_cooperative_sync_latency_acceptance import (
    _decode_datagram,
    _encode_datagram,
    _node_batch_message,
    _target_node_report,
)


@dataclass(frozen=True)
class DistributedNodeSyncRecord:
    frame_index: int
    node_id: str
    target_count: int
    neighbor_batch_count: int
    payload_bytes_sent: int
    report_ready_to_local_consensus_applied_ms: float
    serialization_ms: float
    socket_send_ms: float
    neighbor_receive_and_decode_ms: float
    report_reconstruction_ms: float
    consensus_ci_ms: float
    estimate_count: int


@dataclass(frozen=True)
class DistributedNodeSyncReport:
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
    worker_linear_algebra_threads: int
    timing_clock: str
    local_estimation_included: bool
    consensus_ci_included: bool
    visualization_included: bool
    file_output_included: bool
    independent_node_processes: bool
    process_count: int
    development_prescan: bool
    elapsed_wall_seconds: float
    protocol_clean: bool
    threshold_met: bool
    node_epoch_sample_count: int
    mean_sync_ms: float
    p50_sync_ms: float
    p95_sync_ms: float
    p99_sync_ms: float
    maximum_sync_ms: float
    mean_frame_all_nodes_applied_ms: float
    p95_frame_all_nodes_applied_ms: float
    maximum_frame_all_nodes_applied_ms: float
    mean_serialization_ms: float
    mean_socket_send_ms: float
    mean_neighbor_receive_and_decode_ms: float
    mean_report_reconstruction_ms: float
    mean_consensus_ci_ms: float
    mean_payload_bytes_sent_per_node_epoch: float
    records: tuple[DistributedNodeSyncRecord, ...]


def run_multitarget_distributed_node_sync_latency_acceptance(
    *,
    frame_count: int = 120,
    friendly_observer_count: int = 15,
    hostile_target_count: int = 15,
    sample_interval_seconds: float = 1.0,
    threshold_ms: float = 100.0,
    consensus_grid_points: int = 31,
    worker_linear_algebra_threads: int = 1,
    realtime_pacing: bool = False,
    timeout_seconds: float = 60.0,
) -> DistributedNodeSyncReport:
    """Measure real neighbor exchange and local CI in independent processes."""

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
    if int(worker_linear_algebra_threads) < 1:
        raise ValueError("worker_linear_algebra_threads must be positive.")

    node_ids = tuple(
        f"friendly_{index + 1:02d}"
        for index in range(int(friendly_observer_count))
    )
    topology = ring_topology(node_ids)
    context = mp.get_context("spawn")
    start_event = context.Event()
    ready_queue = context.Queue()
    prepared_queue = context.Queue()
    result_queue = context.Queue()
    task_queues = {node_id: context.Queue() for node_id in node_ids}
    workers = [
        context.Process(
            target=_node_worker_main,
            args=(
                node_id,
                task_queues[node_id],
                ready_queue,
                prepared_queue,
                result_queue,
                start_event,
                int(hostile_target_count),
                int(consensus_grid_points),
                float(timeout_seconds),
            ),
        )
        for node_id in node_ids
    ]
    _start_workers_with_thread_limit(
        workers, int(worker_linear_algebra_threads)
    )

    records: list[DistributedNodeSyncRecord] = []
    frame_barrier_ms: list[float] = []
    protocol_clean = True
    started = perf_counter()
    try:
        ports = {}
        for _ in node_ids:
            item = ready_queue.get(timeout=float(timeout_seconds))
            _raise_worker_error(item)
            ports[str(item["node_id"])] = int(item["port"])
        if set(ports) != set(node_ids):
            raise RuntimeError("Node process socket registration is incomplete.")

        for node_id in node_ids:
            task_queues[node_id].put({
                "kind": "configure",
                "neighbors": {
                    neighbor_id: ports[neighbor_id]
                    for neighbor_id in topology.neighbors(node_id)
                },
            })
        configured = set()
        for _ in node_ids:
            item = ready_queue.get(timeout=float(timeout_seconds))
            _raise_worker_error(item)
            configured.add(str(item["node_id"]))
        if configured != set(node_ids):
            raise RuntimeError("Node process topology configuration is incomplete.")

        next_deadline = perf_counter()
        for frame_index in range(int(frame_count)):
            if realtime_pacing and frame_index:
                next_deadline += float(sample_interval_seconds)
                remaining = next_deadline - perf_counter()
                if remaining > 0.0:
                    sleep(remaining)
            start_event.clear()
            for source_index, node_id in enumerate(node_ids):
                task_queues[node_id].put({
                    "kind": "frame",
                    "message": _node_batch_message(
                        frame_index=frame_index,
                        source_index=source_index,
                        observer_count=int(friendly_observer_count),
                        target_count=int(hostile_target_count),
                        sample_interval_seconds=float(sample_interval_seconds),
                    ),
                })

            prepared = set()
            for _ in node_ids:
                item = prepared_queue.get(timeout=float(timeout_seconds))
                _raise_worker_error(item)
                if int(item["frame_index"]) != frame_index:
                    raise RuntimeError("A node prepared an unexpected frame.")
                prepared.add(str(item["node_id"]))
            if prepared != set(node_ids):
                raise RuntimeError("Not all nodes prepared the current frame.")

            reports_ready_ns = perf_counter_ns()
            start_event.set()
            frame_results = []
            for _ in node_ids:
                item = result_queue.get(timeout=float(timeout_seconds))
                _raise_worker_error(item)
                if int(item["frame_index"]) != frame_index:
                    raise RuntimeError("A node completed an unexpected frame.")
                frame_results.append(item)
            start_event.clear()

            completed_nodes = {str(item["node_id"]) for item in frame_results}
            if completed_nodes != set(node_ids):
                raise RuntimeError("Not all nodes completed the current frame.")
            frame_barrier_ms.append(_ms(
                max(int(item["applied_ns"]) for item in frame_results)
                - reports_ready_ns
            ))
            for item in frame_results:
                record = DistributedNodeSyncRecord(
                    frame_index=frame_index,
                    node_id=str(item["node_id"]),
                    target_count=int(item["target_count"]),
                    neighbor_batch_count=int(item["neighbor_batch_count"]),
                    payload_bytes_sent=int(item["payload_bytes_sent"]),
                    report_ready_to_local_consensus_applied_ms=_ms(
                        int(item["applied_ns"]) - reports_ready_ns
                    ),
                    serialization_ms=_ms(int(item["serialization_ns"])),
                    socket_send_ms=_ms(int(item["socket_send_ns"])),
                    neighbor_receive_and_decode_ms=_ms(
                        int(item["neighbor_receive_and_decode_ns"])
                    ),
                    report_reconstruction_ms=_ms(
                        int(item["report_reconstruction_ns"])
                    ),
                    consensus_ci_ms=_ms(int(item["consensus_ci_ns"])),
                    estimate_count=int(item["estimate_count"]),
                )
                records.append(record)
                protocol_clean = bool(
                    protocol_clean
                    and record.target_count == int(hostile_target_count)
                    and record.neighbor_batch_count == 2
                    and record.estimate_count == int(hostile_target_count)
                )
    except Empty as exc:
        protocol_clean = False
        raise RuntimeError(
            "Timed out waiting for a distributed node process."
        ) from exc
    finally:
        start_event.set()
        for node_id in node_ids:
            task_queues[node_id].put({"kind": "stop"})
        for worker in workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
                protocol_clean = False
        for queue in (*task_queues.values(), ready_queue, prepared_queue, result_queue):
            queue.close()

    sync_values = np.asarray([
        item.report_ready_to_local_consensus_applied_ms for item in records
    ], dtype=float)
    barrier_values = np.asarray(frame_barrier_ms, dtype=float)
    clean = bool(
        protocol_clean
        and all(worker.exitcode == 0 for worker in workers)
        and len(records) == int(frame_count) * int(friendly_observer_count)
    )
    return DistributedNodeSyncReport(
        schema_version="multitarget-distributed-node-sync-latency.v1",
        transport=(
            "SAME_MACHINE_INDEPENDENT_NODE_PROCESSES_IPV4_UDP_"
            f"{int(friendly_observer_count)}_NODE_RING_TARGET_REPORT_JSON"
        ),
        topology="FIXED_UNDIRECTED_RING_ONE_HOP",
        friendly_observer_count=int(friendly_observer_count),
        hostile_target_count=int(hostile_target_count),
        frame_count=int(frame_count),
        sample_interval_seconds=float(sample_interval_seconds),
        realtime_pacing=bool(realtime_pacing),
        threshold_ms=float(threshold_ms),
        consensus_grid_points=int(consensus_grid_points),
        worker_linear_algebra_threads=int(worker_linear_algebra_threads),
        timing_clock="time.perf_counter_ns_monotonic_process_shared",
        local_estimation_included=False,
        consensus_ci_included=True,
        visualization_included=False,
        file_output_included=False,
        independent_node_processes=True,
        process_count=int(friendly_observer_count),
        development_prescan=True,
        elapsed_wall_seconds=float(perf_counter() - started),
        protocol_clean=clean,
        threshold_met=bool(clean and float(np.mean(sync_values)) <= float(threshold_ms)),
        node_epoch_sample_count=len(records),
        mean_sync_ms=float(np.mean(sync_values)),
        p50_sync_ms=float(np.percentile(sync_values, 50)),
        p95_sync_ms=float(np.percentile(sync_values, 95)),
        p99_sync_ms=float(np.percentile(sync_values, 99)),
        maximum_sync_ms=float(np.max(sync_values)),
        mean_frame_all_nodes_applied_ms=float(np.mean(barrier_values)),
        p95_frame_all_nodes_applied_ms=float(np.percentile(barrier_values, 95)),
        maximum_frame_all_nodes_applied_ms=float(np.max(barrier_values)),
        mean_serialization_ms=_mean(records, "serialization_ms"),
        mean_socket_send_ms=_mean(records, "socket_send_ms"),
        mean_neighbor_receive_and_decode_ms=_mean(
            records, "neighbor_receive_and_decode_ms"
        ),
        mean_report_reconstruction_ms=_mean(records, "report_reconstruction_ms"),
        mean_consensus_ci_ms=_mean(records, "consensus_ci_ms"),
        mean_payload_bytes_sent_per_node_epoch=_mean(records, "payload_bytes_sent"),
        records=tuple(records),
    )


def save_multitarget_distributed_node_sync_latency_report(
    report: DistributedNodeSyncReport,
    output_directory: str | Path,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = asdict(report)
    summary.pop("records")
    summary["records_file"] = "distributed_node_sync_latency.csv"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output / "distributed_node_sync_latency.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(asdict(report.records[0]).keys())
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in report.records)
    return output


def _node_worker_main(
    node_id,
    task_queue,
    ready_queue,
    prepared_queue,
    result_queue,
    start_event,
    target_count,
    consensus_grid_points,
    timeout,
):
    transport = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    transport.settimeout(float(timeout))
    try:
        transport.bind(("127.0.0.1", 0))
        ready_queue.put({
            "node_id": node_id,
            "port": int(transport.getsockname()[1]),
        })
        configuration = task_queue.get(timeout=float(timeout))
        if configuration.get("kind") != "configure":
            raise ValueError("Node did not receive its topology configuration.")
        neighbor_ports = {
            str(peer_id): int(port)
            for peer_id, port in configuration["neighbors"].items()
        }
        if len(neighbor_ports) != 2:
            raise ValueError("Every ring node must have exactly two neighbors.")
        ready_queue.put({"node_id": node_id, "configured": True})

        while True:
            task = task_queue.get(timeout=float(timeout))
            if task.get("kind") == "stop":
                break
            if task.get("kind") != "frame":
                raise ValueError("Node received an unsupported task.")
            message = task["message"]
            frame_index = int(message["frameIndex"])
            local_reports = tuple(
                _target_node_report(item) for item in message["reports"]
            )
            local_by_target = {report.target_id: report for report in local_reports}
            if len(local_by_target) != int(target_count):
                raise ValueError("Local target report coverage is incomplete.")
            prepared_queue.put({"node_id": node_id, "frame_index": frame_index})
            start_event.wait(timeout=float(timeout))
            if not start_event.is_set():
                raise TimeoutError("Timed out waiting for the synchronized start.")

            stage_started_ns = perf_counter_ns()
            payload = _encode_datagram(message)
            serialization_ns = perf_counter_ns() - stage_started_ns
            socket_send_ns = 0
            for port in neighbor_ports.values():
                stage_started_ns = perf_counter_ns()
                sent = transport.sendto(payload, ("127.0.0.1", port))
                socket_send_ns += perf_counter_ns() - stage_started_ns
                if sent != len(payload):
                    raise RuntimeError("UDP socket sent a partial node batch.")

            stage_started_ns = perf_counter_ns()
            incoming_by_source = {}
            for _ in neighbor_ports:
                data, _source = transport.recvfrom(65_535)
                incoming = _decode_datagram(data)
                if int(incoming["frameIndex"]) != frame_index:
                    raise ValueError("Received a neighbor batch for another frame.")
                source_id = str(incoming["sourceNodeId"])
                if source_id not in neighbor_ports:
                    raise ValueError("Received a batch from a non-neighbor node.")
                if source_id in incoming_by_source:
                    raise ValueError("Received a duplicate neighbor batch.")
                incoming_by_source[source_id] = incoming
            receive_and_decode_ns = perf_counter_ns() - stage_started_ns
            if set(incoming_by_source) != set(neighbor_ports):
                raise ValueError("Neighbor target report coverage is incomplete.")

            stage_started_ns = perf_counter_ns()
            incoming_by_target: dict[str, list[TargetNodeReport]] = defaultdict(list)
            for source_id in sorted(incoming_by_source):
                for item in incoming_by_source[source_id]["reports"]:
                    report = _target_node_report(item)
                    incoming_by_target[report.target_id].append(report)
            reconstruction_ns = perf_counter_ns() - stage_started_ns

            stage_started_ns = perf_counter_ns()
            estimates = {
                target_id: fuse_target_reports(
                    (local_report, *incoming_by_target[target_id]),
                    grid_points=int(consensus_grid_points),
                )
                for target_id, local_report in sorted(local_by_target.items())
            }
            applied_ns = perf_counter_ns()
            result_queue.put({
                "node_id": node_id,
                "frame_index": frame_index,
                "target_count": int(target_count),
                "neighbor_batch_count": len(incoming_by_source),
                "payload_bytes_sent": len(payload) * len(neighbor_ports),
                "serialization_ns": int(serialization_ns),
                "socket_send_ns": int(socket_send_ns),
                "neighbor_receive_and_decode_ns": int(receive_and_decode_ns),
                "report_reconstruction_ns": int(reconstruction_ns),
                "consensus_ci_ns": int(applied_ns - stage_started_ns),
                "applied_ns": int(applied_ns),
                "estimate_count": len(estimates),
            })
    except Exception as exc:  # pragma: no cover - parent surfaces child errors
        error = {
            "error": f"{type(exc).__name__}: {exc}",
            "node_id": node_id,
        }
        ready_queue.put(error)
        prepared_queue.put(error)
        result_queue.put(error)
    finally:
        transport.close()


def _raise_worker_error(item) -> None:
    if item.get("error"):
        raise RuntimeError(
            f"Distributed node {item.get('node_id', '?')} failed: {item['error']}"
        )


def _start_workers_with_thread_limit(workers, thread_count: int) -> None:
    variable_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in variable_names}
    try:
        for name in variable_names:
            os.environ[name] = str(int(thread_count))
        for worker in workers:
            worker.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _ms(nanoseconds: int) -> float:
    return float(nanoseconds) / 1_000_000.0


def _mean(records, field: str) -> float:
    return float(np.mean([float(getattr(item, field)) for item in records]))
