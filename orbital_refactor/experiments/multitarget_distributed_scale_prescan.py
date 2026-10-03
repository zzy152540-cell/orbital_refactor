"""Bounded scalability prescan for target-centric distributed CI."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cooperative.topology import (
    chain_topology,
    fully_connected_topology,
)
from tracking import DistributedTargetConsensus, TargetNodeReport


def run_multitarget_distributed_scale_prescan(
    *,
    node_counts: tuple[int, ...] = (4, 6, 10),
    target_count: int = 2,
    epochs: int = 5,
    topologies: tuple[str, ...] = ("chain", "fully_connected"),
) -> dict[str, object]:
    if not node_counts or any(value < 2 for value in node_counts):
        raise ValueError("node_counts must contain values of at least two.")
    if target_count < 1 or epochs < 1:
        raise ValueError("target_count and epochs must be positive.")
    unsupported = set(topologies) - {"chain", "fully_connected"}
    if unsupported:
        raise ValueError(f"Unsupported topologies: {sorted(unsupported)}")

    records = []
    for node_count in node_counts:
        node_ids = tuple(f"observer_{index:02d}" for index in range(node_count))
        for topology_name in topologies:
            topology = (
                chain_topology(node_ids)
                if topology_name == "chain"
                else fully_connected_topology(node_ids)
            )
            consensus = DistributedTargetConsensus(
                scene_id=f"scale-{node_count}-{topology_name}",
                topology=topology,
                grid_points=11,
            )
            inherited = {
                (node_id, target_index): ()
                for node_id in node_ids
                for target_index in range(target_count)
            }
            step_seconds = []
            received_count = 0
            final = None
            for epoch in range(epochs):
                reports = []
                for node_index, node_id in enumerate(node_ids):
                    for target_index in range(target_count):
                        information_ids = tuple(dict.fromkeys((
                            *inherited[(node_id, target_index)],
                            f"e{epoch}:{node_id}:t{target_index}",
                        )))
                        reports.append(TargetNodeReport(
                            observer_id=node_id,
                            target_id=f"target_{target_index:02d}",
                            track_id=f"track_{target_index:02d}",
                            timestamp=float(epoch),
                            state_eci=np.array([
                                7.0e6 + target_index * 100_000.0 + node_index,
                                7500.0 * epoch,
                                10.0 * target_index,
                                0.0,
                                7500.0,
                                0.0,
                            ]),
                            covariance_eci=np.diag([
                                25.0 + node_index,
                                25.0 + node_index,
                                25.0 + node_index,
                                0.1,
                                0.1,
                                0.1,
                            ]),
                            quality_score=0.9,
                            used_measurement_ids=information_ids,
                        ))
                started = perf_counter()
                final = consensus.step(
                    timestamp=float(epoch),
                    local_reports=reports,
                    active_source_ids=node_ids,
                )
                step_seconds.append(perf_counter() - started)
                received_count += sum(
                    len(values)
                    for values in final.received_message_ids_by_node.values()
                )
                inherited = {
                    (node_id, target_index): final.estimates_by_node[node_id][
                        f"target_{target_index:02d}"
                    ].information_ids
                    for node_id in node_ids
                    for target_index in range(target_count)
                }

            assert final is not None
            maximum_contributors = max(
                len(estimate.contributing_observer_ids)
                for estimates in final.estimates_by_node.values()
                for estimate in estimates.values()
            )
            endpoint_information_count = len(
                final.estimates_by_node[node_ids[0]]["target_00"].information_ids
            )
            records.append({
                "nodeCount": int(node_count),
                "targetCount": int(target_count),
                "epochs": int(epochs),
                "topology": topology_name,
                "meanStepMilliseconds": 1000.0 * float(np.mean(step_seconds)),
                "maximumStepMilliseconds": 1000.0 * float(np.max(step_seconds)),
                "receivedMessageCount": int(received_count),
                "maximumContributorsPerTarget": int(maximum_contributors),
                "endpointInformationCount": int(endpoint_information_count),
                "pendingMessageCount": int(final.pending_message_count),
            })
    return {
        "kind": "MULTITARGET_DISTRIBUTED_SCALE_PRESCAN",
        "records": records,
    }
