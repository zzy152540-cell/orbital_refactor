"""Role-safe scenario contracts for friendly observers and hostile targets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from cooperative.topology import (
    NetworkTopology,
    chain_topology,
    fully_connected_topology,
    ring_topology,
    star_topology,
    two_hop_chain_topology,
)
from orbital_core.constants import R_EARTH
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    WalkerDeltaScenario,
    generate_walker_delta_scenario,
)


TEMPORARY_FRIENDLY_STATE_ASSUMPTION = (
    "EXACT_KNOWN_TEMPORARY_PENDING_REVIEW"
)


def _identifier(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must be a non-empty identifier.")
    return result


def _unique_identifiers(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(_identifier(value, field_name) for value in values)
    if not result:
        raise ValueError(f"{field_name} must contain at least one identifier.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique identifiers.")
    return result


@dataclass(frozen=True)
class HostileTargetRegistry:
    """Simulation/evaluation roster that never provides target states to filters."""

    target_ids: tuple[str, ...]
    metadata_by_target: Mapping[str, Mapping[str, object]]

    def __init__(
        self,
        target_ids: Iterable[str],
        metadata_by_target: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        identifiers = _unique_identifiers(target_ids, "target_ids")
        metadata = {
            _identifier(target_id, "metadata target_id"): dict(values)
            for target_id, values in (metadata_by_target or {}).items()
        }
        unknown = set(metadata) - set(identifiers)
        if unknown:
            raise ValueError(
                "Hostile target metadata references unknown targets: "
                f"{sorted(unknown)}"
            )
        object.__setattr__(self, "target_ids", identifiers)
        object.__setattr__(
            self,
            "metadata_by_target",
            MappingProxyType({
                target_id: MappingProxyType(metadata.get(target_id, {}))
                for target_id in identifiers
            }),
        )

    def __contains__(self, target_id: object) -> bool:
        return str(target_id) in self.target_ids

    def require(self, target_id: str) -> str:
        identifier = _identifier(target_id, "target_id")
        if identifier not in self.target_ids:
            raise KeyError(f"Unknown hostile target_id: {identifier}")
        return identifier


@dataclass(frozen=True)
class BilateralMultiTargetScenario:
    """Immutable role manifest and communication boundary for a bilateral scene."""

    scene_id: str
    friendly_observer_ids: tuple[str, ...]
    hostile_targets: HostileTargetRegistry
    friendly_topology: NetworkTopology
    friendly_state_assumption: str = TEMPORARY_FRIENDLY_STATE_ASSUMPTION

    def __post_init__(self) -> None:
        scene_id = _identifier(self.scene_id, "scene_id")
        friendly = _unique_identifiers(
            self.friendly_observer_ids, "friendly_observer_ids"
        )
        hostile = self.hostile_targets.target_ids
        overlap = set(friendly) & set(hostile)
        if overlap:
            raise ValueError(
                "Friendly observer and hostile target identifiers must be disjoint: "
                f"{sorted(overlap)}"
            )
        topology_nodes = set(self.friendly_topology.node_ids)
        if topology_nodes != set(friendly):
            missing = sorted(set(friendly) - topology_nodes)
            forbidden = sorted(topology_nodes - set(friendly))
            raise ValueError(
                "Friendly topology must contain exactly the friendly observers; "
                f"missing={missing}, forbidden={forbidden}"
            )
        assumption = _identifier(
            self.friendly_state_assumption, "friendly_state_assumption"
        )
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "friendly_observer_ids", friendly)
        object.__setattr__(self, "friendly_state_assumption", assumption)

    @property
    def hostile_target_ids(self) -> tuple[str, ...]:
        return self.hostile_targets.target_ids

    @property
    def total_satellite_count(self) -> int:
        return len(self.friendly_observer_ids) + len(self.hostile_target_ids)

    def require_observation_pair(self, observer_id: str, target_id: str) -> None:
        observer = _identifier(observer_id, "observer_id")
        target = _identifier(target_id, "target_id")
        if observer not in self.friendly_observer_ids:
            raise ValueError(
                f"Observation source {observer!r} is not a friendly observer."
            )
        if target not in self.hostile_targets:
            raise ValueError(
                f"Observation target {target!r} is not a registered hostile target."
            )

    def require_communication_edge(self, source_id: str, receiver_id: str) -> None:
        source = _identifier(source_id, "source_id")
        receiver = _identifier(receiver_id, "receiver_id")
        friendly = set(self.friendly_observer_ids)
        if source not in friendly or receiver not in friendly:
            raise ValueError(
                "Communication is restricted to friendly observer nodes."
            )
        if not self.friendly_topology.includes_edge(source, receiver):
            raise ValueError(
                f"Communication edge {source!r}->{receiver!r} is absent from "
                "the fixed friendly topology."
            )


@dataclass(frozen=True)
class BilateralWalkerTruth:
    """Non-maneuvering Walker truth histories bound to a bilateral manifest."""

    manifest: BilateralMultiTargetScenario
    timestamps: np.ndarray
    friendly_walker: WalkerDeltaScenario
    hostile_walker: WalkerDeltaScenario

    def __post_init__(self) -> None:
        times = np.array(self.timestamps, dtype=float, copy=True).reshape(-1)
        if times.size == 0 or np.any(~np.isfinite(times)):
            raise ValueError("timestamps must be nonempty and finite.")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("timestamps must be strictly increasing.")
        if tuple(self.friendly_walker.node_ids) != self.manifest.friendly_observer_ids:
            raise ValueError("Friendly Walker IDs must match the bilateral manifest.")
        if tuple(self.hostile_walker.node_ids) != self.manifest.hostile_target_ids:
            raise ValueError("Hostile Walker IDs must match the bilateral manifest.")
        for walker in (self.friendly_walker, self.hostile_walker):
            if not np.array_equal(walker.fleet.timestamps, times):
                raise ValueError("Both Walker populations must share the truth timeline.")
        times.setflags(write=False)
        object.__setattr__(self, "timestamps", times)

    @property
    def friendly_truth_state_history_by_node(self) -> dict[str, np.ndarray]:
        return self.friendly_walker.truth_state_history_by_node

    @property
    def hostile_truth_state_history_by_target(self) -> dict[str, np.ndarray]:
        return self.hostile_walker.truth_state_history_by_node


def build_bilateral_acceptance_scenario(
    *,
    scene_id: str = "bilateral-15x15",
    friendly_count: int = 15,
    hostile_count: int = 15,
    topology_policy: str = "ring",
) -> BilateralMultiTargetScenario:
    """Build the deterministic role manifest for the V2 acceptance scenario."""

    friendly_count = int(friendly_count)
    hostile_count = int(hostile_count)
    if friendly_count <= 0 or hostile_count <= 0:
        raise ValueError("Friendly and hostile satellite counts must be positive.")
    if friendly_count + hostile_count < 25:
        raise ValueError("The V2 acceptance scenario requires at least 25 satellites.")
    friendly_ids = tuple(
        f"friendly_{index:02d}" for index in range(1, friendly_count + 1)
    )
    hostile_ids = tuple(
        f"hostile_{index:02d}" for index in range(1, hostile_count + 1)
    )
    topology = _build_topology(friendly_ids, topology_policy)
    return BilateralMultiTargetScenario(
        scene_id=scene_id,
        friendly_observer_ids=friendly_ids,
        hostile_targets=HostileTargetRegistry(hostile_ids),
        friendly_topology=topology,
    )


def generate_bilateral_walker_truth(
    *,
    duration_seconds: float = 120.0,
    sample_interval_seconds: float = 1.0,
    friendly_config: WalkerDeltaConfig | None = None,
    hostile_config: WalkerDeltaConfig | None = None,
    topology_policy: str = "ring",
    scene_id: str = "bilateral-walker-15x15",
    minimum_total_satellites: int = 25,
) -> BilateralWalkerTruth:
    """Generate the initial 15-friendly/15-hostile non-maneuvering truth case."""

    duration = float(duration_seconds)
    dt = float(sample_interval_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_seconds must be positive and finite.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample_interval_seconds must be positive and finite.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    friendly_config = friendly_config or WalkerDeltaConfig(
        total_satellites=15,
        plane_count=3,
        phasing=1,
        semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.0,
        inclination=np.deg2rad(53.0),
        node_prefix="friendly",
    )
    hostile_config = hostile_config or WalkerDeltaConfig(
        total_satellites=15,
        plane_count=3,
        phasing=1,
        semi_major_axis=R_EARTH + 705e3,
        eccentricity=0.0,
        inclination=np.deg2rad(53.0),
        raan_origin=np.deg2rad(0.5),
        # Half a slot relative to the friendly Walker population.  This keeps
        # the two fleets distinct while giving every hostile target multiple
        # friendly observers under the Stage-B visibility limits.
        base_true_anomaly=np.deg2rad(36.0),
        node_prefix="hostile",
    )
    friendly = generate_walker_delta_scenario(
        timestamps=timestamps, config=friendly_config
    )
    hostile = generate_walker_delta_scenario(
        timestamps=timestamps, config=hostile_config
    )
    if set(friendly.node_ids) & set(hostile.node_ids):
        raise ValueError("Friendly and hostile Walker IDs must be disjoint.")
    minimum_total = int(minimum_total_satellites)
    if minimum_total < 2:
        raise ValueError("minimum_total_satellites must be at least two.")
    if len(friendly.node_ids) + len(hostile.node_ids) < minimum_total:
        raise ValueError(
            "The bilateral Walker truth case requires at least "
            f"{minimum_total} satellites."
        )
    topology = _build_topology(tuple(friendly.node_ids), topology_policy)
    hostile_metadata = {
        target_id: {
            "truth_only": True,
            "plane_index": element.plane_index,
            "slot_index": element.slot_index,
        }
        for target_id, element in hostile.elements_by_node.items()
    }
    manifest = BilateralMultiTargetScenario(
        scene_id=scene_id,
        friendly_observer_ids=tuple(friendly.node_ids),
        hostile_targets=HostileTargetRegistry(
            hostile.node_ids, metadata_by_target=hostile_metadata
        ),
        friendly_topology=topology,
    )
    return BilateralWalkerTruth(
        manifest=manifest,
        timestamps=timestamps,
        friendly_walker=friendly,
        hostile_walker=hostile,
    )


def _build_topology(
    node_ids: tuple[str, ...], topology_policy: str
) -> NetworkTopology:
    policy = str(topology_policy).strip().lower()
    builders = {
        "chain": chain_topology,
        "two_hop_chain": two_hop_chain_topology,
        "ring": ring_topology,
        "star": star_topology,
        "fully_connected": fully_connected_topology,
    }
    if policy not in builders:
        raise ValueError(
            f"Unsupported friendly topology policy {topology_policy!r}; "
            f"expected one of {sorted(builders)}."
        )
    if policy == "ring" and len(node_ids) < 3:
        return chain_topology(node_ids)
    if policy in {"star", "two_hop_chain"} and len(node_ids) < 2:
        return chain_topology(node_ids)
    return builders[policy](node_ids)
