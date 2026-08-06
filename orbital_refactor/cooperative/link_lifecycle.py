from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class LinkLifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RESYNC_REQUIRED = "RESYNC_REQUIRED"


@dataclass(frozen=True)
class LinkLifecycle:
    """Protocol state for one directed covariance-transport link."""

    receiver_id: str
    source_id: str
    lineage_id: str
    topology_version: int = 0
    state: LinkLifecycleState = LinkLifecycleState.ACTIVE
    resynchronization_reason: str | None = None
    suspension_count: int = 0
    resynchronization_count: int = 0

    def __post_init__(self) -> None:
        if not self.receiver_id or not self.source_id:
            raise ValueError("Link endpoints must be nonempty.")
        if self.receiver_id == self.source_id:
            raise ValueError("A transport link cannot be a self-link.")
        if not self.lineage_id:
            raise ValueError("lineage_id must be nonempty.")
        if self.topology_version < 0:
            raise ValueError("topology_version cannot be negative.")
        if self.state == LinkLifecycleState.RESYNC_REQUIRED:
            if not self.resynchronization_reason:
                raise ValueError(
                    "RESYNC_REQUIRED requires a resynchronization reason."
                )
        elif self.resynchronization_reason is not None:
            raise ValueError(
                "Only RESYNC_REQUIRED may carry a resynchronization reason."
            )

    def suspend(self, *, topology_version: int) -> "LinkLifecycle":
        self._require_newer_version(topology_version)
        if self.state == LinkLifecycleState.RESYNC_REQUIRED:
            return replace(self, topology_version=int(topology_version))
        return replace(
            self, topology_version=int(topology_version),
            state=LinkLifecycleState.SUSPENDED,
            suspension_count=self.suspension_count + 1,
        )

    def resume(
        self, *, topology_version: int, history_available: bool,
        unavailable_reason: str = "history_unavailable",
    ) -> "LinkLifecycle":
        self._require_newer_version(topology_version)
        if self.state != LinkLifecycleState.SUSPENDED:
            raise ValueError("Only a suspended link can be resumed.")
        if history_available:
            return replace(
                self, topology_version=int(topology_version),
                state=LinkLifecycleState.ACTIVE,
            )
        if not unavailable_reason:
            raise ValueError("Unavailable history requires a reason.")
        return replace(
            self, topology_version=int(topology_version),
            state=LinkLifecycleState.RESYNC_REQUIRED,
            resynchronization_reason=str(unavailable_reason),
        )

    def require_resynchronization(
        self, *, reason: str, topology_version: int | None = None,
    ) -> "LinkLifecycle":
        if not reason:
            raise ValueError("Resynchronization requires a reason.")
        version = (
            self.topology_version
            if topology_version is None else int(topology_version)
        )
        if version < self.topology_version:
            raise ValueError("Topology versions must be monotonic.")
        return replace(
            self, topology_version=version,
            state=LinkLifecycleState.RESYNC_REQUIRED,
            resynchronization_reason=str(reason),
        )

    def establish_resynchronized_lineage(
        self, *, lineage_id: str, topology_version: int | None = None,
    ) -> "LinkLifecycle":
        if self.state != LinkLifecycleState.RESYNC_REQUIRED:
            raise ValueError(
                "A new lineage can only establish a link awaiting resync."
            )
        if not lineage_id or lineage_id == self.lineage_id:
            raise ValueError("Resynchronization requires a distinct lineage_id.")
        version = (
            self.topology_version
            if topology_version is None else int(topology_version)
        )
        if version < self.topology_version:
            raise ValueError("Topology versions must be monotonic.")
        return replace(
            self, lineage_id=str(lineage_id), topology_version=version,
            state=LinkLifecycleState.ACTIVE,
            resynchronization_reason=None,
            resynchronization_count=self.resynchronization_count + 1,
        )

    def accepts(self, *, lineage_id: str, topology_version: int) -> bool:
        return bool(
            self.state == LinkLifecycleState.ACTIVE
            and lineage_id == self.lineage_id
            and int(topology_version) == self.topology_version
        )

    def observe_topology_version(
        self, topology_version: int,
    ) -> "LinkLifecycle":
        if int(topology_version) < self.topology_version:
            raise ValueError("Topology versions must be monotonic.")
        return replace(self, topology_version=int(topology_version))

    def _require_newer_version(self, topology_version: int) -> None:
        if int(topology_version) <= self.topology_version:
            raise ValueError("A topology transition requires a newer version.")
