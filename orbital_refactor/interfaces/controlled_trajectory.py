"""Bridge the authoritative scene clock to ordered UDP trajectory output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .scene_control import SceneClockController
from .trajectory_udp import validate_trajectory_time_series


class FramePublisher(Protocol):
    def send_frame(self, trajectory: Mapping[str, Any], frame_index: int) -> int:
        ...


@dataclass(frozen=True)
class StreamStepResult:
    clock_state: str
    time_ms: int
    first_frame_index: int | None
    last_frame_index: int | None
    sent_frames: int
    sent_bytes: int
    complete: bool


class ControlledTrajectoryStreamer:
    """Send each trajectory frame once when the scene clock reaches its time."""

    def __init__(
        self,
        controller: SceneClockController,
        trajectory: Mapping[str, Any],
        publisher: FramePublisher,
    ) -> None:
        validate_trajectory_time_series(trajectory)
        config = controller.config
        if trajectory["sceneId"] != config.scene_id:
            raise ValueError("Trajectory sceneId does not match the controller.")
        if trajectory["timeBaseId"] != config.time_base_id:
            raise ValueError("Trajectory timeBaseId does not match the controller.")
        for name, expected in (
            ("startTimeMs", config.start_time_ms),
            ("endTimeMs", config.end_time_ms),
            ("sampleIntervalMs", config.sample_interval_ms),
            ("frameCount", config.frame_count),
        ):
            if int(trajectory[name]) != int(expected):
                raise ValueError(f"Trajectory {name} does not match the controller.")
        self.controller = controller
        self.trajectory = trajectory
        self.publisher = publisher
        self.next_frame_index = 0

    @property
    def complete(self) -> bool:
        return self.next_frame_index >= int(self.trajectory["frameCount"])

    def step(self) -> StreamStepResult:
        snapshot = self.controller.snapshot(applied=None)
        state = snapshot["clockState"]
        first: int | None = None
        last: int | None = None
        sent_frames = 0
        sent_bytes = 0

        # READY has not received START, and STOPPED explicitly cancels output.
        if state not in {"READY", "STOPPED"}:
            frames = self.trajectory["frames"]
            while self.next_frame_index < len(frames):
                frame = frames[self.next_frame_index]
                if int(frame["timeMs"]) > int(snapshot["timeMs"]):
                    break
                if first is None:
                    first = self.next_frame_index
                sent_bytes += self.publisher.send_frame(
                    self.trajectory, self.next_frame_index,
                )
                last = self.next_frame_index
                sent_frames += 1
                self.next_frame_index += 1

        return StreamStepResult(
            clock_state=state,
            time_ms=int(snapshot["timeMs"]),
            first_frame_index=first,
            last_frame_index=last,
            sent_frames=sent_frames,
            sent_bytes=sent_bytes,
            complete=self.complete,
        )
