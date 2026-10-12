"""Run scene control plus online target-free IOD/tracking and UDP output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import threading

from adapters.external_scene_input import adapt_external_scene_input
from brain_inspired.incremental_measurement_preprocessor import (
    IncrementalMultimodalCANNConfig,
    IncrementalMultimodalCANNPreprocessor,
)
from interfaces.online_multitarget_runtime import (
    ONLINE_MEASUREMENT_FRAME_PATH,
    OnlineMultiTargetRuntime,
    OnlineMultiTargetRuntimeEndpoint,
)
from interfaces.scene_control import (
    ALLOWED_SPEEDS,
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)
from interfaces.trajectory_udp import TrajectoryUdpPublisher
from tracking.association import OnlineMultiTargetTracker


def _time_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must be RFC 3339 UTC") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=8089)
    parser.add_argument("--udp-host", required=True)
    parser.add_argument("--udp-port", type=int, default=5010)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--time-base-id", required=True)
    parser.add_argument("--dataset-id", default="ONLINE-MULTITARGET")
    parser.add_argument("--start-time", required=True, type=_time_ms)
    parser.add_argument("--end-time", required=True, type=_time_ms)
    parser.add_argument("--sample-interval-ms", type=int, choices=(1000,), default=1000)
    parser.add_argument("--speed", type=int, choices=sorted(ALLOWED_SPEEDS), default=1)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument(
        "--enable-measurement-cann", action="store_true",
        help="Enable persistent incremental radar/infrared/optical CANN preprocessing.",
    )
    parser.add_argument(
        "--cann-profile", choices=("realtime", "research"), default="realtime",
        help="Realtime uses 90 ring neurons/2 ms; research uses 180/1 ms.",
    )
    return parser


def _activation_validator(config: SceneClockConfig):
    def validate(payload):
        scene = adapt_external_scene_input(payload)
        if scene.scene_id != config.scene_id:
            raise ValueError("activation sceneId does not match this runtime")
        if scene.start_time_ms != config.start_time_ms:
            raise ValueError("activation startTime does not match this runtime")
        if scene.end_time_ms != config.end_time_ms:
            raise ValueError("activation endTime does not match this runtime")
        if int(round(scene.step_seconds * 1_000.0)) != config.sample_interval_ms:
            raise ValueError("activation time step does not match this runtime")
        return {
            "activationMode": "ONLINE_TARGET_FREE_IOD_TRACKING",
            "initialConditionsAccepted": True,
            "acceptedSatelliteCount": len(scene.satellites),
            "observerStateSource": "ONLINE_MEASUREMENT_FRAMES",
        }

    return validate


def main() -> None:
    args = build_parser().parse_args()
    config = SceneClockConfig(
        scene_id=args.scene_id,
        run_id=args.run_id,
        time_base_id=args.time_base_id,
        start_time_ms=args.start_time,
        end_time_ms=args.end_time,
        sample_interval_ms=args.sample_interval_ms,
        speed=args.speed,
    )
    controller = SceneClockController(
        config,
        require_activation=True,
        activation_validator=_activation_validator(config),
    )
    def fresh_runtime():
        tracker = None
        if args.enable_measurement_cann:
            cann_config = IncrementalMultimodalCANNConfig(
                profile=args.cann_profile,
            )
            tracker = OnlineMultiTargetTracker(
                scene_id=args.scene_id,
                measurement_preprocessor_factory=lambda _observer, _target: (
                    IncrementalMultimodalCANNPreprocessor(cann_config)
                ),
            )
        return OnlineMultiTargetRuntime(
            scene_id=args.scene_id,
            time_base_id=args.time_base_id,
            dataset_id=args.dataset_id,
            start_time_ms=args.start_time,
            sample_interval_seconds=args.sample_interval_ms / 1000.0,
            tracker=tracker,
        )

    with TrajectoryUdpPublisher(args.udp_host, args.udp_port) as publisher:
        endpoint = OnlineMultiTargetRuntimeEndpoint(
            fresh_runtime(), controller, publisher,
        )
        server = create_scene_control_server(
            controller,
            args.control_host,
            args.control_port,
            auth_token=args.auth_token,
            post_routes=endpoint.post_routes,
        )
        print(
            f"online service listening on http://{args.control_host}:"
            f"{server.server_address[1]}; measurement frames: "
            f"POST {ONLINE_MEASUREMENT_FRAME_PATH}; UDP -> "
            f"{args.udp_host}:{args.udp_port}",
            flush=True,
        )
        print(
            f"measurement CANN: "
            f"{'enabled (' + args.cann_profile + ')' if args.enable_measurement_cann else 'disabled'}",
            flush=True,
        )
        stop_monitor = threading.Event()

        def monitor_runs():
            while not stop_monitor.wait(0.05):
                state = controller.snapshot(applied=None)["clockState"]
                if state in {"STOPPED", "FINISHED"}:
                    endpoint.reset(fresh_runtime())
                    controller.rearm_for_activation()

        monitor = threading.Thread(target=monitor_runs, daemon=True)
        monitor.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop_monitor.set()
            server.server_close()
            monitor.join(timeout=2.0)


if __name__ == "__main__":
    main()
