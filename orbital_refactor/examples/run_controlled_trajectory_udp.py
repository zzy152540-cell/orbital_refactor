"""Serve SCENE-CTRL commands and stream a prepared trajectory over UDP."""

from __future__ import annotations

import argparse
import threading
import time

from interfaces.controlled_trajectory import ControlledTrajectoryStreamer
from interfaces.scene_control import (
    ALLOWED_SPEEDS,
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)
from interfaces.trajectory_udp import TrajectoryUdpPublisher, load_trajectory_time_series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="TRAJECTORY-2.0 JSON file")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8080)
    parser.add_argument("--udp-host", required=True)
    parser.add_argument("--udp-port", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--speed", type=int, choices=sorted(ALLOWED_SPEEDS), default=1)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--poll-ms", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_ms <= 0:
        raise SystemExit("--poll-ms must be positive")
    trajectory = load_trajectory_time_series(args.trajectory)
    config = SceneClockConfig(
        scene_id=str(trajectory["sceneId"]),
        run_id=args.run_id,
        time_base_id=str(trajectory["timeBaseId"]),
        start_time_ms=int(trajectory["startTimeMs"]),
        end_time_ms=int(trajectory["endTimeMs"]),
        sample_interval_ms=int(trajectory["sampleIntervalMs"]),
        speed=args.speed,
    )
    controller = SceneClockController(config)
    server = create_scene_control_server(
        controller,
        args.control_host,
        args.control_port,
        auth_token=args.auth_token,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(
        f"Control: http://{args.control_host}:{server.server_address[1]} | "
        f"UDP: {args.udp_host}:{args.udp_port} | state=READY",
        flush=True,
    )

    try:
        with TrajectoryUdpPublisher(args.udp_host, args.udp_port) as publisher:
            streamer = ControlledTrajectoryStreamer(controller, trajectory, publisher)
            while True:
                result = streamer.step()
                if result.sent_frames:
                    print(
                        f"sent frames {result.first_frame_index}..{result.last_frame_index} "
                        f"at timeMs={result.time_ms}",
                        flush=True,
                    )
                if result.clock_state == "STOPPED":
                    break
                if result.clock_state == "FINISHED" and result.complete:
                    break
                time.sleep(args.poll_ms / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    main()
