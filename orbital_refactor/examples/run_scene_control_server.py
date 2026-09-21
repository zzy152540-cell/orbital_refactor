"""Run the standalone SCENE-CTRL-2.0 HTTP command receiver."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from interfaces.scene_control import (
    ALLOWED_SPEEDS,
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)


def _time_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must be RFC 3339 UTC") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--time-base-id", required=True)
    parser.add_argument("--start-time", required=True, type=_time_ms)
    parser.add_argument("--end-time", required=True, type=_time_ms)
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--speed", type=int, choices=sorted(ALLOWED_SPEEDS), default=1)
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional X-Auth-Token; omit only for trusted local testing.",
    )
    return parser


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
    controller = SceneClockController(config)
    server = create_scene_control_server(
        controller, args.host, args.port, auth_token=args.auth_token,
    )
    print(
        f"SCENE-CTRL-2.0 listening on http://{args.host}:{server.server_address[1]} "
        f"for sceneId={args.scene_id} runId={args.run_id}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
