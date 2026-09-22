"""Receive and validate complete TRAJECTORY-2.0 UDP frames."""

from __future__ import annotations

import argparse
import socket
import time

from interfaces.trajectory_udp import decode_trajectory_datagram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--expected-satellites", type=int, default=31)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be in 1..65535")
    if args.timeout <= 0.0 or args.print_every <= 0:
        raise SystemExit("--timeout and --print-every must be positive")

    received: dict[int, dict] = {}
    duplicate_count = 0
    byte_count = 0
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind((args.host, args.port))
        receiver.settimeout(args.timeout)
        print(
            f"Listening for trajectory UDP on {args.host}:{args.port} "
            f"(timeout={args.timeout:g}s)",
            flush=True,
        )
        try:
            while args.expected_frames is None or len(received) < args.expected_frames:
                data, source = receiver.recvfrom(65_535)
                message = decode_trajectory_datagram(data)
                index = int(message["frameIndex"])
                satellites = message["satelliteList"]
                if len(satellites) != args.expected_satellites:
                    raise ValueError(
                        f"Frame {index} has {len(satellites)} satellites; "
                        f"expected {args.expected_satellites}."
                    )
                if index in received:
                    duplicate_count += 1
                else:
                    received[index] = message
                byte_count += len(data)
                if len(received) == 1 or len(received) % args.print_every == 0:
                    print(
                        f"received={len(received)} frameIndex={index} "
                        f"timeMs={message['timeMs']} bytes={len(data)} source={source}",
                        flush=True,
                    )
        except socket.timeout:
            print("Receiver timeout reached; reporting partial result.", flush=True)
        except KeyboardInterrupt:
            print("Receiver interrupted; reporting partial result.", flush=True)

    indices = sorted(received)
    missing = []
    if indices:
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
    elapsed = max(time.monotonic() - started, 1e-12)
    print(
        "summary: "
        f"uniqueFrames={len(received)} duplicates={duplicate_count} "
        f"missingWithinRange={len(missing)} bytes={byte_count} "
        f"elapsedSeconds={elapsed:.3f} framesPerSecond={len(received) / elapsed:.3f}",
        flush=True,
    )
    if missing:
        print(f"missingFrameIndices={missing[:50]}", flush=True)
        raise SystemExit(2)
    if args.expected_frames is not None and len(received) != args.expected_frames:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
