"""Replay one-frame-at-a-time multi-target JSON input without network transport."""

from __future__ import annotations

import argparse
import json

from tracking.online_json import run_online_json_replay_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_online_json_replay_file(args.input, args.output)
    print(json.dumps({
        "ok": True,
        "sceneId": result["sceneId"],
        "frameCount": result["frameCount"],
        "output": args.output,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
