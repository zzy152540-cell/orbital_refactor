import pytest

from examples.run_scene_control_server import build_parser


def test_scene_control_server_cli_parses_time_base():
    args = build_parser().parse_args([
        "--scene-id", "scene-a",
        "--run-id", "run-a",
        "--time-base-id", "TB-A",
        "--start-time", "2026-09-18T04:00:00.000Z",
        "--end-time", "2026-09-18T04:21:00.000Z",
    ])
    assert args.start_time == 1_789_704_000_000
    assert args.end_time == 1_789_705_260_000


def test_scene_control_server_cli_rejects_unknown_speed():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--scene-id", "scene-a", "--run-id", "run-a",
            "--time-base-id", "TB-A",
            "--start-time", "2026-09-18T04:00:00.000Z",
            "--end-time", "2026-09-18T04:21:00.000Z",
            "--speed", "3",
        ])
