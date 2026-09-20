import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_shirt_example_has_no_machine_specific_default_paths():
    source = (PROJECT_ROOT / "examples" / "run_shirt_interface.py").read_text(
        encoding="utf-8"
    )
    for marker in ("C:\\", "D:\\", "E:\\"):
        assert marker not in source


def test_shirt_example_exposes_path_arguments_without_loading_data():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.run_shirt_interface", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--metadata" in completed.stdout
    assert "--orbit" in completed.stdout
    assert "--predictions" in completed.stdout
