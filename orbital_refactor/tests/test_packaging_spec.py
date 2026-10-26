from pathlib import Path


def test_linux_bundle_pins_conda_expat_for_matplotlib_gui_imports():
    spec = (
        Path(__file__).resolve().parents[1]
        / "packaging" / "SatelliteStateEstimation.spec"
    ).read_text(encoding="utf-8")

    assert 'Path(sys.prefix) / "lib" / "libexpat.so.1"' in spec
    assert 'binaries.append((str(conda_expat), "."))' in spec
