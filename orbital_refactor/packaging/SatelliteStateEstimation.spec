from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPEC).resolve().parent.parent

# These modules are launched by name from the GUI.  PyInstaller cannot infer
# them from importlib calls, so keep this list aligned with
# experiments.run_v15_app_worker.PACKAGED_WORKER_MODULES.
hidden_imports = [
    "experiments.run_bilateral_acceptance_control",
    "experiments.run_bilateral_baseline_matrix",
    "experiments.run_bilateral_dropout_prescan",
    "experiments.run_bilateral_node_failure_prescan",
    "experiments.run_bilateral_visualization_recording",
    "experiments.run_interface_conformance_acceptance",
    "experiments.run_multitarget_distributed_node_sync_latency_acceptance",
    "experiments.run_tango_like_physical_frontend_demo",
    "experiments.run_ubuntu_environment_acceptance",
    "visualization.distance_response_window",
]

datas = []
datas += collect_data_files("matplotlib", include_py_files=False)

binaries = []
if sys.platform.startswith("linux"):
    # A conda pyexpat extension can require newer symbols than Ubuntu's system
    # libexpat.  PyInstaller normally treats libexpat as a system library and
    # leaves it out, which makes the frozen GUI fail when Matplotlib imports
    # its font manager.  Bundle the ABI-matched conda library explicitly.
    conda_expat = Path(sys.prefix) / "lib" / "libexpat.so.1"
    if not conda_expat.is_file():
        raise SystemExit(
            f"The active build environment has no {conda_expat}; "
            "install expat in the conda environment before packaging."
        )
    binaries.append((str(conda_expat), "."))

a = Analysis(
    [str(project_root / "experiments" / "run_v15_visualization_launcher.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "torchaudio"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SatelliteStateEstimation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SatelliteStateEstimation",
)
