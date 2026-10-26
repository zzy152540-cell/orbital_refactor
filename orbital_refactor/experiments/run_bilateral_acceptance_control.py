"""Generate one reusable matched control for bilateral robustness tests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.bilateral_multitarget_frontend import (
    BilateralFrontendConfig,
    build_bilateral_raw_frontend,
    electro_optical_radial_failover_sensor_config,
    run_bilateral_cann_acceptance_frontend,
)
from experiments.run_bilateral_visualization_recording import (
    write_bilateral_visualization_recording,
)


def build_control_contract(
    *, duration: float, event_start: float, event_end: float, seed: int,
    family: str, distributed_consensus: bool = True,
) -> dict:
    if family not in {"nominal", "radar-backup", "electro-optical-failover"}:
        raise ValueError(f"Unsupported acceptance control family: {family}")
    radial = family == "radar-backup"
    eo_failover = family == "electro-optical-failover"
    return {
        "schemaVersion": "bilateral-acceptance-control-contract.v1",
        "family": family,
        "friendlyObserverCount": 15,
        "hostileTargetCount": 15,
        "durationSeconds": float(duration),
        "sampleIntervalSeconds": 1.0,
        "eventStartSeconds": float(event_start),
        "eventEndSeconds": float(event_end),
        "seed": int(seed),
        "localUnifiedMultimodalMeasurementUpdate": True,
        "distributedConsensus": bool(distributed_consensus),
        "consensusExchangeHops": 1,
        "processNoiseAcceleration": 1e-4,
        "bearingWindowEpochs": 1,
        "multiObserverBearingEnabled": True,
        "radialBackup": {
            "enabled": radial,
            "rangeSigmaM": 1.0 if radial else None,
            "rangeRateSigmaMps": 0.0125 if radial else None,
            "observerIndices": "ALL",
            "independentChannel": False,
            "physicalValidation": False,
        },
        "electroOpticalRadialFailover": {
            "enabled": eo_failover,
            "policy": "radar_primary_failover" if eo_failover else None,
            "rangeSigmaM": 2.5 if eo_failover else None,
            "rangeRateSigmaMps": 0.010 if eo_failover else None,
            "physicalValidation": False,
        },
    }


def control_contract_sha256(contract: dict) -> str:
    encoded = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_matching_control(path: str | Path, expected_contract: dict) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = control_contract_sha256(expected_contract)
    actual = payload.get("controlContractSha256")
    if actual != expected:
        raise ValueError(
            "Reusable control configuration does not match the experiment: "
            f"expected {expected}, got {actual}."
        )
    control = payload.get("control")
    if not isinstance(control, dict):
        raise ValueError("Reusable control payload is missing its control arms.")
    return control


def _arm(value):
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in {
            "phase_metrics", "consensus_phase_metrics",
            "representative_node_metrics", "independent_node_metrics",
        }
    }
    payload["phase_metrics"] = {
        name: asdict(metrics) for name, metrics in value.phase_metrics.items()
    }
    payload["consensus_phase_metrics"] = {
        name: asdict(metrics)
        for name, metrics in value.consensus_phase_metrics.items()
    }
    payload["representative_node_metrics"] = _node_view(
        value.representative_node_metrics
    )
    payload["independent_node_metrics"] = _node_view(
        value.independent_node_metrics
    )
    return _jsonable(payload)


def _node_view(value):
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != "phase_metrics"
    }
    payload["phase_metrics"] = {
        name: asdict(metrics) for name, metrics in value.phase_metrics.items()
    }
    return payload


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--event-start", type=float, required=True)
    parser.add_argument("--event-end", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--family",
        choices=("nominal", "radar-backup", "electro-optical-failover"),
        required=True,
    )
    parser.add_argument(
        "--distributed-consensus", action="store_true",
        help="Run the R08 cooperative control arm; omit for the R06 local arm.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-output", type=Path)
    args = parser.parse_args()

    contract = build_control_contract(
        duration=args.duration,
        event_start=args.event_start,
        event_end=args.event_end,
        seed=args.seed,
        family=args.family,
        distributed_consensus=args.distributed_consensus,
    )
    radial = args.family == "radar-backup"
    eo_failover = args.family == "electro-optical-failover"
    config = BilateralFrontendConfig(
            duration_seconds=args.duration,
            sample_interval_seconds=1.0,
            random_seed=args.seed,
            sensor_config=(
                electro_optical_radial_failover_sensor_config()
                if eo_failover else None
            ),
            retain_raw_frames=args.recording_output is not None,
            backup_radial_range_sigma_m=1.0 if radial else None,
            backup_radial_range_rate_sigma_mps=0.0125 if radial else None,
            use_local_unified_multimodal_measurement_update=True,
            dropout_modality="OPTICAL",
            dropout_start_seconds=args.event_start,
            dropout_end_seconds=args.event_end,
            dropout_target_indices=(),
        )
    frontend = build_bilateral_raw_frontend(config)
    captured = []
    result = run_bilateral_cann_acceptance_frontend(
        frontend,
        enable_consensus=args.distributed_consensus,
        epoch_callback=(captured.append if args.recording_output else None),
    )
    payload = {
        "schemaVersion": "bilateral-acceptance-control.v1",
        "controlContract": contract,
        "controlContractSha256": control_contract_sha256(contract),
        "control": {
            "incrementalCann": _arm(result),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.recording_output is not None:
        write_bilateral_visualization_recording(
            args.recording_output,
            frontend=frontend,
            captured=captured,
            include_cann=True,
            cann_node_limit=0,
            enable_consensus=args.distributed_consensus,
            show_communication_topology=args.distributed_consensus,
            radial_backup_profile=("legacy" if radial else "none"),
            electro_optical_radial_failover=eo_failover,
            seed=args.seed,
            source_result_path=args.output,
        )
    print(
        f"wrote {args.output} family={args.family} "
        f"fingerprint={payload['controlContractSha256']}"
    )


if __name__ == "__main__":
    main()
