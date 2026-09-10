from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.run_v15_robust_snapshot_collection import main as collect_shard
from experiments.topology_snapshot_counterfactual import (
    load_topology_snapshot_tensor_dataset,
    merge_topology_snapshot_tensor_datasets,
    save_topology_snapshot_tensor_dataset,
)


@dataclass(frozen=True)
class ExpansionShard:
    name: str
    condition_seeds: tuple[int, ...]
    dropout_node_count: int
    initial_topology_types: tuple[str, ...]


@dataclass(frozen=True)
class ExpansionConditionSplit:
    training: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def physical_expansion_condition_split() -> ExpansionConditionSplit:
    """Return the frozen topology/pressure-stratified 27-condition split."""
    return ExpansionConditionSplit(
        training=(
            124, 125, 133, 134, 130,
            126, 127, 135, 136, 131,
            128, 129, 137, 138, 132,
        ),
        validation=(142, 139, 144, 140, 146, 141),
        test=(143, 148, 145, 149, 147, 150),
    )


def physical_expansion_batch(batch_index: int) -> tuple[ExpansionShard, ...]:
    """Return one nine-condition, pressure/topology-balanced batch."""
    if batch_index < 1:
        raise ValueError("batch_index must be positive.")
    first = 124 + 9 * (batch_index - 1)
    return (
        ExpansionShard("compact_normal", (first, first + 1), 0,
                       ("chain", "ring", "star")),
        ExpansionShard("compact_single", (first + 2, first + 3), 1,
                       ("chain", "ring", "star")),
        ExpansionShard("compact_double", (first + 4, first + 5), 2,
                       ("chain", "ring", "star")),
        ExpansionShard("full_normal", (first + 6,), 0,
                       ("fully_connected",)),
        ExpansionShard("full_single", (first + 7,), 1,
                       ("fully_connected",)),
        ExpansionShard("full_double", (first + 8,), 2,
                       ("fully_connected",)),
    )


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Collect one resumable v15.1 CANN physical expansion batch."
    )
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+",
                        default=(500, 501, 502))
    parser.add_argument("--output-directory", type=Path,
                        default=Path("results/cann/datasets"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        shards = physical_expansion_batch(arguments.batch_index)
    except ValueError as error:
        parser.error(str(error))
    output_directory = arguments.output_directory
    paths = tuple(
        output_directory / f"v15_1_cann_expansion_batch{arguments.batch_index}_"
        f"{shard.name}_condition{shard.condition_seeds[0]}_"
        f"{shard.condition_seeds[-1]}.npz"
        for shard in shards
    )
    merged_path = output_directory / (
        f"v15_1_cann_expansion_batch{arguments.batch_index}_condition"
        f"{shards[0].condition_seeds[0]}_{shards[-1].condition_seeds[-1]}.npz"
    )
    for shard, path in zip(shards, paths):
        status = "reuse" if path.exists() else "collect"
        print(f"{status}: {shard.name} {shard.condition_seeds} -> {path}")
        if arguments.dry_run or path.exists():
            continue
        collect_shard([
            "--condition-seeds", *(str(seed) for seed in shard.condition_seeds),
            "--noise-seeds", *(str(seed) for seed in arguments.noise_seeds),
            "--epochs", "0", "1", "2", "3", "4",
            "--lookahead", "1", "--episode-epochs", "10",
            "--decision-interval", "2", "--maximum-switches", "1",
            "--stratified-physical-scenarios",
            "--relative-modalities", "RANGE", "RANGE_RATE", "AZ_EL",
            "OPTICAL", "--cann-policy-features",
            "--navigation-dropout-node-count", str(shard.dropout_node_count),
            "--initial-topology-types", *shard.initial_topology_types,
            "--include-all-noise-observations", "--output", str(path),
        ])
    if arguments.dry_run:
        return merged_path
    missing = tuple(path for path in paths if not path.exists())
    if missing:
        raise RuntimeError(f"Expansion batch has missing shards: {missing}")
    datasets = tuple(load_topology_snapshot_tensor_dataset(path) for path in paths)
    merged = merge_topology_snapshot_tensor_datasets(datasets)
    save_topology_snapshot_tensor_dataset(merged, merged_path)
    print(
        f"merged {len(merged.groups)} groups / "
        f"{sum(len(group.action_kinds) for group in merged.groups)} actions "
        f"to {merged_path}"
    )
    return merged_path


if __name__ == "__main__":
    main()
