from __future__ import annotations

import json

from experiments.multitarget_dynamic_membership_acceptance import (
    run_multitarget_dynamic_membership_acceptance,
)


def main() -> None:
    result = run_multitarget_dynamic_membership_acceptance()
    print(json.dumps({
        "initialNodeIds": list(result.initial_node_ids),
        "joinedNodeIds": list(result.joined_node_ids),
        "removedNodeIds": list(result.removed_node_ids),
        "initialCooperativeTargetCount": (
            result.initial_cooperative_target_count
        ),
        "finalCooperativeTargetCount": result.final_cooperative_target_count,
        "identitySwitchCount": result.identity_switch_count,
        "falseCooperativeBirthCount": result.false_cooperative_birth_count,
        "departedContributorCount": result.departed_contributor_count,
        "finalTargetCount": result.final_target_count,
        "brainTargetCount": result.brain_target_count,
        "brainValidFraction": result.brain_valid_fraction,
        "brainDirectionNeuronCount": result.brain_direction_neuron_count,
        "brainPlaceHeatmapShape": list(result.brain_place_heatmap_shape),
        "brainPlaceNormalizationError": (
            result.brain_place_normalization_error
        ),
        "passed": result.passed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
