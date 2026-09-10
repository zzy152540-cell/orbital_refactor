from experiments.run_v15_cann_physical_expansion_batch import (
    physical_expansion_condition_split,
    physical_expansion_cross_validation_folds,
    physical_expansion_batch,
)


def test_physical_expansion_batch_is_balanced_and_disjoint():
    first = physical_expansion_batch(1)
    second = physical_expansion_batch(2)
    assert tuple(seed for shard in first for seed in shard.condition_seeds) == (
        124, 125, 126, 127, 128, 129, 130, 131, 132,
    )
    assert tuple(seed for shard in second for seed in shard.condition_seeds) == (
        133, 134, 135, 136, 137, 138, 139, 140, 141,
    )
    assert [shard.dropout_node_count for shard in first] == [0, 1, 2, 0, 1, 2]
    assert all(
        shard.initial_topology_types == ("fully_connected",)
        for shard in first[3:]
    )


def test_physical_expansion_condition_split_is_disjoint_and_complete():
    split = physical_expansion_condition_split()
    groups = (set(split.training), set(split.validation), set(split.test))
    assert not any(
        left & right
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    )
    assert set.union(*groups) == set(range(124, 151))
    assert tuple(map(len, groups)) == (15, 6, 6)


def test_physical_expansion_cross_validation_leaves_each_batch_out_once():
    folds = physical_expansion_cross_validation_folds()
    assert len(folds) == 3
    assert all(len(fold.training) == 18 for fold in folds)
    assert all(len(fold.validation) == 9 for fold in folds)
    assert all(not set(fold.training) & set(fold.validation) for fold in folds)
    assert set.union(*(set(fold.validation) for fold in folds)) == set(
        range(124, 151)
    )
