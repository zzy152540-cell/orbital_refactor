"""Compatibility import for :mod:`experiments.training.topology_ppo_stage0`."""

from experiments.training import topology_ppo_stage0 as _implementation

globals().update({
    name: getattr(_implementation, name)
    for name in dir(_implementation)
    if not name.startswith("__")
})
