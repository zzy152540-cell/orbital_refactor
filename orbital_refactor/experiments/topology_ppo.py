"""Compatibility import for :mod:`experiments.training.topology_ppo`."""

from experiments.training import topology_ppo as _implementation

globals().update({
    name: getattr(_implementation, name)
    for name in dir(_implementation)
    if not name.startswith("__")
})
