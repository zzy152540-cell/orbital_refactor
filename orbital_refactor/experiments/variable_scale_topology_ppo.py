"""Compatibility import for variable-scale PPO training implementation."""

from experiments.training import variable_scale_topology_ppo as _implementation

globals().update({
    name: getattr(_implementation, name)
    for name in dir(_implementation)
    if not name.startswith("__")
})
