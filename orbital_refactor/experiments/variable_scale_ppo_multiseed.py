"""Compatibility import for variable-scale multi-seed PPO summaries."""

from experiments.training import variable_scale_ppo_multiseed as _implementation

globals().update({
    name: getattr(_implementation, name)
    for name in dir(_implementation)
    if not name.startswith("__")
})
