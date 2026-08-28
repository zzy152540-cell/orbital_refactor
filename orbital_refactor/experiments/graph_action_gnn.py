"""Compatibility import for :mod:`experiments.training.graph_action_gnn`."""

from experiments.training import graph_action_gnn as _implementation

globals().update({
    name: getattr(_implementation, name)
    for name in dir(_implementation)
    if not name.startswith("__")
})
