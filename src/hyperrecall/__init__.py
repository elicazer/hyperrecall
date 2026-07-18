"""HyperRecall — hypergraph memory for LLM agents.

Memory that works the way brains do: a web of experiences with spreading
activation, decay, reinforcement, contradiction and supersession — not a bag of
chunks.

Public surface::

    from hyperrecall import Mesh, Node, Hyperedge, HyperedgeMember
"""

from __future__ import annotations

from .decay import exponential_decay, power_law_decay, reinforce
from .mesh import Mesh
from .models import (
    CONTRADICTS,
    EXPERIENCE,
    SUPERSEDES,
    Hyperedge,
    HyperedgeMember,
    Node,
)
from .retrieval.query import ScoredNode, Subgraph

__version__ = "0.0.1"

__all__ = [
    "Mesh",
    "Node",
    "Hyperedge",
    "HyperedgeMember",
    "Subgraph",
    "ScoredNode",
    "EXPERIENCE",
    "CONTRADICTS",
    "SUPERSEDES",
    "exponential_decay",
    "power_law_decay",
    "reinforce",
    "__version__",
]
