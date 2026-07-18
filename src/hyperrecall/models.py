"""Core data model for HyperRecall.

A HyperRecall memory is a *hypergraph*:

- :class:`Node` — a memory unit (a fact, an entity, a decision, an outcome).
- :class:`Hyperedge` — a first-class relation connecting **N >= 2** nodes.
- :class:`HyperedgeMember` — the join between a hyperedge and one node,
  carrying that node's ``role`` within the edge and a role ``weight``.

This is deliberately *not* a (head, relation, tail) triple store. A single
``Experience`` hyperedge can bind a person, a project, a decision, an outcome
and a timestamp all at once, and each participant keeps its role. See
``DESIGN.md`` for why that matters.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    """Return a short, sortable-ish unique id like ``node_1a2b3c4d``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ts() -> float:
    """Current wall-clock time as a UNIX timestamp (seconds, float)."""
    return time.time()


# Canonical hyperedge types. These are conventions, not a closed enum — callers
# may introduce new types — but the retrieval/contradiction/supersession logic
# gives special meaning to the ones listed here.
EXPERIENCE = "Experience"          # episodic: person + project + decision + outcome + time
CONTRADICTS = "Contradicts"        # two (or more) nodes that cannot both be true
SUPERSEDES = "Supersedes"          # a newer node replaces an older one
REFINES = "Refines"                # a node clarifies/extends another
CAUSED_BY = "CausedBy"             # causal link
MENTIONED_TOGETHER = "MentionedTogether"  # weak co-occurrence

HYPEREDGE_TYPES = (
    EXPERIENCE,
    CONTRADICTS,
    SUPERSEDES,
    REFINES,
    CAUSED_BY,
    MENTIONED_TOGETHER,
)


@dataclass
class Node:
    """A single memory unit.

    ``activation`` is the current, time-decayed salience of the node. It is
    boosted on access (reinforcement) and decays over time (forgetting). It is
    the quantity that spreading-activation retrieval propagates.
    """

    text: str
    id: str = field(default_factory=lambda: new_id("node"))
    kind: str = "fact"                 # fact | entity | decision | outcome | ...
    confidence: float = 1.0            # [0, 1] uncertainty of the memory itself
    activation: float = 1.0            # current salience (decays / reinforces)
    decay_rate: float = 0.05           # per-day decay strength for this node
    created_at: float = field(default_factory=now_ts)
    last_access: float = field(default_factory=now_ts)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)


@dataclass
class HyperedgeMember:
    """One node's participation in a hyperedge.

    ``role`` describes *how* the node participates (e.g. ``"subject"``,
    ``"project"``, ``"decision"``, ``"outcome"``, ``"time"``). ``weight`` scales
    how strongly activation flows to/from this node through the edge.
    """

    node_id: str
    role: str = "member"
    weight: float = 1.0


@dataclass
class Hyperedge:
    """A first-class relation over an arbitrary number of nodes.

    A hyperedge holds **N >= 2** members. Its ``activation_weight`` and
    ``decay_rate`` govern how much activation the edge conducts and how quickly
    that conductance fades. ``provenance`` records where the edge came from
    (which ingestion turn / session / tool).
    """

    type: str
    members: list[HyperedgeMember] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("edge"))
    activation_weight: float = 1.0
    decay_rate: float = 0.03
    confidence: float = 1.0
    created_at: float = field(default_factory=now_ts)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)

    @property
    def arity(self) -> int:
        """Number of nodes this hyperedge connects."""
        return len(self.members)

    @property
    def node_ids(self) -> list[str]:
        return [m.node_id for m in self.members]

    def role_of(self, node_id: str) -> str | None:
        for m in self.members:
            if m.node_id == node_id:
                return m.role
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dumps_json(obj: Any) -> str:
    """Compact, deterministic JSON for storing dict columns."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def loads_json(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)
