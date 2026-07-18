"""Turn raw text into nodes + hyperedges.

This is intentionally a **heuristic stub**. Production HyperRecall will use an LLM
to decompose a conversation turn into atomic memory units and to type the
relations between them. Until then we do something honest and deterministic:

  * create one primary ``fact`` node for the whole utterance,
  * create one ``entity`` node per named participant,
  * bind them with a single ``Experience`` hyperedge whose members carry roles
    (``statement``, ``participant``, ``context``...).

Crucially this already produces a **real hyperedge of arity >= 3** for any turn
with two or more participants — proving the substrate before the LLM lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import (
    EXPERIENCE,
    Hyperedge,
    HyperedgeMember,
    Node,
)


@dataclass
class Extraction:
    """The nodes and edges pulled out of one ingest call."""

    nodes: list[Node] = field(default_factory=list)
    hyperedges: list[Hyperedge] = field(default_factory=list)
    primary: Node | None = None


def extract(
    text: str,
    *,
    participants: list[str] | None = None,
    context: dict[str, Any] | None = None,
    confidence: float = 1.0,
    edge_type: str = EXPERIENCE,
    provenance: dict[str, Any] | None = None,
) -> Extraction:
    """Decompose one utterance into a small hypergraph fragment."""
    participants = participants or []
    context = context or {}
    ex = Extraction()

    primary = Node(text=text.strip(), kind="fact", confidence=confidence, metadata=dict(context))
    ex.nodes.append(primary)
    ex.primary = primary

    members: list[HyperedgeMember] = [HyperedgeMember(primary.id, role="statement", weight=1.0)]

    for name in participants:
        person = Node(text=name, kind="entity", confidence=confidence, metadata={"role": "participant"})
        ex.nodes.append(person)
        members.append(HyperedgeMember(person.id, role="participant", weight=0.8))

    # Promote salient context keys to their own nodes so they can be recalled.
    for key in ("topic", "project", "decision", "outcome"):
        val = context.get(key)
        if isinstance(val, str) and val.strip():
            cnode = Node(text=val.strip(), kind=key, confidence=confidence, metadata={"from": key})
            ex.nodes.append(cnode)
            members.append(HyperedgeMember(cnode.id, role=key, weight=0.7))

    # Only build the edge if it's genuinely a relation (arity >= 2).
    if len(members) >= 2:
        edge = Hyperedge(
            type=edge_type,
            members=members,
            confidence=confidence,
            provenance=provenance or {},
            metadata={"session": context.get("session")} if context.get("session") else {},
        )
        ex.hyperedges.append(edge)

    return ex
