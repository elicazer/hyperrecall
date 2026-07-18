"""High-level recall: query text in, a scored, structured subgraph out.

``recall`` ties the pieces together:

  1. embed the query and find seed nodes (semantic + lexical),
  2. spread activation for ``k`` hops through hyperedges,
  3. assemble the lit-up nodes and edges into a :class:`Subgraph`,
  4. annotate contradictions and supersessions,
  5. trim to a token budget, reinforcing what survives (Hebbian access boost).

The returned :class:`Subgraph` renders to markdown or to a compact context
string suitable for stuffing into an LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..decay import DecayFn, exponential_decay
from ..models import CONTRADICTS, SUPERSEDES, Hyperedge, Node
from ..storage.embeddings import EmbedFn, hash_embed
from ..storage.sqlite_store import SqliteStore
from .activation import spread

# Rough chars-per-token heuristic for budgeting without a tokenizer dependency.
_CHARS_PER_TOKEN = 4


@dataclass
class ScoredNode:
    node: Node
    score: float
    hop: int
    superseded: bool = False
    contradicted_by: list[str] = field(default_factory=list)


@dataclass
class Subgraph:
    """The result of a recall: nodes + the hyperedges that connect them."""

    query: str
    nodes: list[ScoredNode]
    hyperedges: list[Hyperedge]

    # -- rendering ---------------------------------------------------------
    def to_markdown(self) -> str:
        lines = [f"# Recall: {self.query}", ""]
        if not self.nodes:
            lines.append("_(no memories activated)_")
            return "\n".join(lines)
        lines.append("## Memories")
        for sn in self.nodes:
            flags = []
            if sn.superseded:
                flags.append("superseded")
            if sn.contradicted_by:
                flags.append(f"contradicted×{len(sn.contradicted_by)}")
            flag = f"  _[{', '.join(flags)}]_" if flags else ""
            lines.append(
                f"- **{sn.node.text}** "
                f"(score={sn.score:.3f}, conf={sn.node.confidence:.2f}, hop={sn.hop}){flag}"
            )
        lines.append("")
        lines.append("## Relations")
        for e in self.hyperedges:
            roles = ", ".join(f"{m.role}={m.node_id[:12]}" for m in e.members)
            lines.append(f"- `{e.type}` (arity={e.arity}): {roles}")
        return "\n".join(lines)

    def to_context_string(self) -> str:
        """Dense, prompt-friendly rendering with conflict flags inline."""
        out = []
        for sn in self.nodes:
            prefix = ""
            if sn.superseded:
                prefix = "[OUTDATED] "
            elif sn.contradicted_by:
                prefix = "[CONFLICT] "
            out.append(f"{prefix}{sn.node.text}")
        return "\n".join(out)

    def node_ids(self) -> list[str]:
        return [sn.node.id for sn in self.nodes]

    def has_conflicts(self) -> bool:
        return any(sn.contradicted_by or sn.superseded for sn in self.nodes)


def recall(
    store: SqliteStore,
    query: str,
    *,
    embed: EmbedFn = hash_embed,
    k_hops: int = 2,
    max_seeds: int = 5,
    budget_tokens: int | None = None,
    prefer_newest: bool = True,
    reinforce_on_access: bool = True,
    curve: DecayFn = exponential_decay,
) -> Subgraph:
    """Retrieve a connected subgraph relevant to ``query``.

    See module docstring for the pipeline. ``budget_tokens`` (if given) trims
    the lowest-scoring nodes until the rendered context fits.
    """
    seeds = _find_seeds(store, query, embed=embed, max_seeds=max_seeds)
    result = spread(store, seeds, k_hops=k_hops, curve=curve)

    # Materialise scored nodes.
    scored: list[ScoredNode] = []
    for nid, score in result.scores.items():
        node = store.get_node(nid, curve=curve)
        if node is None:
            continue
        scored.append(ScoredNode(node=node, score=score, hop=result.hops.get(nid, 0)))

    by_id = {sn.node.id: sn for sn in scored}
    _annotate_supersession(store, by_id)
    _annotate_contradiction(store, by_id)

    # Rank: score first; superseded nodes sink if we prefer newest.
    def sort_key(sn: ScoredNode) -> tuple[float, float]:
        penalty = 0.4 if (prefer_newest and sn.superseded) else 1.0
        return (sn.score * penalty, sn.node.confidence)

    scored.sort(key=sort_key, reverse=True)

    if budget_tokens is not None:
        scored = _apply_budget(scored, budget_tokens)

    kept_ids = {sn.node.id for sn in scored}
    edges = [e for e in result.edges.values() if any(m.node_id in kept_ids for m in e.members)]

    if reinforce_on_access:
        for sn in scored:
            store.reinforce_node(sn.node.id, curve=curve)

    return Subgraph(query=query, nodes=scored, hyperedges=edges)


def _find_seeds(
    store: SqliteStore,
    query: str,
    *,
    embed: EmbedFn,
    max_seeds: int,
) -> dict[str, float]:
    """Blend semantic (embedding) and lexical (FTS) seed discovery."""
    seeds: dict[str, float] = {}
    qvec = embed(query)
    for nid, sim in store.semantic_search(qvec, top_k=max_seeds):
        if sim > 0:
            seeds[nid] = max(seeds.get(nid, 0.0), sim)
    # Lexical hits get a solid baseline energy so exact matches always seed.
    for nid in store.fts_search(query, limit=max_seeds):
        seeds[nid] = max(seeds.get(nid, 0.0), 0.6)
    return seeds


def _annotate_supersession(store: SqliteStore, by_id: dict[str, ScoredNode]) -> None:
    """Mark nodes that a ``Supersedes`` edge points *away from* as superseded.

    Convention: role ``old`` is the superseded node, role ``new`` is the winner.
    """
    for edge in store.edges_of_type(SUPERSEDES):
        old_ids = [m.node_id for m in edge.members if m.role == "old"]
        # Fallback: if roles unset, oldest member is the superseded one.
        if not old_ids and edge.members:
            oldest = min(edge.members, key=lambda m: m.node_id)
            old_ids = [oldest.node_id]
        for oid in old_ids:
            if oid in by_id:
                by_id[oid].superseded = True


def _annotate_contradiction(store: SqliteStore, by_id: dict[str, ScoredNode]) -> None:
    """Flag every node touched by a ``Contradicts`` edge with its opponents."""
    for edge in store.edges_of_type(CONTRADICTS):
        ids = [m.node_id for m in edge.members]
        for nid in ids:
            if nid in by_id:
                by_id[nid].contradicted_by.extend(x for x in ids if x != nid)


def _apply_budget(scored: list[ScoredNode], budget_tokens: int) -> list[ScoredNode]:
    kept: list[ScoredNode] = []
    used = 0
    for sn in scored:
        cost = max(1, len(sn.node.text) // _CHARS_PER_TOKEN)
        if used + cost > budget_tokens and kept:
            break
        kept.append(sn)
        used += cost
    return kept
