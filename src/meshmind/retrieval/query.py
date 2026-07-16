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

import numpy as np

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
    sim: float = 0.0  # direct query<->node cosine similarity (set when sim_rerank>0)
    rank_score: float = 0.0  # blended activation+similarity score used for ranking


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
    sim_rerank: float = 0.0,
    curve: DecayFn = exponential_decay,
) -> Subgraph:
    """Retrieve a connected subgraph relevant to ``query``.

    See module docstring for the pipeline. ``budget_tokens`` (if given) trims
    the lowest-scoring nodes until the rendered context fits.

    ``sim_rerank`` (0..1) blends direct query<->node cosine similarity into the
    final ranking. Spreading activation is great at *finding* a relevant
    neighbourhood but tends to reward well-connected hub nodes (a frequently
    mentioned person) over the specific statement that actually answers the
    query. Reranking the whole lit-up neighbourhood by semantic similarity to
    the query pulls the on-topic nodes back to the top. ``0`` keeps the pure
    activation ordering (default, backward-compatible).
    """
    qvec = embed(query)
    seeds = _find_seeds(store, query, embed=embed, max_seeds=max_seeds, qvec=qvec)
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

    if sim_rerank > 0.0 and scored:
        _apply_sim_rerank(store, scored, qvec, weight=sim_rerank)

    # Rank: score first; superseded nodes sink if we prefer newest.
    def sort_key(sn: ScoredNode) -> tuple[float, float]:
        penalty = 0.4 if (prefer_newest and sn.superseded) else 1.0
        base = sn.rank_score if sim_rerank > 0.0 else sn.score
        return (base * penalty, sn.node.confidence)

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
    qvec: "np.ndarray | None" = None,
) -> dict[str, float]:
    """Blend semantic (embedding) and lexical (FTS) seed discovery."""
    seeds: dict[str, float] = {}
    if qvec is None:
        qvec = embed(query)
    for nid, sim in store.semantic_search(qvec, top_k=max_seeds):
        if sim > 0:
            seeds[nid] = max(seeds.get(nid, 0.0), sim)
    # Lexical hits get a solid baseline energy so exact matches always seed.
    for nid in store.fts_search(query, limit=max_seeds):
        seeds[nid] = max(seeds.get(nid, 0.0), 0.6)
    return seeds


def _apply_sim_rerank(
    store: SqliteStore,
    scored: list[ScoredNode],
    qvec: "np.ndarray",
    *,
    weight: float,
) -> None:
    """Blend each node's activation score with its direct query similarity.

    Activation scores are normalised to ``[0, 1]`` (relative to the strongest
    node in this recall) so they combine sensibly with cosine similarity, which
    already lives in ``[-1, 1]``. ``rank_score`` becomes
    ``(1-weight)*norm_activation + weight*sim`` and is what the caller sorts on.
    """
    mat, ids = store.embedding_matrix()
    idx = {nid: i for i, nid in enumerate(ids)}
    qn = _l2(np.asarray(qvec, dtype=np.float32))
    max_score = max((sn.score for sn in scored), default=0.0) or 1.0
    for sn in scored:
        i = idx.get(sn.node.id)
        sn.sim = float(mat[i] @ qn) if i is not None and mat.size else 0.0
        norm_act = sn.score / max_score
        sn.rank_score = (1.0 - weight) * norm_act + weight * max(sn.sim, 0.0)


def _l2(vec: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


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
