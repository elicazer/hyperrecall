"""Spreading-activation retrieval over the hypergraph.

The core idea, borrowed from cognitive psychology (Collins & Loftus, 1975): a
query lights up a few *seed* nodes, and that activation *spreads* outward
through the edges that touch them, attenuating with each hop. Because our edges
are hyperedges, activation entering one member of an edge lights up **all**
other members of that edge at once — a person, a project, a decision and an
outcome co-activate because they share one ``Experience`` edge.

Energy flowing across an edge is scaled by:

  * the edge's ``activation_weight`` (how strong the relation is),
  * the source and destination member ``weight`` (role coupling),
  * a per-hop ``decay`` factor (distance attenuation),

and each node's own live activation biases how much it seeds. The result is a
connected *subgraph*, not a ranked list of chunks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..decay import DecayFn, exponential_decay
from ..models import Hyperedge, Node
from ..storage.sqlite_store import SqliteStore


@dataclass
class ActivationResult:
    """Raw output of the spread: per-node activation and the edges traversed."""

    scores: dict[str, float]
    edges: dict[str, Hyperedge]
    seeds: dict[str, float]
    hops: dict[str, int] = field(default_factory=dict)


def spread(
    store: SqliteStore,
    seeds: dict[str, float],
    *,
    k_hops: int = 2,
    hop_decay: float = 0.5,
    min_energy: float = 1e-3,
    curve: DecayFn = exponential_decay,
) -> ActivationResult:
    """Run k-hop spreading activation from weighted ``seeds``.

    Args:
        seeds: mapping of seed ``node_id`` -> initial energy (usually the
            query/node similarity).
        k_hops: number of propagation rounds.
        hop_decay: multiplicative attenuation applied to energy each hop.
        min_energy: energy below this is not propagated further (prunes the
            frontier so the spread stays local and cheap).
        curve: forgetting curve used to read live node activation.

    Returns:
        An :class:`ActivationResult` with accumulated node scores, the
        hyperedges that carried energy, and the hop at which each node was first
        reached.
    """
    scores: dict[str, float] = defaultdict(float)
    edges_seen: dict[str, Hyperedge] = {}
    hops: dict[str, int] = {}

    # The frontier holds energy still to be propagated this round.
    frontier: dict[str, float] = {}
    for nid, energy in seeds.items():
        # Bias the seed by the node's own live salience (a hot memory seeds harder).
        live = store.live_activation(nid, curve=curve)
        e = energy * (0.5 + 0.5 * live)  # keep seeds meaningful even when cold
        scores[nid] += e
        frontier[nid] = e
        hops[nid] = 0

    for hop in range(1, k_hops + 1):
        next_frontier: dict[str, float] = defaultdict(float)
        for src_id, energy in frontier.items():
            if energy < min_energy:
                continue
            for edge in store.edges_for_node(src_id):
                edges_seen[edge.id] = edge
                src_w = _member_weight(edge, src_id)
                # Energy available to push through this edge.
                conducted = energy * edge.activation_weight * hop_decay
                if conducted < min_energy:
                    continue
                for member in edge.members:
                    if member.node_id == src_id:
                        continue
                    delivered = conducted * member.weight * src_w
                    if delivered < min_energy:
                        continue
                    scores[member.node_id] += delivered
                    next_frontier[member.node_id] += delivered
                    if member.node_id not in hops:
                        hops[member.node_id] = hop
        frontier = dict(next_frontier)
        if not frontier:
            break

    return ActivationResult(scores=dict(scores), edges=edges_seen, seeds=dict(seeds), hops=hops)


def _member_weight(edge: Hyperedge, node_id: str) -> float:
    for m in edge.members:
        if m.node_id == node_id:
            return m.weight
    return 1.0
