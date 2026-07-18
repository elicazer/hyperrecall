"""Tests for spreading-activation retrieval math."""

from __future__ import annotations

from hyperrecall import Hyperedge, HyperedgeMember, Mesh, Node
from hyperrecall.models import EXPERIENCE
from hyperrecall.retrieval.activation import spread


def _chain_mesh() -> tuple[Mesh, list[str]]:
    """A -- B -- C chain of hyperedges to test hop propagation."""
    mesh = Mesh(":memory:")
    a = mesh.add_node(Node(text="alpha apple"))
    b = mesh.add_node(Node(text="bravo banana"))
    c = mesh.add_node(Node(text="charlie cherry"))
    mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(a.id, "x"), HyperedgeMember(b.id, "y")]))
    mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(b.id, "y"), HyperedgeMember(c.id, "z")]))
    return mesh, [a.id, b.id, c.id]


def test_activation_spreads_from_seed():
    mesh, (a, b, c) = _chain_mesh()
    res = spread(mesh.store, {a: 1.0}, k_hops=2)
    assert res.scores[a] > 0
    assert b in res.scores, "activation should reach the 1-hop neighbour"
    assert c in res.scores, "activation should reach the 2-hop neighbour"
    mesh.close()


def test_activation_decays_with_distance():
    mesh, (a, b, c) = _chain_mesh()
    res = spread(mesh.store, {a: 1.0}, k_hops=2, hop_decay=0.5)
    assert res.scores[a] > res.scores[b] > res.scores[c], "energy must attenuate per hop"
    mesh.close()


def test_hop_limit_is_respected():
    mesh, (a, b, c) = _chain_mesh()
    res = spread(mesh.store, {a: 1.0}, k_hops=1)
    assert b in res.scores
    assert c not in res.scores, "2-hop node must not activate with k_hops=1"
    mesh.close()


def test_hyperedge_lights_all_members_at_once():
    """A single hyperedge co-activates ALL its other members from one seed."""
    mesh = Mesh(":memory:")
    seed = mesh.add_node(Node(text="seed"))
    others = [mesh.add_node(Node(text=f"member {i}")) for i in range(4)]
    members = [HyperedgeMember(seed.id, "seed")] + [HyperedgeMember(o.id, "m") for o in others]
    mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=members))
    res = spread(mesh.store, {seed.id: 1.0}, k_hops=1)
    for o in others:
        assert res.scores.get(o.id, 0.0) > 0, "every co-member should light up in one hop"
    mesh.close()


def test_returns_traversed_edges():
    mesh, (a, b, c) = _chain_mesh()
    res = spread(mesh.store, {a: 1.0}, k_hops=2)
    assert len(res.edges) >= 1
    for edge in res.edges.values():
        assert edge.arity >= 2
    mesh.close()


def test_recall_ranks_direct_match_first():
    mesh = Mesh(":memory:")
    mesh.remember("HyperRecall is a hypergraph memory", context={"topic": "HyperRecall"})
    mesh.remember("The weather in Newport is sunny", context={"topic": "weather"})
    res = mesh.recall("hypergraph memory")
    assert res.nodes[0].node.text.lower().startswith("hyperrecall")
    mesh.close()
