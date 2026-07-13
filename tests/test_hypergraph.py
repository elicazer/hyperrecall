"""Tests that enforce the 'real hypergraph, not a disguised knowledge graph' rule."""

from __future__ import annotations

import pytest

from meshmind import Hyperedge, HyperedgeMember, Mesh, Node
from meshmind.models import EXPERIENCE


def test_hyperedge_has_arity_at_least_three():
    """The non-negotiable: a hyperedge must be able to bind N >= 3 nodes at once.

    An ``Experience`` from a two-participant turn binds a statement + 2 people
    (+ a topic node) — arity >= 3. If this ever drops to 2 we've quietly become
    a triple store.
    """
    mesh = Mesh(":memory:")
    mesh.remember(
        "Eli asked David about TEDx applications",
        participants=["Eli", "David"],
        context={"topic": "TEDx"},
    )
    edges = mesh.store.all_hyperedges()
    assert edges, "an Experience hyperedge should have been created"
    max_arity = max(e.arity for e in edges)
    assert max_arity >= 3, f"expected a hyperedge with arity >= 3, got {max_arity}"
    mesh.close()


def test_hyperedge_members_carry_roles():
    mesh = Mesh(":memory:")
    mesh.remember("Eli decided to apply", participants=["Eli"], context={"topic": "TEDx"})
    edge = mesh.store.all_hyperedges()[0]
    roles = {m.role for m in edge.members}
    assert "statement" in roles
    assert "participant" in roles
    # roles are distinct per structural position, not a single 'relation' label
    assert len(roles) >= 2
    mesh.close()


def test_arbitrary_arity_experience_edge():
    """Build a 5-ary Experience edge directly: person+project+decision+outcome+time."""
    mesh = Mesh(":memory:")
    person = mesh.add_node(Node(text="Eli", kind="entity"))
    project = mesh.add_node(Node(text="TEDx San Joaquin Hills", kind="project"))
    decision = mesh.add_node(Node(text="apply on Aug 22 2026", kind="decision"))
    outcome = mesh.add_node(Node(text="submitted application", kind="outcome"))
    when = mesh.add_node(Node(text="2026-07-13T20:00", kind="time"))

    edge = mesh.add_hyperedge(
        Hyperedge(
            type=EXPERIENCE,
            members=[
                HyperedgeMember(person.id, "person"),
                HyperedgeMember(project.id, "project"),
                HyperedgeMember(decision.id, "decision"),
                HyperedgeMember(outcome.id, "outcome"),
                HyperedgeMember(when.id, "time"),
            ],
        )
    )
    assert edge.arity == 5
    fetched = mesh.store.get_hyperedge(edge.id)
    assert fetched is not None and fetched.arity == 5
    assert fetched.role_of(project.id) == "project"
    mesh.close()


def test_edge_with_arity_below_two_is_rejected():
    mesh = Mesh(":memory:")
    n = mesh.add_node(Node(text="lonely node"))
    with pytest.raises(ValueError):
        mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(n.id, "x")]))
    mesh.close()


def test_same_node_participates_in_multiple_edges():
    mesh = Mesh(":memory:")
    a = mesh.add_node(Node(text="Eli"))
    b = mesh.add_node(Node(text="MeshMind"))
    c = mesh.add_node(Node(text="TEDx"))
    mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(a.id, "p"), HyperedgeMember(b.id, "proj")]))
    mesh.add_hyperedge(Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(a.id, "p"), HyperedgeMember(c.id, "topic")]))
    edges = mesh.store.edges_for_node(a.id)
    assert len(edges) == 2
    mesh.close()
