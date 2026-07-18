"""Round-trip tests for the portable Markdown + YAML file format."""

from __future__ import annotations

from hyperrecall import Hyperedge, HyperedgeMember, Mesh, Node
from hyperrecall.models import EXPERIENCE


def _build_mesh() -> Mesh:
    mesh = Mesh(":memory:")
    mesh.remember(
        "Eli asked David about TEDx applications",
        participants=["Eli", "David"],
        context={"topic": "TEDx", "session": "abc123"},
        confidence=0.9,
    )
    mesh.remember("HyperRecall uses a real hypergraph", context={"topic": "HyperRecall"})
    a = mesh.add_node(Node(text="claim one", confidence=0.7))
    b = mesh.add_node(Node(text="claim two", confidence=0.7))
    mesh.contradict(a.id, b.id, note="conflict")
    return mesh


def test_export_creates_expected_layout(tmp_path):
    mesh = _build_mesh()
    out = mesh.export(tmp_path / "export")
    assert (out / "manifest.yaml").exists()
    assert (out / "nodes").is_dir()
    assert (out / "edges").is_dir()
    node_files = list((out / "nodes").glob("*.md"))
    edge_files = list((out / "edges").glob("*.md"))
    assert len(node_files) == mesh.stats()["nodes"]
    assert len(edge_files) == mesh.stats()["hyperedges"]
    # files really are markdown + frontmatter
    sample = node_files[0].read_text()
    assert sample.startswith("---")
    mesh.close()


def test_roundtrip_is_lossless(tmp_path):
    mesh = _build_mesh()
    original = mesh.stats()
    orig_nodes = {n.id: n for n in mesh.store.all_nodes()}
    orig_edges = {e.id: e for e in mesh.store.all_hyperedges()}

    out = mesh.export(tmp_path / "export")
    mesh.close()

    restored = Mesh.import_dir(out, ":memory:")
    assert restored.stats() == original

    new_nodes = {n.id: n for n in restored.store.all_nodes()}
    new_edges = {e.id: e for e in restored.store.all_hyperedges()}

    assert set(new_nodes) == set(orig_nodes)
    assert set(new_edges) == set(orig_edges)

    for nid, on in orig_nodes.items():
        nn = new_nodes[nid]
        assert nn.text == on.text
        assert nn.kind == on.kind
        assert abs(nn.confidence - on.confidence) < 1e-9
        assert nn.metadata == on.metadata

    for eid, oe in orig_edges.items():
        ne = new_edges[eid]
        assert ne.type == oe.type
        assert ne.arity == oe.arity
        orig_members = sorted((m.node_id, m.role, m.weight) for m in oe.members)
        new_members = sorted((m.node_id, m.role, m.weight) for m in ne.members)
        assert orig_members == new_members

    restored.close()


def test_roundtrip_preserves_recall_behavior(tmp_path):
    mesh = _build_mesh()
    out = mesh.export(tmp_path / "export")
    mesh.close()

    restored = Mesh.import_dir(out, ":memory:")
    result = restored.recall("TEDx applications")
    assert result.nodes
    assert any("tedx" in sn.node.text.lower() for sn in result.nodes)
    assert restored.contradictions(), "contradiction edges survive the round-trip"
    restored.close()


def test_roundtrip_preserves_high_arity_edge(tmp_path):
    mesh = Mesh(":memory:")
    ids = [mesh.add_node(Node(text=f"n{i}")).id for i in range(4)]
    mesh.add_hyperedge(
        Hyperedge(type=EXPERIENCE, members=[HyperedgeMember(i, f"role{k}") for k, i in enumerate(ids)])
    )
    out = mesh.export(tmp_path / "export")
    mesh.close()

    restored = Mesh.import_dir(out, ":memory:")
    edge = restored.store.all_hyperedges()[0]
    assert edge.arity == 4
    restored.close()
