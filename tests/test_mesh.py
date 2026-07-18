"""End-to-end smoke tests for the public Mesh API."""

from __future__ import annotations

from hyperrecall import Mesh


def test_remember_and_recall_roundtrip():
    mesh = Mesh(":memory:")
    mesh.remember("Eli is building HyperRecall", participants=["Eli"], context={"topic": "HyperRecall"})
    mesh.remember("HyperRecall uses hypergraphs", context={"topic": "HyperRecall"})
    mesh.remember("Hypergraphs beat knowledge graphs for memory", context={"topic": "HyperRecall"})

    result = mesh.recall("what is hyperrecall")
    assert result.nodes, "recall should activate at least one node"
    texts = " ".join(sn.node.text.lower() for sn in result.nodes)
    assert "hyperrecall" in texts
    mesh.close()


def test_recall_returns_subgraph_not_flat_list():
    mesh = Mesh(":memory:")
    mesh.remember("Eli asked David about TEDx", participants=["Eli", "David"], context={"topic": "TEDx"})
    result = mesh.recall("TEDx")
    assert hasattr(result, "nodes")
    assert hasattr(result, "hyperedges")
    assert result.hyperedges, "recall should surface connecting hyperedges"
    md = result.to_markdown()
    assert "Recall:" in md and "Relations" in md
    ctx = result.to_context_string()
    assert isinstance(ctx, str) and len(ctx) > 0
    mesh.close()


def test_inspect_node_reports_edges_and_activation():
    mesh = Mesh(":memory:")
    node = mesh.remember("Eli lives in Newport", participants=["Eli"])
    info = mesh.inspect_node(node.id)
    assert info["text"] == "Eli lives in Newport"
    assert info["activation"] > 0
    assert any(e["type"] == "Experience" for e in info["edges"])
    mesh.close()


def test_budget_limits_result_size():
    mesh = Mesh(":memory:")
    for i in range(10):
        mesh.remember(f"Fact number {i} about hyperrecall and memory systems", context={"topic": "HyperRecall"})
    small = mesh.recall("hyperrecall memory", budget_tokens=20)
    big = mesh.recall("hyperrecall memory", budget_tokens=100000)
    assert len(small.nodes) <= len(big.nodes)
    mesh.close()


def test_stats_counts_grow():
    mesh = Mesh(":memory:")
    before = mesh.stats()["nodes"]
    mesh.remember("a new memory", participants=["Eli"])
    after = mesh.stats()["nodes"]
    assert after > before
    mesh.close()


def test_contradiction_is_surfaced_with_flag():
    mesh = Mesh(":memory:")
    a = mesh.remember("The TEDx event is in Newport", context={"topic": "TEDx"})
    b = mesh.remember("The TEDx event is in Irvine", context={"topic": "TEDx"})
    mesh.contradict(a.id, b.id, note="location conflict")

    pairs = mesh.contradictions()
    assert len(pairs) == 1
    result = mesh.recall("TEDx event location")
    flagged = [sn for sn in result.nodes if sn.contradicted_by]
    assert flagged, "recall must flag contradicting nodes"
    assert "CONFLICT" in result.to_context_string()
    mesh.close()


def test_supersession_prefers_newest_but_keeps_history():
    mesh = Mesh(":memory:")
    old = mesh.remember("The event is on Aug 20 2026", context={"topic": "TEDx"})
    new = mesh.remember("The event is on Aug 22 2026", context={"topic": "TEDx"})
    mesh.supersede(old.id, new.id, note="date moved")

    result = mesh.recall("event date", prefer_newest=True)
    ids = result.node_ids()
    assert old.id in ids, "history should still be retrievable"
    # The superseded node should be flagged.
    old_scored = [sn for sn in result.nodes if sn.node.id == old.id][0]
    assert old_scored.superseded is True
    assert "OUTDATED" in result.to_context_string()
    mesh.close()
