"""Tests for the dense, coreference-aware v2 extractor.

These run fully offline (``mock_mode=True`` / no ``GEMINI_API_KEY``): they never
touch Gemini. They cover the three pass boundaries we care about — dense
extraction, N-ary hyperedge materialization, and cross-turn coreference.
"""
from __future__ import annotations

from meshmind import Mesh
from meshmind.ingest.extractor_v2 import (
    HYPEREDGE_VOCAB,
    EntityV2,
    ExtractorV2,
    HyperedgeV2,
    RelationParticipant,
    TurnExtraction,
    normalize_name,
)


def _mesh() -> Mesh:
    return Mesh(":memory:")


def test_normalize_name_doctest_cases():
    assert normalize_name("Caroline's") == "caroline"
    assert normalize_name("  The  Border-Collie ") == "the border collie"
    assert normalize_name("Luna") == normalize_name("luna")


def test_extract_turn_is_dense():
    ext = ExtractorV2(mock_mode=True)
    te = ext.extract_turn(
        "Caroline adopted Luna and moved to Portland.", speaker="Caroline"
    )
    assert isinstance(te, TurnExtraction)
    # speaker + Luna + Portland = 3 entities, several relations
    assert len(te.entities) >= 3
    assert len(te.hyperedges) >= 2
    # every hyperedge type is in the fixed vocabulary
    assert all(h.type in HYPEREDGE_VOCAB for h in te.hyperedges)


def test_ingest_persists_nary_hyperedges():
    mesh = _mesh()
    ext = ExtractorV2(mock_mode=True)
    te = ext.ingest(mesh, "Caroline adopted Luna in Portland.", speaker="Caroline")
    assert te.edge_ids, "expected at least one hyperedge persisted"
    stats = mesh.stats()
    assert stats["hyperedges"] == len(te.edge_ids)
    # every persisted edge is a real N-ary relation (arity >= 2)
    for eid in te.edge_ids:
        edge = mesh.store.get_hyperedge(eid)
        assert edge is not None and edge.arity >= 2
        # a summary node and the raw-turn node are always members
        assert "summary" in edge.metadata


def test_coreference_reuses_entity_id_across_turns():
    mesh = _mesh()
    ext = ExtractorV2(mock_mode=True)
    t1 = ext.ingest(mesh, "Caroline adopted a dog named Luna.", speaker="Caroline")
    t2 = ext.ingest(mesh, "Luna is doing great, said Melanie.", speaker="Melanie")
    # "Luna" seen in both turns must map to the SAME node id (coreference).
    luna_ids = {
        r["id"]
        for r in mesh.store._conn.execute(
            "SELECT id FROM nodes WHERE lower(text)='luna'"
        ).fetchall()
    }
    assert len(luna_ids) == 1, f"Luna should be one entity, got {luna_ids}"
    # ...and the second turn must have recorded a merge decision for it.
    assert any(m["canonical"].lower() == "luna" for m in t2.merges)
    assert t1.entity_ids["Luna"] == t2.entity_ids["Luna"]


def test_mesh_ingest_text_v2_flag():
    mesh = _mesh()
    # String flag path — no extractor instance needed.
    te = mesh.ingest_text(
        "Melanie prefers tea over coffee.",
        extractor="v2",
        mock_mode=True,
        speaker="Melanie",
    )
    assert isinstance(te, TurnExtraction)
    assert mesh.stats()["hyperedges"] >= 1


def test_duplicate_participants_are_deduped():
    # Two participants that canonicalize to the SAME node id must not violate
    # the store's UNIQUE(hyperedge_id, node_id) constraint (regression: conv-26
    # D1:2 crashed on this).
    mesh = _mesh()
    ext = ExtractorV2(mock_mode=True)
    forced = TurnExtraction(
        source_text="Luna chased Luna's tail.",
        entities=[EntityV2(name="Luna", type="Person", description="a dog")],
        hyperedges=[
            HyperedgeV2(
                type="Action",
                summary="Luna chased her own tail.",
                participants=[
                    RelationParticipant(entity="Luna", role="subject"),
                    RelationParticipant(entity="Luna", role="object"),
                ],
            )
        ],
    )
    ext.extract_turn = lambda *a, **k: forced  # type: ignore[assignment]
    te = ext.ingest(mesh, forced.source_text, speaker="Luna")
    assert len(te.edge_ids) == 1
    edge = mesh.store.get_hyperedge(te.edge_ids[0])
    assert edge is not None
    # summary + turn + one deduped Luna == arity 3, no duplicate node ids
    assert edge.arity == 3
    assert len(edge.node_ids) == len(set(edge.node_ids))


def test_empty_turn_is_noop():
    mesh = _mesh()
    ext = ExtractorV2(mock_mode=True)
    te = ext.ingest(mesh, "   ", speaker="Caroline")
    assert te.edge_ids == []
    assert mesh.stats()["nodes"] == 0


def test_dataclass_validation_coerces_bad_input():
    # Off-vocabulary hyperedge type falls back to Statement, not a crash.
    h = HyperedgeV2.from_dict(
        {"type": "MadeUpType", "summary": "x", "participants": [{"entity": "A", "role": "subject"}]}
    )
    assert h is not None and h.type == "Statement"
    # Off-vocabulary entity type falls back to Artifact.
    e = EntityV2.from_dict({"name": "Widget", "type": "Nonsense", "description": "d"})
    assert e is not None and e.type == "Artifact"
    # Missing name -> dropped.
    assert EntityV2.from_dict({"type": "Person"}) is None
    assert RelationParticipant.from_dict({"role": "subject"}) is None
