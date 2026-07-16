"""Query Planner v2 classification, routing, filtering, and assembly tests."""

from __future__ import annotations

import re

from meshmind import Hyperedge, HyperedgeMember, Mesh, Node
from meshmind.query.planner import QueryPlanner


def _paint_mesh() -> Mesh:
    """Two 'Melanie painted X' edges: one on-topic (sunrise), one distractor
    (horse) that retrieval ranks alongside it -- the reranker must separate them.
    """
    mesh = Mesh(":memory:")
    mel = Node("Melanie", kind="entity", created_at=100)
    sunrise = Node("Melanie painted a lake sunrise", created_at=200)
    horse = Node("Melanie painted a horse", created_at=300)
    for node in (mel, sunrise, horse):
        mesh.add_node(node)
    mesh.add_hyperedge(Hyperedge(
        type="Experience", created_at=200,
        provenance={"source_text": "Melanie painted a lake sunrise", "timestamp": 200},
        members=[HyperedgeMember(mel.id, "person"), HyperedgeMember(sunrise.id, "statement")],
    ))
    mesh.add_hyperedge(Hyperedge(
        type="Experience", created_at=300,
        provenance={"source_text": "Melanie painted a horse", "timestamp": 300},
        members=[HyperedgeMember(mel.id, "person"), HyperedgeMember(horse.id, "statement")],
    ))
    return mesh


def _memory_mesh() -> tuple[Mesh, Node, Node, Node]:
    mesh = Mesh(":memory:")
    eli = Node("Eli", kind="entity", created_at=100)
    old = Node("Eli lives in Newport", created_at=200)
    new = Node("Eli lives in Irvine", created_at=300)
    for node in (eli, old, new):
        mesh.add_node(node)
    mesh.add_hyperedge(Hyperedge(
        type="Experience", created_at=200,
        provenance={"source_text": "Eli lives in Newport", "timestamp": 200},
        members=[HyperedgeMember(eli.id, "person"), HyperedgeMember(old.id, "statement")],
    ))
    mesh.add_hyperedge(Hyperedge(
        type="Experience", created_at=300,
        provenance={"source_text": "Eli lives in Irvine", "timestamp": 300},
        members=[HyperedgeMember(eli.id, "person"), HyperedgeMember(new.id, "statement")],
    ))
    mesh.contradict(old.id, new.id)
    mesh.supersede(old.id, new.id)
    return mesh, eli, old, new


def test_classification_all_five_classes_offline():
    mesh, *_ = _memory_mesh()
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=None, use_gemini=False)
    cases = {
        "What does Eli remember?": "single_hop",
        "Why did the move happen and then affect work?": "multi_hop",
        "Where did Eli live before 2025?": "temporal",
        "Summarize remembered locations": "open_domain",
        "What is the current president's stock price?": "adversarial",
    }
    assert {q: planner.classify(q).question_class for q in cases} == cases


def test_gemini_json_classification_and_multihop_decomposition():
    responses = iter([
        {"question_class": "multi_hop", "entities": ["Eli"],
         "time_constraints": [], "question_kind": "causal"},
        {"sub_questions": ["Why did Eli move?", "How did the move affect work?"]},
    ])
    mesh, *_ = _memory_mesh()
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=lambda _: next(responses))
    plan = planner.decompose("Why did Eli move and how did it affect work?", planner.classify("q"))
    assert plan.used_llm
    assert len(plan.sub_questions) == 2


def test_entity_retrieval_materializes_edge_records():
    mesh, _, _, new = _memory_mesh()
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=None, use_gemini=False)
    result = planner.recall("What does Eli remember?", reinforce_on_access=False)
    assert result.plan.question_class == "single_hop"
    assert result.results
    assert any(p.node_id == new.id for row in result.results for p in row.participants)
    assert all(isinstance(row.provenance, dict) and row.timestamp for row in result.results)


def test_superseded_contradiction_keeps_newest_unless_history_requested():
    mesh, _, old, new = _memory_mesh()
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=None, use_gemini=False)
    current = planner.recall("What does Eli remember?", reinforce_on_access=False)
    assert new.id in current.node_ids()
    assert old.id not in current.node_ids()

    history = planner.recall(
        "What did we used to think about Eli?", reinforce_on_access=False
    )
    assert old.id in history.node_ids() and new.id in history.node_ids()
    assert any("HISTORICAL_CONFLICT" in row.annotations for row in history.results)


def test_temporal_before_filter():
    mesh, *_ = _memory_mesh()
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=lambda _: {
        "question_class": "temporal", "entities": ["Eli"],
        "time_constraints": [{"relation": "before", "value": "1971-01-01"}],
        "question_kind": "time",
    })
    result = planner.recall("Where was Eli before 1971-01-01?", reinforce_on_access=False)
    assert result.results
    assert all(row.timestamp < 31_536_000 for row in result.results)


def test_rerank_promotes_llm_preferred_edge():
    """A batched-LLM score reorders candidates; the on-topic edge wins top-1."""
    mesh = _paint_mesh()

    def fake_gemini(prompt: str):
        # Score each listed candidate high iff its text is on-topic (sunrise).
        listed = [line for line in prompt.splitlines() if re.match(r"^\d+: ", line)]
        return {"scores": [9 if "sunrise" in line else 1 for line in listed]}

    planner = QueryPlanner(
        mesh.store, embed=mesh.embed, llm=None, use_gemini=False, rerank_llm=fake_gemini
    )
    result = planner.recall(
        "What did Melanie paint recently?", reinforce_on_access=False,
        rerank=True, k_candidate=25, k_final=1,
    )
    assert result.rerank is not None and result.rerank["applied"]
    # Exactly one edge survives, and it is the sunrise (not the higher-scored,
    # newer horse distractor that plain retrieval would have surfaced).
    assert len(result.hyperedges) == 1
    assert "sunrise" in result.to_context_string()
    assert "horse" not in result.to_context_string()
    # The rerank trace records both candidates with their scores.
    assert result.rerank["n_candidates"] == 2
    top = [d for d in result.rerank["deltas"] if d["in_top_k"]]
    assert len(top) == 1 and top[0]["llm_score"] == 9


def test_rerank_falls_back_on_malformed_scores():
    """Malformed LLM output must not crash: keep original order, applied=False."""
    mesh = _paint_mesh()

    def broken_gemini(prompt: str):
        return {"scores": [5]}  # wrong count vs. the 2 candidates

    planner = QueryPlanner(
        mesh.store, embed=mesh.embed, llm=None, use_gemini=False, rerank_llm=broken_gemini
    )
    result = planner.recall(
        "What did Melanie paint?", reinforce_on_access=False,
        rerank=True, k_final=8,
    )
    assert result.rerank is not None
    assert result.rerank["applied"] is False
    assert result.rerank["reason"] == "score_count_mismatch"
    # No crash and both edges preserved (fallback keeps retrieval order).
    assert len(result.hyperedges) == 2


def test_public_api_opt_in_and_unknown_plan(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mesh, *_ = _memory_mesh()
    result = mesh.recall("What does Eli remember?", plan="v2", reinforce_on_access=False)
    assert result.plan is not None
    try:
        mesh.recall("test", plan="v3")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "unknown recall plan" in str(exc)
    else:
        raise AssertionError("unknown plan must fail loudly")
