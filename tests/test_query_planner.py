"""Query Planner v2 classification, routing, filtering, and assembly tests."""

from __future__ import annotations

from meshmind import Hyperedge, HyperedgeMember, Mesh, Node
from meshmind.query.planner import QueryPlanner


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
    current = planner.recall("What does Eli remember?", reinforce_on_access=False, moat=True)
    assert new.id in current.node_ids()
    assert old.id not in current.node_ids()

    history = planner.recall(
        "What did we used to think about Eli?", reinforce_on_access=False, moat=True
    )
    assert old.id in history.node_ids() and new.id in history.node_ids()
    assert {annotation for row in history.results for annotation in row.annotations} >= {
        "before", "after"
    }


def test_moat_supersession_annotates_surviving_edge_with_topic_and_date(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mesh, _, old, new = _memory_mesh()
    result = mesh.recall(
        "Where does Eli live?", plan="v2-moat", reinforce_on_access=False
    )
    assert old.id not in result.node_ids() and new.id in result.node_ids()
    annotations = [annotation for row in result.results for annotation in row.annotations]
    assert any(
        annotation.startswith("supersedes previous statement of Eli lives in Irvine on 1970-01-01")
        for annotation in annotations
    )


def test_moat_direct_contradiction_prefers_recent_and_annotates(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mesh, eli, old, new = _memory_mesh()
    # Isolate contradiction behavior from the helper's supersession relation.
    mesh.store._conn.execute("DELETE FROM hyperedges WHERE type = ?", ("Supersedes",))
    mesh.store._conn.commit()
    result = mesh.recall(
        "What does Eli remember?", plan="v2-moat", reinforce_on_access=False
    )
    assert old.id not in result.node_ids() and new.id in result.node_ids() and eli.id in result.node_ids()
    assert any(
        "this contradicts an earlier statement on 1970-01-01, preferring recent"
        in row.annotations
        for row in result.results
    )


def test_moat_change_question_labels_before_and_after(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mesh, _, old, new = _memory_mesh()
    result = mesh.recall(
        "How did Eli's view change?", plan="v2-moat", reinforce_on_access=False
    )
    assert old.id in result.node_ids() and new.id in result.node_ids()
    labels = {annotation for row in result.results for annotation in row.annotations}
    assert {"before", "after"}.issubset(labels)


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
