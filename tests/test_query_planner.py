"""Query Planner v2 classification, routing, filtering, and assembly tests."""

from __future__ import annotations

from meshmind import Hyperedge, HyperedgeMember, Mesh, Node
from meshmind.query.planner import QueryPlan, QueryPlanner
import meshmind.query.planner as planner_module


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


def test_chain_conditions_next_retrieval_and_exposes_reasoning(monkeypatch):
    mesh, *_ = _memory_mesh()
    calls: list[str] = []
    answers = iter(["Newport", "Irvine"])

    def fake_llm(prompt: str):
        if prompt.startswith("Classify"):
            return {"question_class": "multi_hop", "entities": ["Eli"],
                    "time_constraints": [], "question_kind": "causal"}
        if prompt.startswith("Decompose"):
            return {"sub_questions": ["Where did Eli live first?", "Where next?"]}
        return {"answer": next(answers)}

    real_recall = planner_module.legacy_recall

    def recording_recall(store, query, **kwargs):
        calls.append(query)
        return real_recall(store, query, **kwargs)

    monkeypatch.setattr(planner_module, "legacy_recall", recording_recall)
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=fake_llm)
    result = planner.chain_execute("How did Eli move?", reinforce_on_access=False)

    assert calls[0] == "Where did Eli live first?"
    assert "You already learned: Newport" in calls[1]
    assert result.explanation.startswith("Reasoning steps: 1)")
    assert "2) Where next? -> Irvine" in result.to_context_string()
    assert all(len(result.hyperedges) <= 16 for _ in [0])


def test_chain_keeps_unknown_step_and_allows_final_override():
    mesh, *_ = _memory_mesh()

    def fake_llm(prompt: str):
        if prompt.startswith("Classify"):
            return {"question_class": "multi_hop", "entities": [],
                    "time_constraints": [], "question_kind": "causal"}
        if prompt.startswith("Decompose"):
            return {"sub_questions": ["Unknown first step?", "Where did Eli live?"]}
        return {"answer": "I don't know." if "Unknown first" in prompt else "Irvine"}

    result = QueryPlanner(mesh.store, embed=mesh.embed, llm=fake_llm).chain_execute(
        "Why did this happen?", reinforce_on_access=False
    )
    assert "I don't know." in result.explanation
    assert "may override" in result.explanation


def test_chain_with_no_subquestions_falls_back_to_plain_v2():
    mesh, *_ = _memory_mesh()

    def fake_llm(prompt: str):
        if prompt.startswith("Classify"):
            return {"question_class": "multi_hop", "entities": [],
                    "time_constraints": [], "question_kind": "fact"}
        return {"sub_questions": []}

    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=fake_llm)
    planner.decompose = lambda question, plan: QueryPlan(  # type: ignore[method-assign]
        "multi_hop", question_kind="fact", sub_questions=()
    )
    result = planner.chain_execute("Opaque memory query", reinforce_on_access=False)
    assert result.plan.question_class == "multi_hop"
    assert result.plan.sub_questions == ()
    assert result.explanation == ""
