"""Run a structural contradiction benchmark against MeshMind and vector top-k."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from meshmind import Hyperedge, HyperedgeMember, Mesh, Node  # noqa: E402

HERE = Path(__file__).resolve().parent


def timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def vector_top_k(question: str, turns: list[dict[str, str]], k: int = 2) -> list[dict[str, str]]:
    """A transparent bag-of-words vector analogue with cosine-like overlap."""
    query = tokens(question)
    return sorted(
        turns,
        key=lambda turn: (len(query & tokens(turn["text"])) / max(1, len(query | tokens(turn["text"]))), turn["date"]),
        reverse=True,
    )[:k]


def build_mesh(cases: list[dict[str, str]]) -> tuple[Mesh, list[dict[str, str]]]:
    mesh = Mesh(":memory:")
    turns: list[dict[str, str]] = []
    for case in cases:
        person = mesh.add_node(Node(case["person"], kind="entity", created_at=timestamp(case["old_date"])))
        claims = []
        for phase in ("old", "new"):
            date = case[f"{phase}_date"]
            claim = mesh.add_node(Node(case[phase], kind="fact", created_at=timestamp(date)))
            edge = mesh.add_hyperedge(Hyperedge(
                type="Statement", created_at=timestamp(date),
                provenance={"source_text": case[phase], "timestamp": date},
                members=[HyperedgeMember(person.id, "person"), HyperedgeMember(claim.id, "statement")],
            ))
            claims.append((claim, edge))
            turns.append({"person": case["person"], "phase": phase, "date": date, "text": case[phase]})
        if case["relation"] == "Supersession":
            mesh.supersede(claims[0][0].id, claims[1][0].id, note=case["topic"])
        else:
            mesh.contradict(claims[0][0].id, claims[1][0].id, note=case["topic"])
    return mesh, turns


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())
    mesh, turns = build_mesh(cases)
    rows = []
    for index, case in enumerate(cases):
        history = index >= 5
        question = (
            f"How did {case['person']}'s view change about {case['topic']}?"
            if history else f"What is {case['person']}'s current {case['topic']}?"
        )
        result = mesh.recall(question, plan="v2-moat", reinforce_on_access=False)
        moat_context = result.to_context_string()
        vector_turns = vector_top_k(question, turns)
        vector_context = "\n".join(f"[{turn['date']}] {turn['text']}" for turn in vector_turns)
        if history:
            moat_ok = case["old"] in moat_context and case["new"] in moat_context and "[before]" in moat_context and "[after]" in moat_context
            vector_ok = case["old"] in vector_context and case["new"] in vector_context and "[before]" in vector_context and "[after]" in vector_context
        else:
            moat_ok = case["new"] in moat_context and case["old"] not in moat_context
            vector_ok = case["new"] in vector_context and case["old"] not in vector_context
        rows.append({
            "question": question, "mode": "history" if history else "current",
            "mesh_pass": moat_ok, "vector_pass": vector_ok,
            "mesh_context": moat_context, "vector_context": vector_context,
        })
    output = {
        "n_turns": len(turns), "n_questions": len(rows),
        "meshmind_v2_moat": {"correct": sum(row["mesh_pass"] for row in rows), "total": len(rows)},
        "vector_rag": {"correct": sum(row["vector_pass"] for row in rows), "total": len(rows)},
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
