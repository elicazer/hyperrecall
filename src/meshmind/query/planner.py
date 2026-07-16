"""Gemini-backed query planning and hypergraph-aware post-processing.

This module is deliberately separate from :mod:`meshmind.retrieval.query`.
Callers opt in with ``mesh.recall(question, plan="v2")``; the established
retrieval behaviour is otherwise byte-for-byte unchanged.

The planner has four phases: classify, select a retrieval strategy, resolve
supersession/contradiction, and assemble answer-ready edge records.  Gemini is
used only for classification and multi-hop decomposition.  An injectable LLM
function makes the reasoning deterministic in tests, while a conservative
heuristic fallback keeps local/offline use functional.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Sequence

from ..decay import DecayFn, exponential_decay
from ..models import CONTRADICTS, SUPERSEDES, Hyperedge, HyperedgeMember, Node
from ..retrieval.query import ScoredNode, Subgraph, recall as legacy_recall
from ..storage.embeddings import EmbedFn, hash_embed
from ..storage.sqlite_store import SqliteStore

QuestionClass = Literal[
    "single_hop", "multi_hop", "temporal", "open_domain", "adversarial"
]
_CLASSES = {"single_hop", "multi_hop", "temporal", "open_domain", "adversarial"}
_OLD_BELIEF = re.compile(
    r"\b(used to (?:think|believe|say|like)|previously|formerly|old belief|at first|"
    r"how did .{0,80}(?:change|view evolve)|how .{0,80}(?:view|opinion|preference) changed)\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(when|before|after|during|while|until|since|latest|newest|first|last|"
    r"earlier|later|how long|what year|what month|what date)\b", re.I
)
_MULTIHOP = re.compile(
    r"\b(because|lead to|result(?:ed)? in|relationship between|through|and then|"
    r"how did .+ affect|why did)\b", re.I
)
_ADVERSARIAL = re.compile(
    r"\b(according to the internet|world record|capital of|current president|"
    r"stock price|weather today)\b", re.I
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "before", "after", "did", "do",
    "does", "during", "for", "from", "how", "in", "is", "it", "of", "on",
    "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "we", "they", "he", "she", "their", "his", "her",
    "summarize", "remember", "tell", "explain", "describe",
}


@dataclass(frozen=True)
class TimeConstraint:
    """A normalized temporal condition extracted from the question."""

    relation: Literal["before", "after", "during", "at", "latest", "earliest"]
    value: str = ""
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class QueryPlan:
    """Output of Stage 1, including any Stage 2 decomposition."""

    question_class: QuestionClass
    entities: tuple[str, ...] = ()
    time_constraints: tuple[TimeConstraint, ...] = ()
    question_kind: str = "fact"
    sub_questions: tuple[str, ...] = ()
    used_llm: bool = False


@dataclass(frozen=True)
class Participant:
    node_id: str
    role: str
    text: str
    kind: str


@dataclass(frozen=True)
class EdgeResult:
    """An answerer-facing hyperedge with materialized participants."""

    edge: Hyperedge
    participants: tuple[Participant, ...]
    timestamp: float
    provenance: dict[str, Any]
    annotations: tuple[str, ...] = ()


@dataclass
class PlannedRecall(Subgraph):
    """Backward-compatible ``Subgraph`` enriched with planner output."""

    plan: QueryPlan | None = None
    results: list[EdgeResult] = field(default_factory=list)

    def to_context_string(self) -> str:
        if not self.results:
            return super().to_context_string()
        lines: list[str] = []
        for result in self.results:
            stamp = _format_timestamp(result.timestamp)
            source = result.provenance.get("source_text")
            annotations = " ".join(f"[{x}]" for x in result.annotations)
            rendered = str(source).strip() if source else "; ".join(
                p.text for p in result.participants if p.role != "entity"
            )
            if not rendered:
                rendered = "; ".join(p.text for p in result.participants)
            relation = ", ".join(
                f"{p.role}={p.text}" for p in result.participants if p.role
            )
            prefix = " ".join(x for x in (annotations, f"[{stamp}]" if stamp else "") if x)
            lines.append(f"{prefix} {rendered}".strip())
            if relation:
                lines.append(f"  Relation [{result.edge.type}]: {relation}")
        return "\n".join(lines)


LLMCallable = Callable[[str], str | dict[str, Any]]


class GeminiPlannerClient:
    """Small ``google-generativeai`` adapter with exponential backoff."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini query planning")
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Install google-generativeai to use the v2 query planner"
            ) from exc
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)

    def __call__(self, prompt: str) -> str:
        for attempt in range(4):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config={"temperature": 0, "response_mime_type": "application/json"},
                )
                return response.text or "{}"
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (2**attempt))
        return "{}"  # pragma: no cover


class QueryPlanner:
    """Classify a question, retrieve by class, and resolve graph conflicts."""

    def __init__(
        self,
        store: SqliteStore,
        *,
        embed: EmbedFn = hash_embed,
        curve: DecayFn = exponential_decay,
        llm: LLMCallable | None = None,
        use_gemini: bool = True,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        self.store = store
        self.embed = embed
        self.curve = curve
        self.allow_heuristic_fallback = allow_heuristic_fallback
        if llm is not None:
            self.llm = llm
        elif use_gemini and os.environ.get("GEMINI_API_KEY"):
            try:
                self.llm = GeminiPlannerClient()
            except RuntimeError:
                if not allow_heuristic_fallback:
                    raise
                self.llm = None
        else:
            self.llm = None

    def classify(self, question: str) -> QueryPlan:
        """Stage 1: classify and extract routing fields with one Gemini call."""
        if self.llm is not None:
            prompt = (
                "Classify this memory-recall question. Return JSON only with keys: "
                "question_class (single_hop|multi_hop|temporal|open_domain|adversarial), "
                "entities (array of exact names), time_constraints (array of objects with "
                "relation before|after|during|at|latest|earliest and value), and "
                "question_kind (short snake_case string). Adversarial means the question "
                "likely asks for information outside personal memory.\nQuestion: " + question
            )
            try:
                raw = _as_json(self.llm(prompt))
                return _plan_from_json(raw, question, used_llm=True)
            except Exception:
                if not self.allow_heuristic_fallback:
                    raise
        return _heuristic_plan(question)

    def decompose(self, question: str, plan: QueryPlan) -> QueryPlan:
        """Stage 2: use Gemini to turn multi-hop questions into 2-3 joins."""
        if plan.question_class != "multi_hop":
            return plan
        sub_questions: tuple[str, ...] = ()
        if self.llm is not None:
            prompt = (
                "Decompose this multi-hop personal-memory question into 2 or 3 atomic "
                "retrieval questions whose answers can be joined. Preserve entity names. "
                "Return JSON only: {\"sub_questions\": [\"...\"]}.\nQuestion: " + question
            )
            try:
                raw = _as_json(self.llm(prompt))
                candidates = raw.get("sub_questions", [])
                sub_questions = tuple(str(x).strip() for x in candidates if str(x).strip())[:3]
            except Exception:
                if not self.allow_heuristic_fallback:
                    raise
        if len(sub_questions) < 2:
            sub_questions = _heuristic_decompose(question)
        return QueryPlan(
            question_class=plan.question_class,
            entities=plan.entities,
            time_constraints=plan.time_constraints,
            question_kind=plan.question_kind,
            sub_questions=sub_questions,
            used_llm=plan.used_llm,
        )

    def recall(
        self,
        question: str,
        *,
        budget_tokens: int | None = None,
        k_hops: int = 2,
        max_seeds: int = 5,
        prefer_newest: bool = True,
        reinforce_on_access: bool = True,
        sim_rerank: float = 0.0,
        moat: bool = False,
    ) -> PlannedRecall:
        plan = self.decompose(question, self.classify(question))
        kwargs = dict(
            embed=self.embed, budget_tokens=budget_tokens, k_hops=k_hops,
            max_seeds=max_seeds, prefer_newest=prefer_newest,
            reinforce_on_access=reinforce_on_access, sim_rerank=sim_rerank,
            curve=self.curve,
        )
        if plan.question_class == "single_hop" and plan.entities:
            subgraph = self._entity_recall(question, plan.entities, **kwargs)
        elif plan.question_class == "multi_hop":
            subgraph = self._multi_hop_recall(question, plan.sub_questions, **kwargs)
        else:
            subgraph = legacy_recall(self.store, question, **kwargs)

        keep_history = bool(_OLD_BELIEF.search(question))
        subgraph, annotations = self._post_filter(subgraph, keep_history=keep_history)
        if plan.question_class == "temporal":
            subgraph = self._temporal_filter(subgraph, plan.time_constraints)
        results = self._assemble(subgraph, annotations=annotations)
        if moat:
            for result in results:
                self.store.bump_hyperedge_strength(result.edge.id)
        return PlannedRecall(
            query=question, nodes=subgraph.nodes, hyperedges=subgraph.hyperedges,
            plan=plan, results=results,
        )

    def _entity_recall(self, question: str, entities: Sequence[str], **kwargs: Any) -> Subgraph:
        """Anchor seeds on exact participant/entity matches before expansion."""
        matched: set[str] = set()
        lowered = [e.casefold() for e in entities]
        for node in self.store.all_nodes(curve=self.curve):
            text = node.text.casefold()
            if any(entity == text or entity in text for entity in lowered):
                matched.add(node.id)
                for edge in self.store.edges_for_node(node.id):
                    matched.update(edge.node_ids)
        # Legacy recall provides stable scoring; union anchored nodes/edges into it.
        base = legacy_recall(self.store, question, **kwargs)
        by_id = {sn.node.id: sn for sn in base.nodes}
        for nid in matched:
            if nid not in by_id:
                node = self.store.get_node(nid, curve=self.curve)
                if node:
                    by_id[nid] = ScoredNode(node=node, score=0.65, hop=1)
        edges = {e.id: e for e in base.hyperedges}
        for nid in matched:
            for edge in self.store.edges_for_node(nid):
                edges[edge.id] = edge
        nodes = sorted(by_id.values(), key=lambda sn: sn.score, reverse=True)
        return Subgraph(question, nodes, list(edges.values()))

    def _multi_hop_recall(self, question: str, subs: Sequence[str], **kwargs: Any) -> Subgraph:
        """Retrieve each atomic question, then join on shared graph participants."""
        graphs = [legacy_recall(self.store, sub, **kwargs) for sub in subs]
        by_id: dict[str, ScoredNode] = {}
        edges: dict[str, Hyperedge] = {}
        for graph in graphs:
            for sn in graph.nodes:
                old = by_id.get(sn.node.id)
                if old is None or sn.score > old.score:
                    by_id[sn.node.id] = sn
            edges.update((edge.id, edge) for edge in graph.hyperedges)
        # Include bridge edges touching nodes retrieved by two or more branches.
        counts: dict[str, int] = {}
        for graph in graphs:
            for nid in set(graph.node_ids()):
                counts[nid] = counts.get(nid, 0) + 1
        for nid, count in counts.items():
            if count >= 2:
                edges.update((edge.id, edge) for edge in self.store.edges_for_node(nid))
        return Subgraph(question, sorted(by_id.values(), key=lambda x: x.score, reverse=True), list(edges.values()))

    def _post_filter(
        self, graph: Subgraph, *, keep_history: bool
    ) -> tuple[Subgraph, dict[str, list[str]]]:
        """Resolve conflict relations between the retrieved content edges."""
        relation_types = {
            SUPERSEDES, "Supersession", "Supersedes",
            CONTRADICTS, "Contradiction", "Contradicts",
        }
        content = [edge for edge in graph.hyperedges if edge.type not in relation_types]
        by_node: dict[str, list[Hyperedge]] = {}
        for edge in content:
            for node_id in edge.node_ids:
                by_node.setdefault(node_id, []).append(edge)
        annotations: dict[str, list[str]] = {}
        discard: set[str] = set()

        relations = _edges_of_types(self.store, *relation_types)
        for relation in relations:
            involved = {edge.id: edge for nid in relation.node_ids for edge in by_node.get(nid, [])}
            if len(involved) < 2:
                continue
            ordered = sorted(involved.values(), key=lambda edge: (_edge_time(edge), edge.id))
            old, new = ordered[0], ordered[-1]
            old_date = _format_date(_edge_time(old))
            if keep_history:
                annotations.setdefault(old.id, []).append("before")
                annotations.setdefault(new.id, []).append("after")
                continue
            discard.add(old.id)
            if relation.type in {SUPERSEDES, "Supersession", "Supersedes"}:
                topic = str(relation.metadata.get("note") or _edge_topic(new, self.store))
                annotations.setdefault(new.id, []).append(
                    f"supersedes previous statement of {topic} on {old_date}"
                )
            else:
                annotations.setdefault(new.id, []).append(
                    f"this contradicts an earlier statement on {old_date}, preferring recent"
                )

        edges = [edge for edge in content if edge.id not in discard]
        kept_ids = {nid for edge in edges for nid in edge.node_ids}
        nodes = [sn for sn in graph.nodes if sn.node.id in kept_ids]
        return Subgraph(graph.query, nodes, edges), annotations

    def _temporal_filter(
        self, graph: Subgraph, constraints: Sequence[TimeConstraint]
    ) -> Subgraph:
        if not constraints:
            return graph
        edges = graph.hyperedges
        for constraint in constraints:
            if constraint.relation == "before" and constraint.start is not None:
                edges = [e for e in edges if _edge_time(e) < constraint.start]
            elif constraint.relation == "after" and constraint.start is not None:
                edges = [e for e in edges if _edge_time(e) > constraint.start]
            elif constraint.relation in {"during", "at"} and constraint.start is not None:
                end = constraint.end if constraint.end is not None else constraint.start + 86400
                edges = [e for e in edges if constraint.start <= _edge_time(e) <= end]
            elif constraint.relation in {"latest", "earliest"} and edges:
                fn = max if constraint.relation == "latest" else min
                chosen = fn(edges, key=_edge_time)
                edges = [chosen]
        ids = {nid for edge in edges for nid in edge.node_ids}
        return Subgraph(graph.query, [sn for sn in graph.nodes if sn.node.id in ids], edges)

    def _assemble(
        self, graph: Subgraph, *, annotations: dict[str, list[str]] | None = None
    ) -> list[EdgeResult]:
        """Stage 4: materialize participants, timestamp, and provenance."""
        results: list[EdgeResult] = []
        node_scores = {sn.node.id: sn.score for sn in graph.nodes}
        for edge in sorted(
            graph.hyperedges,
            key=lambda item: (
                max((node_scores.get(nid, 0.0) for nid in item.node_ids), default=0.0),
                item.activation_weight,
                _edge_time(item),
            ),
            reverse=True,
        ):
            participants = []
            for member in edge.members:
                node = self.store.get_node(member.node_id, curve=self.curve)
                if node:
                    participants.append(Participant(node.id, member.role, node.text, node.kind))
            results.append(EdgeResult(
                edge=edge, participants=tuple(participants), timestamp=_edge_time(edge),
                provenance=dict(edge.provenance),
                annotations=tuple((annotations or {}).get(edge.id, ())),
            ))
        return results


def _heuristic_plan(question: str) -> QueryPlan:
    entities = tuple(dict.fromkeys(re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", question)))
    entities = tuple(x for x in entities if x.lower() not in _STOPWORDS)
    constraints = tuple(_extract_time_constraints(question))
    if _ADVERSARIAL.search(question):
        cls: QuestionClass = "adversarial"
    elif constraints or _TEMPORAL.search(question):
        cls = "temporal"
    elif _MULTIHOP.search(question) or question.count("?") > 1:
        cls = "multi_hop"
    elif entities:
        cls = "single_hop"
    else:
        cls = "open_domain"
    kind = "time" if cls == "temporal" else "why" if question.lower().startswith("why") else "fact"
    return QueryPlan(cls, entities, constraints, kind, used_llm=False)


def _plan_from_json(raw: dict[str, Any], question: str, *, used_llm: bool) -> QueryPlan:
    cls = str(raw.get("question_class", "")).lower()
    if cls not in _CLASSES:
        return _heuristic_plan(question)
    entities = tuple(str(x).strip() for x in raw.get("entities", []) if str(x).strip())
    constraints = []
    for item in raw.get("time_constraints", []):
        if not isinstance(item, dict) or item.get("relation") not in {
            "before", "after", "during", "at", "latest", "earliest"
        }:
            continue
        value = str(item.get("value", ""))
        start, end = _parse_date_range(value)
        constraints.append(TimeConstraint(item["relation"], value, start, end))
    return QueryPlan(cls, entities, tuple(constraints), str(raw.get("question_kind", "fact")), used_llm=used_llm)  # type: ignore[arg-type]


def _extract_time_constraints(question: str) -> Iterable[TimeConstraint]:
    lower = question.lower()
    if re.search(r"\b(latest|newest|last)\b", lower):
        yield TimeConstraint("latest")
    if re.search(r"\b(first|earliest)\b", lower):
        yield TimeConstraint("earliest")
    match = re.search(r"\b(before|after|during|in|on)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})", question, re.I)
    if match:
        relation = {"in": "during", "on": "at"}.get(match.group(1).lower(), match.group(1).lower())
        start, end = _parse_date_range(match.group(2))
        yield TimeConstraint(relation, match.group(2), start, end)  # type: ignore[arg-type]


def _parse_date_range(value: str) -> tuple[float | None, float | None]:
    value = value.strip().replace(",", "")
    formats = ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%Y")
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            start = dt.timestamp()
            if fmt == "%Y":
                end = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc).timestamp() - 1
            else:
                end = start + 86400 - 1
            return start, end
        except ValueError:
            pass
    return None, None


def _heuristic_decompose(question: str) -> tuple[str, ...]:
    pieces = re.split(r"\b(?:and then|because|after|before|and|which)\b", question, flags=re.I)
    cleaned = tuple(" ".join(x.strip(" ,?.").split()) + "?" for x in pieces if len(x.split()) >= 2)
    if len(cleaned) >= 2:
        return cleaned[:3]
    return (question, f"What facts connect the entities in: {question}")


def _as_json(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("planner response must be a JSON object")
    return parsed


def _edges_of_types(store: SqliteStore, *types: str) -> list[Hyperedge]:
    found: dict[str, Hyperedge] = {}
    for edge_type in types:
        found.update((e.id, e) for e in store.edges_of_type(edge_type))
    return list(found.values())


def _node_time(store: SqliteStore, node_id: str) -> float:
    node = store.get_node(node_id)
    return node.created_at if node else 0.0


def _edge_time(edge: Hyperedge) -> float:
    for mapping in (edge.provenance, edge.metadata):
        for key in ("timestamp", "created_at", "date", "date_time"):
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                parsed, _ = _parse_date_range(value[:10])
                if parsed is not None:
                    return parsed
    return edge.created_at


def _format_timestamp(timestamp: float) -> str:
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_date(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def _edge_topic(edge: Hyperedge, store: SqliteStore) -> str:
    for member in edge.members:
        if member.role not in {"person", "speaker", "subject", "entity"}:
            node = store.get_node(member.node_id)
            if node:
                return node.text
    return edge.type.lower()


def _demo() -> int:
    from ..mesh import Mesh

    mesh = Mesh(":memory:")
    old = Node("Eli used Python", kind="fact", created_at=1704067200)
    new = Node("Eli uses Rust", kind="fact", created_at=1735689600)
    eli = Node("Eli", kind="entity", created_at=1704067200)
    for node in (old, new, eli):
        mesh.add_node(node)
    mesh.add_hyperedge(Hyperedge(type="Experience", created_at=1704067200, members=[HyperedgeMember(eli.id, "person"), HyperedgeMember(old.id, "statement")]))
    mesh.add_hyperedge(Hyperedge(type="Experience", created_at=1735689600, members=[HyperedgeMember(eli.id, "person"), HyperedgeMember(new.id, "statement")]))
    mesh.contradict(old.id, new.id)
    mesh.supersede(old.id, new.id)
    examples = {
        "single_hop": "What does Eli use?",
        "multi_hop": "Why did Eli change tools because of his project?",
        "temporal": "What did Eli use before 2025?",
        "open_domain": "Summarize the remembered tools",
        "adversarial": "What is the current president's stock price?",
    }
    planner = QueryPlanner(mesh.store, embed=mesh.embed, llm=None, use_gemini=False)
    for expected, question in examples.items():
        result = planner.recall(question, reinforce_on_access=False)
        print(f"{expected:12} -> {result.plan.question_class:12} | {len(result.results)} edges")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run an offline five-class demo")
    args = parser.parse_args(argv)
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
