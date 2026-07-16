"""Extractor v2 — dense, coreference-aware, 3-pass ingestion.

The v0 extractor asked one LLM prompt per turn and emitted a *single* hyperedge.
On LoCoMo conv-26 that starves the hypergraph: it misses most facts and never
canonicalizes entities across turns, so "Caroline" in turn 3 and "Caroline" in
turn 200 become two disconnected nodes. Mem0/Zep beat us because their
extractors are *dense* (many facts per turn) and *canonicalized* (one entity ->
one id, forever).

v2 fixes both with a three-pass pipeline, all backed by **Gemini 2.5 Pro** via
Google's ``google.genai`` SDK (``GEMINI_API_KEY``). No Bedrock, no AWS.

Pass 1 — Entity extraction
    One Gemini call returns *all* entities in a turn, each typed
    (Person/Project/Decision/Event/Place/Time/Artifact/Belief/Preference) with a
    short canonical name and a one-line description.

Pass 2 — Relation extraction
    Given those entities + the turn text, a second Gemini call emits a list of
    typed N-ary hyperedges. Each hyperedge has a type from a fixed vocabulary,
    participants with roles, an optional timestamp, and a one-line summary.

Pass 3 — Canonicalization
    Each newly-extracted entity is checked against entities already in the mesh
    (normalized-name match first, embedding similarity second). A match reuses
    the existing node id — coreference across turns. Every merge decision is
    logged so we can debug the graph.

One turn typically yields 3-10 hyperedges instead of 1.

Run ``python -m meshmind.ingest.extractor_v2 --demo`` for a live demo. In
``mock_mode`` (or with no API key) a deterministic, network-free heuristic is
used so tests and offline demos never touch the network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..models import Hyperedge, HyperedgeMember, Node
from ..storage import embeddings as emb

if TYPE_CHECKING:  # avoid a hard import cycle at runtime
    from ..mesh import Mesh
    from ..storage.sqlite_store import SqliteStore

# --------------------------------------------------------------------------- #
# Vocabularies.
# --------------------------------------------------------------------------- #

#: Entity types Pass 1 is allowed to emit. The node ``kind`` is the lowercased
#: form of these.
ENTITY_TYPES = (
    "Person",
    "Project",
    "Decision",
    "Event",
    "Place",
    "Time",
    "Artifact",
    "Belief",
    "Preference",
)

#: Hyperedge types Pass 2 is allowed to emit (fixed vocabulary).
HYPEREDGE_VOCAB = (
    "Decision",
    "Preference",
    "Action",
    "Statement",
    "Observation",
    "Question",
    "Contradiction",
    "Supersession",
    "Ownership",
    "Location",
    "TemporalOrder",
)

DEFAULT_MODEL = "gemini-2.5-pro"


# --------------------------------------------------------------------------- #
# Validated result schema (plain dataclasses, no pydantic).
# --------------------------------------------------------------------------- #


@dataclass
class EntityV2:
    name: str
    type: str = "Artifact"
    description: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "EntityV2 | None":
        if not isinstance(d, dict):
            return None
        name = str(d.get("name") or "").strip()
        if not name:
            return None
        etype = str(d.get("type") or "").strip()
        if etype not in ENTITY_TYPES:
            etype = "Artifact"
        return cls(name=name, type=etype, description=str(d.get("description") or "").strip())


@dataclass
class RelationParticipant:
    entity: str
    role: str = "subject"

    @classmethod
    def from_dict(cls, d: Any) -> "RelationParticipant | None":
        if not isinstance(d, dict):
            return None
        entity = str(d.get("entity") or d.get("entity_name") or d.get("name") or "").strip()
        if not entity:
            return None
        role = str(d.get("role") or "subject").strip() or "subject"
        return cls(entity=entity, role=role)


@dataclass
class HyperedgeV2:
    type: str
    summary: str
    participants: list[RelationParticipant] = field(default_factory=list)
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "HyperedgeV2 | None":
        if not isinstance(d, dict):
            return None
        etype = str(d.get("type") or "").strip()
        if etype not in HYPEREDGE_VOCAB:
            etype = "Statement"
        summary = str(d.get("summary") or "").strip()
        parts = [RelationParticipant.from_dict(p) for p in (d.get("participants") or [])]
        parts = [p for p in parts if p is not None]
        ts = d.get("timestamp")
        ts = str(ts).strip() if isinstance(ts, str) and ts.strip() else None
        if not summary and not parts:
            return None
        return cls(type=etype, summary=summary, participants=parts, timestamp=ts)


@dataclass
class TurnExtraction:
    """The validated, backend-agnostic result of extracting one turn, plus the
    concrete node ids assigned during canonicalization (filled in by
    :meth:`ExtractorV2.ingest`)."""

    source_text: str
    entities: list[EntityV2] = field(default_factory=list)
    hyperedges: list[HyperedgeV2] = field(default_factory=list)
    entity_ids: dict[str, str] = field(default_factory=dict)  # entity name -> node id
    merges: list[dict[str, Any]] = field(default_factory=list)  # coref decisions
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Prompts.
# --------------------------------------------------------------------------- #

_PASS1_SYSTEM = (
    "You are MeshMind's entity extractor. Given ONE conversation turn, list "
    "EVERY distinct entity it mentions or implies. Use the speaker's real name "
    "(never 'I', 'me', 'you', or 'user'). Each entity has:\n"
    "- name: a short canonical noun phrase (e.g. 'Caroline', 'border collie', "
    "'trip to Japan').\n"
    "- type: EXACTLY one of " + ", ".join(ENTITY_TYPES) + ".\n"
    "- description: one short line describing it in context.\n"
    "Be dense: a single turn usually mentions 2-6 entities. Split compound "
    "ideas into atomic entities. Prefer 'Person' for named people, 'Preference' "
    "for likes/dislikes, 'Decision' for choices, 'Belief' for opinions/claims, "
    "'Time' for stated dates/periods."
)

_PASS2_SYSTEM = (
    "You are MeshMind's relation extractor. Given a conversation turn and the "
    "entities already pulled from it, emit the typed N-ary relations "
    "(hyperedges) that bind them. Be dense: 3-10 hyperedges per substantive "
    "turn. Each hyperedge has:\n"
    "- type: EXACTLY one of " + ", ".join(HYPEREDGE_VOCAB) + ".\n"
    "- participants: each an {entity, role} where entity is one of the given "
    "entity names and role describes how it participates "
    "(subject, object, instrument, topic, location, time, preferred, "
    "less_preferred, cause, effect, owner, owned, ...).\n"
    "- summary: one declarative sentence stating the fact, self-contained "
    "(resolve pronouns to names).\n"
    "- timestamp: ISO-8601 if the turn states/implies a specific date, else null.\n"
    "Do NOT invent entities that were not provided. Pick the single best type "
    "per relation; use 'Statement' if nothing else fits."
)


def _pass1_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "type", "description"],
                },
            }
        },
        "required": ["entities"],
    }


def _pass2_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hyperedges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(HYPEREDGE_VOCAB)},
                        "summary": {"type": "string"},
                        "timestamp": {"type": "string", "nullable": True},
                        "participants": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "role": {"type": "string"},
                                },
                                "required": ["entity", "role"],
                            },
                        },
                    },
                    "required": ["type", "summary", "participants"],
                },
            }
        },
        "required": ["hyperedges"],
    }


# --------------------------------------------------------------------------- #
# Canonicalization helpers.
# --------------------------------------------------------------------------- #

_POSSESSIVE = re.compile(r"'s\b")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name: str) -> str:
    """Canonical key for coreference: lowercased, de-possessived, punctuation
    stripped, whitespace collapsed.

    >>> normalize_name("Caroline's")
    'caroline'
    >>> normalize_name("  The  Border-Collie ")
    'the border collie'
    """
    s = name.strip().lower()
    s = _POSSESSIVE.sub("", s)
    s = _NONWORD.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class _EntityRecord:
    id: str
    name: str
    type: str
    vec: np.ndarray | None


def _load_entity_index(store: "SqliteStore") -> list[_EntityRecord]:
    """Load every entity node already in the mesh (with its embedding)."""
    kinds = tuple(t.lower() for t in ENTITY_TYPES)
    placeholders = ",".join("?" for _ in kinds)
    rows = store._conn.execute(
        f"""SELECT n.id, n.text, n.kind, e.vector
            FROM nodes n LEFT JOIN embeddings e ON e.node_id = n.id
            WHERE n.kind IN ({placeholders})""",
        kinds,
    ).fetchall()
    out: list[_EntityRecord] = []
    for r in rows:
        vec = emb.from_blob(r["vector"]) if r["vector"] is not None else None
        out.append(_EntityRecord(id=r["id"], name=r["text"], type=r["kind"], vec=vec))
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Deterministic offline fallback (mock mode / no key / tests).
# --------------------------------------------------------------------------- #

_STOP = {
    "The", "A", "An", "On", "In", "At", "It", "They", "We", "I", "He", "She",
    "You", "This", "That", "My", "Our", "Your", "Their", "And", "But", "So",
}


def _heuristic_extract(text: str, speaker: str | None) -> TurnExtraction:
    """Network-free extraction: capitalized tokens become Person/Artifact
    entities; each is bound to the speaker by a Statement edge, plus one
    Observation over everything. Deterministic -> good for tests & demos."""
    src = text.strip()
    names: list[str] = []
    if speaker:
        names.append(speaker)
    for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", src):
        if tok in _STOP or tok in names:
            continue
        names.append(tok)
    # The offline stub can't type semantically, so it treats every proper-name
    # token as a Person. That keeps cross-turn coreference clean in the demo
    # (real Gemini does proper typing in Pass 1).
    entities = [
        EntityV2(name=n, type="Person", description=f"mentioned in: {src[:60]}") for n in names
    ]

    edges: list[HyperedgeV2] = []
    subject = speaker or (names[0] if names else None)
    for n in names:
        if n == subject:
            continue
        edges.append(
            HyperedgeV2(
                type="Statement",
                summary=f"{subject or 'Someone'} mentioned {n}.",
                participants=[
                    RelationParticipant(entity=subject or n, role="subject"),
                    RelationParticipant(entity=n, role="topic"),
                ],
            )
        )
    if len(names) >= 2:
        edges.append(
            HyperedgeV2(
                type="Observation",
                summary=src[:120],
                participants=[RelationParticipant(entity=n, role="topic") for n in names],
            )
        )
    if not edges and subject:
        edges.append(
            HyperedgeV2(
                type="Statement",
                summary=src[:120] or f"{subject} spoke.",
                participants=[RelationParticipant(entity=subject, role="subject")],
            )
        )
    return TurnExtraction(
        source_text=src, entities=entities, hyperedges=edges, raw={"backend": "heuristic"}
    )


# --------------------------------------------------------------------------- #
# The extractor.
# --------------------------------------------------------------------------- #


class ExtractorV2:
    """Three-pass, dense, coreference-aware extractor backed by Gemini 2.5 Pro.

    Parameters
    ----------
    model:
        Gemini model id (default ``gemini-2.5-pro``).
    mock_mode:
        When True, no network call is made; a deterministic heuristic parse is
        returned. Used by tests and offline demos.
    sim_threshold:
        Cosine-similarity threshold above which two same-type entities are
        merged during Pass 3 (only used when a real, semantic embedder is
        configured on the mesh).
    client:
        Inject a pre-built ``google.genai`` client (mostly for tests).
    logger:
        Optional callable for merge/decision logs (defaults to stderr).
    """

    backend = "v2"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        mock_mode: bool = False,
        sim_threshold: float = 0.86,
        max_retries: int = 4,
        backoff_base: float = 0.75,
        temperature: float = 0.2,
        client: Any = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.mock_mode = mock_mode
        self.sim_threshold = sim_threshold
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.temperature = temperature
        self._client = client
        self._log = logger or (lambda m: print(m, file=sys.stderr, flush=True))

    # -- gemini client ------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            import os

            from google import genai  # lazy: not needed in mock_mode

            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def _generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """One structured Gemini call with exponential-backoff retries."""
        from google.genai import types

        client = self._get_client()
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=self.temperature,
        )
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
                    config=cfg,
                )
                return json.loads(resp.text)
            except BaseException as exc:  # noqa: BLE001 - retried/reraised below
                last = exc
                if attempt >= self.max_retries:
                    raise
                self._log(f"  gemini retry {attempt + 1}/{self.max_retries}: {exc}")
                time.sleep(self.backoff_base * (2 ** attempt))
        assert last is not None
        raise last

    # -- passes 1 & 2 -------------------------------------------------------
    def extract_turn(
        self, text: str, *, speaker: str | None = None, when: str | None = None
    ) -> TurnExtraction:
        """Run Pass 1 (entities) then Pass 2 (relations). No persistence."""
        src = text.strip()
        if not src:
            return TurnExtraction(source_text="", raw={"empty": True})
        if self.mock_mode:
            return _heuristic_extract(src, speaker)

        header = f"[{speaker or 'speaker'}"
        header += f" at {when}]" if when else "]"
        turn_block = f"{header} {src}"

        # Pass 1 — entities
        p1 = self._generate(_PASS1_SYSTEM, f"Turn:\n{turn_block}", _pass1_schema())
        entities = [EntityV2.from_dict(e) for e in (p1.get("entities") or [])]
        entities = [e for e in entities if e is not None]

        # Pass 2 — relations, conditioned on the entities we just found
        entity_lines = "\n".join(f"- {e.name} ({e.type}): {e.description}" for e in entities)
        p2_user = (
            f"Turn:\n{turn_block}\n\nEntities extracted from this turn:\n"
            f"{entity_lines or '(none)'}"
        )
        p2 = self._generate(_PASS2_SYSTEM, p2_user, _pass2_schema())
        edges = [HyperedgeV2.from_dict(h) for h in (p2.get("hyperedges") or [])]
        edges = [h for h in edges if h is not None]

        return TurnExtraction(
            source_text=src, entities=entities, hyperedges=edges, raw={"pass1": p1, "pass2": p2}
        )

    # -- pass 3 + persistence ----------------------------------------------
    def ingest(
        self,
        mesh: "Mesh",
        text: str,
        *,
        speaker: str | None = None,
        when: str | None = None,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
    ) -> TurnExtraction:
        """Full pipeline for one turn: extract (1+2), canonicalize (3), persist.

        Returns the :class:`TurnExtraction` with ``entity_ids``/``merges``/
        ``edge_ids`` filled in. All nodes and hyperedges are already committed
        to ``mesh`` and immediately recallable.
        """
        context = context or {}
        speaker = speaker or context.get("speaker")
        when = when or context.get("session_dt") or context.get("when")

        te = self.extract_turn(text, speaker=speaker, when=when)
        if not te.source_text:
            return te

        # Pass 3 — canonicalize each entity against what's already in the mesh.
        index = _load_entity_index(mesh.store)
        by_key: dict[tuple[str, str], _EntityRecord] = {
            (r.type, normalize_name(r.name)): r for r in index
        }
        # Same normalized name under *any* type. Pass 1 sometimes retypes the
        # same proper noun across turns (e.g. Luna typed Artifact then Person);
        # for LoCoMo-style dialogue an exact name match is almost always the
        # same entity, so we merge cross-type rather than fragment the graph.
        by_name: dict[str, _EntityRecord] = {}
        for r in index:
            by_name.setdefault(normalize_name(r.name), r)
        prov = dict(provenance or {})
        prov.setdefault("extractor", "v2")
        prov.setdefault("source_text", te.source_text)

        for ent in te.entities:
            kind = ent.type.lower()
            norm = normalize_name(ent.name)
            key = (kind, norm)
            match = by_key.get(key)
            method = "name"
            # Exact name under a different type (Pass 1 retyped the same noun).
            if match is None and norm in by_name:
                match = by_name[norm]
                method = f"name-xtype({match.type})"
            # Embedding fallback: only meaningful with a real semantic embedder,
            # but harmless (and logged) with the default hash embedder.
            if match is None:
                match, score = self._best_embed_match(mesh, ent, index)
                method = f"embed:{score:.2f}" if match else "new"
            if match is not None:
                te.entity_ids[ent.name] = match.id
                te.merges.append(
                    {"entity": ent.name, "type": ent.type, "merged_into": match.id,
                     "canonical": match.name, "via": method}
                )
                self._log(
                    f"    coref: '{ent.name}' ({ent.type}) -> {match.id} "
                    f"('{match.name}') via {method}"
                )
            else:
                node = Node(
                    text=ent.name,
                    kind=kind,
                    confidence=confidence,
                    metadata={"entity_type": ent.type, "description": ent.description},
                )
                mesh.add_node(node)
                te.entity_ids[ent.name] = node.id
                te.node_ids.append(node.id)
                rec = _EntityRecord(id=node.id, name=ent.name, type=kind, vec=mesh.embed(ent.name))
                index.append(rec)
                by_key[key] = rec
                by_name.setdefault(norm, rec)
                self._log(f"    new entity: '{ent.name}' ({ent.type}) -> {node.id}")

        # One node holding the raw turn text, so retrieval can surface it.
        turn_node = Node(
            text=te.source_text,
            kind="fact",
            confidence=confidence,
            metadata={**context, **({"when": when} if when else {})},
        )
        mesh.add_node(turn_node)
        te.node_ids.append(turn_node.id)

        # Materialize hyperedges. Each edge gets a summary node + its entity
        # participants + the raw-turn node (role 'source'), guaranteeing arity>=2.
        for h in te.hyperedges:
            summary_text = h.summary or te.source_text[:120]
            summary_node = Node(
                text=summary_text,
                kind="statement",
                confidence=confidence,
                metadata={"edge_type": h.type, **({"timestamp": h.timestamp} if h.timestamp else {})},
            )
            mesh.add_node(summary_node)
            te.node_ids.append(summary_node.id)

            members = [
                HyperedgeMember(summary_node.id, role="summary", weight=1.0),
                HyperedgeMember(turn_node.id, role="source", weight=0.3),
            ]
            # Dedup members by node id: distinct participant names can canonicalize
            # to the same entity node (a hyperedge can't list a node twice — the
            # store enforces UNIQUE(hyperedge_id, node_id)). Keep the first role seen.
            seen_ids = {summary_node.id, turn_node.id}
            for p in h.participants:
                nid = te.entity_ids.get(p.entity)
                if nid is None:
                    # Participant the entity pass missed: create it on the fly.
                    node = Node(text=p.entity, kind="entity", confidence=confidence)
                    mesh.add_node(node)
                    te.entity_ids[p.entity] = node.id
                    te.node_ids.append(node.id)
                    nid = node.id
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                members.append(HyperedgeMember(nid, role=p.role, weight=0.8))

            edge = Hyperedge(
                type=h.type,
                members=members,
                confidence=confidence,
                provenance=prov,
                metadata={
                    "summary": summary_text,
                    **({"timestamp": h.timestamp} if h.timestamp else {}),
                },
            )
            mesh.add_hyperedge(edge)
            te.edge_ids.append(edge.id)

        return te

    def _best_embed_match(
        self, mesh: "Mesh", ent: EntityV2, index: list[_EntityRecord]
    ) -> tuple[_EntityRecord | None, float]:
        """Best same-type entity in the mesh by embedding cosine, or (None, 0)."""
        kind = ent.type.lower()
        probe = mesh.embed(ent.name)
        best: _EntityRecord | None = None
        best_score = 0.0
        for rec in index:
            if rec.type != kind or rec.vec is None:
                continue
            score = _cosine(probe, rec.vec)
            if score > best_score:
                best, best_score = rec, score
        if best is not None and best_score >= self.sim_threshold:
            return best, best_score
        return None, best_score


# --------------------------------------------------------------------------- #
# Demo CLI.
# --------------------------------------------------------------------------- #

_DEMO_TURNS = [
    ("Caroline", "2023-05-08", "I finally adopted a border collie last weekend! Her name is Luna."),
    ("Melanie", "2023-05-08", "That's wonderful, Caroline! Is Luna keeping you busy?"),
    ("Caroline", "2023-06-20", "Luna and I decided to start agility training in July. I love it."),
    ("Melanie", "2023-06-20", "Nice. I'm thinking of moving to Portland for the new studio job."),
]


def _run_demo(mock: bool) -> int:
    from ..mesh import Mesh

    mesh = Mesh(":memory:")
    ext = ExtractorV2(mock_mode=mock)
    print(f"# Extractor v2 demo (mock={mock}, model={ext.model})\n")
    total_edges = 0
    for speaker, when, text in _DEMO_TURNS:
        print(f"── [{speaker} @ {when}] {text}")
        te = ext.ingest(mesh, text, speaker=speaker, when=when)
        total_edges += len(te.edge_ids)
        print(f"   entities={len(te.entities)}  hyperedges={len(te.edge_ids)}  "
              f"merges={len(te.merges)}")
        for h in te.hyperedges:
            print(f"     · {h.type}: {h.summary}")
        print()
    print(f"# stats: {mesh.stats()}   total hyperedges={total_edges}")
    # Show coreference: how many distinct ids does 'Caroline'/'Luna' have?
    for name in ("Caroline", "Luna"):
        ids = {
            r["id"]
            for r in mesh.store._conn.execute(
                "SELECT id FROM nodes WHERE lower(text)=?", (name.lower(),)
            ).fetchall()
        }
        print(f"# '{name}' resolves to {len(ids)} node id(s): {sorted(ids)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MeshMind extractor v2 demo")
    ap.add_argument("--demo", action="store_true", help="run the built-in demo")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="force offline heuristic (default: real Gemini if GEMINI_API_KEY set)",
    )
    args = ap.parse_args(argv)
    import os

    mock = args.mock or not os.environ.get("GEMINI_API_KEY")
    if args.demo:
        return _run_demo(mock)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
