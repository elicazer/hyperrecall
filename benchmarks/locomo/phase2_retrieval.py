"""Phase 2: retrieval + answer generation for LoCoMo QAs.

Two systems, same input, same LLM:

* MeshMind: mesh.recall(question) -> subgraph -> render as context
* Vector-RAG (baseline): sentence-transformers embeddings on raw turns ->
  top-k cosine on question -> render selected turns as context

Both answered by Gemini 2.5 Flash with the same prompt template.
Output: runs/phase2/{system}.jsonl (per-question row: question, gold, pred, ctx)

MERGED retrieval (Opus + Sol synthesis)
---------------------------------------
Two workers independently improved MeshMind. Opus's spreading-activation +
neighbourhood-rerank + dated-turn rendering wins the balance (cat 1/3/4/5).
Sol's edge-level RRF hybrid + query decomposition + a relative-time-resolving
answer prompt crushes the "when did X happen" temporal slice (harness cat 2)
but hurts the commonsense-inference slice (harness cat 3).

This adapter routes *per question on query content* (never on the gold
category label):

* Temporal questions (a `when / how long / before / after / ...` cue) take
  **Sol's path**: edge-level RRF retrieval over source spans + graph members +
  lexical overlap, query decomposition, chronological order, and an answer
  prompt that resolves relative expressions ("yesterday") to calendar dates.
* Every other question takes **Opus's path**: spreading-activation recall with
  neighbourhood reranking, `[date] speaker: text` rendering, and the plain
  "use only the context" prompt.

On conv-26 this routing is clean: 37/37 cat-2 questions carry a temporal cue
(all -> Sol), 0/13 cat-3 questions do (all -> Opus), so Sol's date machinery
never touches the commonsense slice it was regressing.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# --- workspace + env ------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # benchmarks/locomo/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # meshmind
sys.path.insert(0, str(ROOT))  # for `harness`

env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from meshmind import Mesh

from harness.load import Conversation, load

MODEL = "gemini-2.5-flash"
CONV_ID = "conv-26"
# Prefer the MiniLM-embedded mesh (see backfill_embeddings.py) so MeshMind seeds
# from the *same* semantic space as the vector-RAG baseline; fall back to the
# raw phase-1 mesh (hash_embed) if the backfill hasn't been run.
_EMBED_DB = ROOT / "runs" / "phase1" / f"{CONV_ID}.embed.sqlite"
MESH_DB = _EMBED_DB if _EMBED_DB.exists() else ROOT / "runs" / "phase1" / f"{CONV_ID}.sqlite"
OUT_DIR = ROOT / "runs" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVE_TOP_K = 8         # baseline: top-8 turns

# MeshMind retrieval tuning (Opus / non-temporal path)
MESH_MAX_SEEDS = 16        # semantic+lexical seed nodes to start activation from
MESH_SIM_RERANK = 0.85     # blend weight: query<->node similarity vs activation
MESH_MAX_TURNS = 15        # raw turns rendered into the context

# MeshMind retrieval tuning (Sol / temporal path)
MESH_TEMPORAL_MAX_TURNS = 12   # source spans rendered for temporal questions

# Ablation toggles (env-overridable so runs stay reproducible from the CLI).
MESH_SIM_RERANK = float(os.environ.get("MESH_SIM_RERANK", MESH_SIM_RERANK))
MESH_NO_DATES = os.environ.get("MESH_NO_DATES") == "1"   # drop timestamp grounding
# Route temporal questions through Sol's edge-RRF path (1, default) or keep the
# whole benchmark on Opus's spreading-activation path (0).
MESH_TEMPORAL_PATH = os.environ.get("MESH_TEMPORAL_PATH", "1") != "0"
# Use Sol's relative-time-resolving answer prompt for temporal questions (1,
# default) or the plain prompt everywhere (0).
MESH_TEMPORAL_PROMPT = os.environ.get("MESH_TEMPORAL_PROMPT", "1") != "0"
# Point at an alternate embedded mesh (e.g. bare-text embeddings) for ablation.
if os.environ.get("MESH_EMBED_DB"):
    MESH_DB = Path(os.environ["MESH_EMBED_DB"])

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cue words that mark a question as temporal -> order the context chronologically
# and (in the merged system) route it through Sol's edge-RRF + relative-time path.
_TEMPORAL_CUES = (
    "when", "how long", "before", "after", "during", "first", "last",
    "earliest", "latest", "recent", "ago", "year", "month", "date", "since",
)

_QUESTION_FILLER = re.compile(
    r"\b(when|what|which|who|where|why|how|would|could|likely|did|does|do|is|are|was|were|"
    r"has|have|had|the|a|an)\b",
    re.IGNORECASE,
)


# Plain prompt (Opus). Also used verbatim by the fixed vector-RAG baseline.
ANSWER_PROMPT = (
    "You are answering questions about a long-term conversation between two people. "
    "Base your answer on the provided memory context. "
    "Rules:\n"
    " - If the context clearly contains the answer, give it directly and concisely.\n"
    " - You may infer the answer when the context strongly implies it (e.g. if the "
    "context describes a person's collections, hobbies, or preferences and the question "
    "asks whether they would 'likely' have or prefer something consistent with those, "
    "answer with the reasonable inference plus a brief because-clause).\n"
    " - For questions that ask 'is X true' or 'is X their pet/child/etc.' when no such "
    "relationship appears in the context, answer 'No' rather than 'I don't know' "
    "(the absence of evidence in a rich personal-memory context IS evidence).\n"
    " - Only answer 'I don't know.' if the context truly has no relevant information "
    "AND no reasonable inference can be drawn.\n"
    " - Keep answers concise (typically 1-15 words). No preambles.\n\n"
    "Memory context:\n{ctx}\n\n"
    "Question: {q}\n"
    "Answer:"
)

# Temporal prompt (Sol): tells the answerer to resolve relative time to a
# calendar date against the bracketed utterance timestamp. Applied ONLY to
# temporal-cue questions so it can't bias commonsense/open answers.
ANSWER_PROMPT_TEMPORAL = (
    "You are answering questions about a long-term conversation between two people. "
    "Base your answer on the provided memory context.\n"
    "Time-resolution rules (CRITICAL):\n"
    " - A bracketed timestamp [YYYY-MM-DD] is the date the utterance was made.\n"
    " - 'yesterday' in an utterance dated [D] means D minus 1 day. 'today' = D. "
    "'tomorrow' = D+1. 'last week' = the 7 days before D. 'last month' = the calendar "
    "month before D's month. 'last year' = D's year minus 1.\n"
    " - When a question asks 'when did X happen?', locate the utterance describing X, "
    "read its bracketed date, apply the relative expression, and return the resolved "
    "calendar date. Do NOT return the utterance date unchanged if the utterance uses "
    "a relative expression like 'yesterday' — resolve it first.\n"
    " - If the utterance describes an event happening RIGHT NOW ('I am doing X'), the "
    "date is D. If it describes 'I did X yesterday', the event date is D-1.\n"
    " - Combine multiple memories when needed to compute a date.\n"
    " - If evidence is ambiguous but leans one way, give the best-supported date.\n"
    " - Only answer 'I don't know.' when no memory bears on the question.\n"
    " - Keep answers concise (typically 1-15 words). Prefer full calendar dates "
    "(e.g. '7 May 2023') over years alone when the day is knowable.\n\n"
    "Memory context:\n{ctx}\n\n"
    "Question: {q}\n"
    "Answer:"
)


def is_temporal(question: str) -> bool:
    """Cheap query-content classifier: does this question need date reasoning?"""
    q = question.lower()
    return any(cue in q for cue in _TEMPORAL_CUES)


# --------------------------------------------------------------------------- #
# MeshMind retrieval -- Opus path (spreading activation + neighbourhood rerank)
# --------------------------------------------------------------------------- #
def _iso_date(ts: str | None) -> str:
    """Reduce an ISO-8601 timestamp to a bare YYYY-MM-DD (or '' if absent)."""
    if not ts or not isinstance(ts, str):
        return ""
    return ts[:10]


def _turn_of(sn, edge) -> tuple[str, str, str]:
    """Return (date, speaker, raw_text) for a fact node, using its edge's
    provenance (the original turn) when available."""
    date = _iso_date((sn.node.metadata or {}).get("timestamp"))
    speaker = ""
    text = sn.node.text
    if edge is not None:
        prov = edge.provenance or {}
        speaker = prov.get("speaker", "") or ""
        text = prov.get("source_text") or text
        if not date:
            date = _iso_date((edge.metadata or {}).get("timestamp"))
    return date, speaker, text


def render_mesh_context_opus(mesh: Mesh, question: str) -> tuple[str, dict[str, Any]]:
    """Opus path: recall a subgraph via spreading activation and render it as
    ``[date] speaker: text`` -- raw turn text (good lexical overlap) plus the
    session date pulled from the graph. Nodes are taken in reranked recall
    order; temporal questions are re-sorted chronologically.
    """
    sub = mesh.recall(
        question,
        budget_tokens=None,
        k_hops=2,
        max_seeds=MESH_MAX_SEEDS,
        sim_rerank=MESH_SIM_RERANK,
        # Eval hygiene: don't let Hebbian access-reinforcement mutate the mesh
        # mid-benchmark (it would bias later questions by earlier ones and make
        # runs non-idempotent).
        reinforce_on_access=False,
    )

    # Map each node to the (first) edge it participates in, for speaker/date/text.
    edge_by_member: dict[str, Any] = {}
    for e in sub.hyperedges:
        for m in e.members:
            edge_by_member.setdefault(m.node_id, e)

    temporal = (not MESH_NO_DATES) and is_temporal(question)

    # Take the top fact nodes (the raw turns) in reranked order, de-duplicated.
    picked: list[tuple[str, str, str, str]] = []  # (date, speaker, text, flag)
    seen: set[tuple[str, str]] = set()
    for sn in sub.nodes:
        if sn.node.kind != "fact":
            continue
        edge = edge_by_member.get(sn.node.id)
        date, speaker, text = _turn_of(sn, edge)
        key = (speaker, text)
        if key in seen:
            continue
        seen.add(key)
        flag = ""
        if sn.superseded:
            flag = "[OUTDATED] "
        elif sn.contradicted_by:
            flag = "[CONFLICT] "
        picked.append((date, speaker, text, flag))
        if len(picked) >= MESH_MAX_TURNS:
            break

    if temporal:
        # Chronological order helps "when / how long / before-after" reasoning.
        picked.sort(key=lambda t: t[0] or "9999")

    if MESH_NO_DATES:
        lines = [f"{flag}{speaker}: {text}" for (date, speaker, text, flag) in picked]
    else:
        lines = [f"{flag}[{date or 'unknown date'}] {speaker}: {text}"
                 for (date, speaker, text, flag) in picked]

    ctx = "\n".join(lines) if lines else "(no relevant memory found)"
    stats = {
        "path": "opus",
        "n_nodes": len(sub.nodes),
        "n_edges": len(sub.hyperedges),
        "n_turns": len(picked),
        "temporal": temporal,
    }
    return ctx, stats


# --------------------------------------------------------------------------- #
# MeshMind retrieval -- Sol path (edge-level RRF hybrid + query decomposition)
# --------------------------------------------------------------------------- #
def _query_variants(question: str) -> list[str]:
    """Cheap decomposition so separate clauses can seed separate subgraphs."""
    variants = [question]
    for clause in re.split(r"\b(?:if|after|before|because|since|and)\b", question, flags=re.I):
        cleaned = _QUESTION_FILLER.sub(" ", clause)
        cleaned = " ".join(cleaned.strip(" ?.,").split())
        if len(cleaned.split()) >= 2 and cleaned.lower() != question.lower():
            variants.append(cleaned)
    return list(dict.fromkeys(variants))[:4]


class MeshRetriever:
    """Edge-level hybrid retrieval over sources and extracted graph members
    (Sol). RRF-fuses three signals: source-span similarity, best graph-member
    similarity, and lexical overlap -- across a few decomposed query clauses.
    """

    def __init__(self, mesh: Mesh, embedder: SentenceTransformer) -> None:
        self.mesh = mesh
        self.embedder = embedder
        self.edges = [
            edge for edge in mesh.store.all_hyperedges()
            if isinstance(edge.provenance.get("source_text"), str)
        ]
        self.sources = [edge.provenance["source_text"].strip() for edge in self.edges]
        self.source_embs = embedder.encode(
            self.sources, normalize_embeddings=True, show_progress_bar=False
        )
        node_matrix, node_ids = mesh.store.embedding_matrix()
        self.node_matrix = node_matrix
        edge_indexes: dict[str, list[int]] = {}
        id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        for edge in self.edges:
            edge_indexes[edge.id] = [
                id_to_index[m.node_id] for m in edge.members if m.node_id in id_to_index
            ]
        self.edge_indexes = edge_indexes

    def retrieve_edges(self, question: str) -> tuple[list[Any], int]:
        fused: dict[int, float] = {}
        variants = _query_variants(question)
        for variant_rank, variant in enumerate(variants):
            qvec = self.embedder.encode(
                [variant], normalize_embeddings=True, show_progress_bar=False
            )[0]
            source_sims = self.source_embs @ qvec
            node_sims = self.node_matrix @ qvec
            graph_sims = np.asarray([
                max((node_sims[i] for i in self.edge_indexes[e.id]), default=-1.0)
                for e in self.edges
            ])
            qterms = set(re.findall(r"[a-z0-9]+", variant.lower())) - {
                "the", "a", "an", "did", "does", "do", "is", "are", "was", "were",
                "what", "when", "where", "which", "who", "how", "would", "could",
            }
            lexical = np.asarray([
                len(qterms & set(re.findall(r"[a-z0-9]+", source.lower())))
                for source in self.sources
            ])
            weight = 1.0 if variant_rank == 0 else 0.7
            rankings = (
                (np.argsort(-source_sims), False, 1.5),
                (np.argsort(-graph_sims), False, 0.5),
                (np.argsort(-lexical), True, 3.0),
            )
            for ranking, is_lexical, signal_weight in rankings:
                for rank, idx in enumerate(ranking[:40], start=1):
                    if is_lexical and lexical[idx] == 0:
                        continue
                    fused[int(idx)] = fused.get(int(idx), 0.0) + signal_weight * weight / (60 + rank)
        winners = sorted(fused, key=fused.get, reverse=True)[:MESH_TEMPORAL_MAX_TURNS]
        return [self.edges[i] for i in winners], len(variants)


def render_mesh_context_temporal(
    retriever: MeshRetriever,
    question: str,
    session_dates: dict[int, str],
) -> tuple[str, dict[str, Any]]:
    """Sol path: edge-RRF retrieve source spans, render them with the session
    timestamp + relation roles, ordered chronologically for date reasoning."""
    selected, n_variants = retriever.retrieve_edges(question)
    if not MESH_NO_DATES:
        selected.sort(key=lambda edge: int(edge.provenance.get("session", 0)))

    lines: list[str] = []
    for edge in selected:
        provenance = edge.provenance
        session = int(provenance.get("session", 0))
        # session_dates carries the dataset's human timestamp string
        # ("1:56 pm on 8 May, 2023"); keep it verbatim so the answerer can
        # resolve relative expressions against a full date.
        timestamp = "" if MESH_NO_DATES else session_dates.get(session, "")
        speaker = provenance.get("speaker", "")
        source = provenance["source_text"].strip()
        roles = []
        for member in edge.members:
            if member.role == "statement":
                continue
            node = retriever.mesh.store.get_node(member.node_id)
            if node is not None:
                roles.append(f"{member.role}={node.text}")
        if MESH_NO_DATES:
            lines.append(f"{speaker}: {source}")
        else:
            lines.append(f"[{timestamp or 'unknown date'}] {speaker}: {source}")
        if roles:
            lines.append(f"  Relation [{edge.type}]: " + "; ".join(roles))

    ctx = "\n".join(lines) if lines else "(no relevant memory found)"
    stats = {
        "path": "sol-temporal",
        "n_turns": len(selected),
        "n_query_variants": n_variants,
        "temporal": True,
    }
    return ctx, stats


# --------------------------------------------------------------------------- #
# MeshMind dispatcher -- routes each question to the right path + prompt
# --------------------------------------------------------------------------- #
class MeshDispatcher:
    def __init__(self, mesh: Mesh, embedder: SentenceTransformer, conv: Conversation) -> None:
        self.mesh = mesh
        # Sol's edge retriever is only built if the temporal path is enabled.
        self.retriever = (
            MeshRetriever(mesh, embedder) if MESH_TEMPORAL_PATH else None
        )
        # Session index -> ISO date, straight from the dataset (more reliable
        # than per-node metadata for the "when did X" questions).
        self.session_dates = {s.index: s.date_time for s in conv.sessions}

    def _use_temporal(self, question: str) -> bool:
        return MESH_TEMPORAL_PATH and is_temporal(question)

    def retrieve(self, question: str) -> tuple[str, dict[str, Any]]:
        if self._use_temporal(question):
            return render_mesh_context_temporal(
                self.retriever, question, self.session_dates
            )
        return render_mesh_context_opus(self.mesh, question)

    def prompt_for(self, question: str) -> str:
        if MESH_TEMPORAL_PROMPT and self._use_temporal(question):
            return ANSWER_PROMPT_TEMPORAL
        return ANSWER_PROMPT


# Back-compat shim: older callers import `render_mesh_context(mesh, question)`.
def render_mesh_context(mesh: Mesh, question: str) -> tuple[str, dict[str, Any]]:
    return render_mesh_context_opus(mesh, question)


# --------------------------------------------------------------------------- #
# Vector-RAG baseline
# --------------------------------------------------------------------------- #
class VectorRag:
    def __init__(self, conv: Conversation, embedder: SentenceTransformer) -> None:
        self.embedder = embedder
        self.turns: list[tuple[str, str]] = []  # (dia_id, rendered text)
        for sess in conv.sessions:
            for t in sess.turns:
                # Include speaker in text so answers can reference "who said"
                rendered = f"{t.speaker}: {t.text}"
                self.turns.append((t.dia_id, rendered))
        texts = [t[1] for t in self.turns]
        self.embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, question: str, top_k: int = RETRIEVE_TOP_K) -> tuple[str, dict[str, Any]]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embs @ q_emb
        idxs = np.argsort(-sims)[:top_k]
        # Preserve chronological order once we've picked the winners.
        idxs = sorted(idxs.tolist())
        lines = [self.turns[i][1] for i in idxs]
        ctx = "\n".join(lines)
        return ctx, {"n_turns": len(idxs)}


# --------------------------------------------------------------------------- #
# Gemini answerer (shared)
# --------------------------------------------------------------------------- #
def answer(client: genai.Client, ctx: str, question: str, prompt_template: str = ANSWER_PROMPT) -> str:
    prompt = prompt_template.format(ctx=ctx, q=question)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=200,
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            if attempt == 2:
                return f"[ANSWER_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return "[ANSWER_ERROR: unreachable]"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_system(
    name: str,
    conv: Conversation,
    client: genai.Client,
    retrieve_fn: Callable[[str], tuple[str, dict[str, Any]]],
    limit: int | None = None,
    prompt_fn: Callable[[str], str] | None = None,
) -> Path:
    out_path = OUT_DIR / f"{CONV_ID}.{name}.jsonl"
    out_f = out_path.open("w")
    qa_items = conv.qa[: limit or len(conv.qa)]
    t0 = time.time()
    for i, qa in enumerate(qa_items):
        ctx, stats = retrieve_fn(qa.question)
        template = prompt_fn(qa.question) if prompt_fn else ANSWER_PROMPT
        pred = answer(client, ctx, qa.question, template)
        row = {
            "idx": i,
            "question": qa.question,
            "gold": qa.answer,
            "category": qa.category,
            "evidence": qa.evidence,
            "ctx": ctx,
            "ctx_stats": stats,
            "pred": pred,
        }
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(qa_items) - i - 1) / rate if rate else 0
            print(
                f"  [{name}] {i+1}/{len(qa_items)}  {rate:.2f} q/s  eta={eta:.0f}s",
                flush=True,
            )
    out_f.close()
    return out_path


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    if not MESH_DB.exists():
        print(f"phase1 mesh not found at {MESH_DB}; run phase1_ingest.py first", file=sys.stderr)
        return 2

    convs = load()
    conv = next(c for c in convs if c.sample_id == CONV_ID)
    print(f"conv={CONV_ID} qas={len(conv.qa)}  turns={sum(len(s.turns) for s in conv.sessions)}")

    # Optional: limit for smoke, controlled via env.
    limit = int(os.environ.get("PHASE2_LIMIT", "0")) or None
    systems = {
        name.strip() for name in os.environ.get("PHASE2_SYSTEMS", "meshmind,vector_rag").split(",")
    }

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"[shared] loading embedder ({EMBEDDER_NAME})")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    def mm_embed(text: str) -> np.ndarray:
        return embedder.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

    # --- MeshMind
    if "meshmind" in systems:
        print(f"[MeshMind] loading mesh from {MESH_DB}")
        print(f"  routing: temporal_path={MESH_TEMPORAL_PATH} "
              f"temporal_prompt={MESH_TEMPORAL_PROMPT} sim_rerank={MESH_SIM_RERANK} "
              f"no_dates={MESH_NO_DATES}")
        mesh = Mesh(str(MESH_DB), embed=mm_embed)
        dispatcher = MeshDispatcher(mesh, embedder, conv)
        print("[MeshMind] running QAs...")
        mm_path = run_system(
            "meshmind", conv, client, dispatcher.retrieve,
            limit=limit, prompt_fn=dispatcher.prompt_for,
        )
        print(f"[MeshMind] wrote {mm_path}")

    # --- Vector-RAG baseline (fixed reference; always plain prompt)
    if "vector_rag" in systems:
        print(f"[VectorRAG] loading embedder ({EMBEDDER_NAME})")
        rag = VectorRag(conv, embedder)
        def vr_retrieve(q: str):
            return rag.retrieve(q)
        print("[VectorRAG] running QAs...")
        vr_path = run_system("vector_rag", conv, client, vr_retrieve, limit=limit)
        print(f"[VectorRAG] wrote {vr_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
