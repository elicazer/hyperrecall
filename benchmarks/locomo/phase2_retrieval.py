"""Phase 2: retrieval + answer generation for LoCoMo QAs.

Two systems, same input, same LLM:

* MeshMind: mesh.recall(question) -> subgraph -> render as context
* Vector-RAG (baseline): sentence-transformers embeddings on raw turns ->
  top-k cosine on question -> render selected turns as context

Both answered by Gemini 2.5 Flash with the same prompt template.
Output: runs/phase2/{system}.jsonl (per-question row: question, gold, pred, ctx)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# MeshMind retrieval tuning
MESH_MAX_SEEDS = 16        # semantic+lexical seed nodes to start activation from
MESH_SIM_RERANK = 0.85     # blend weight: query<->node similarity vs activation
MESH_MAX_TURNS = 15        # raw turns rendered into the context

# Ablation toggles (env-overridable so runs stay reproducible from the CLI).
MESH_SIM_RERANK = float(os.environ.get("MESH_SIM_RERANK", MESH_SIM_RERANK))
MESH_NO_DATES = os.environ.get("MESH_NO_DATES") == "1"   # drop timestamp grounding
# Point at an alternate embedded mesh (e.g. bare-text embeddings) for ablation.
if os.environ.get("MESH_EMBED_DB"):
    MESH_DB = Path(os.environ["MESH_EMBED_DB"])

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cue words that mark a question as temporal -> order the context chronologically.
_TEMPORAL_CUES = (
    "when", "how long", "before", "after", "during", "first", "last",
    "earliest", "latest", "recent", "ago", "year", "month", "date", "since",
)


ANSWER_PROMPT = (
    "You are answering questions about a long-term conversation between two people. "
    "Use ONLY the provided memory context. If the context does not contain enough "
    "information to answer, reply exactly: 'I don't know.' "
    "Keep answers concise (typically 1-15 words).\n\n"
    "Memory context:\n{ctx}\n\n"
    "Question: {q}\n"
    "Answer:"
)


# --------------------------------------------------------------------------- #
# MeshMind retrieval
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


def render_mesh_context(mesh: Mesh, question: str) -> tuple[str, dict[str, Any]]:
    """Recall a subgraph for `question` and render it as prompt text.

    The rendering is *turn-centric*: MeshMind's ``fact`` nodes are the original
    conversation turns, so we surface them as ``[date] speaker: text`` — raw
    text (good lexical overlap with gold) plus the session date pulled from the
    graph (which the raw-turn RAG baseline never sees). Nodes are taken in the
    reranked recall order; for temporal questions we re-sort chronologically so
    the model can reason over the timeline.
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

    q_lower = question.lower()
    temporal = (not MESH_NO_DATES) and any(cue in q_lower for cue in _TEMPORAL_CUES)

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
        "n_nodes": len(sub.nodes),
        "n_edges": len(sub.hyperedges),
        "n_turns": len(picked),
        "temporal": temporal,
    }
    return ctx, stats


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
def answer(client: genai.Client, ctx: str, question: str) -> str:
    prompt = ANSWER_PROMPT.format(ctx=ctx, q=question)
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
    retrieve_fn,
    limit: int | None = None,
) -> Path:
    out_path = OUT_DIR / f"{CONV_ID}.{name}.jsonl"
    out_f = out_path.open("w")
    qa_items = conv.qa[: limit or len(conv.qa)]
    t0 = time.time()
    for i, qa in enumerate(qa_items):
        ctx, stats = retrieve_fn(qa.question)
        pred = answer(client, ctx, qa.question)
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

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # --- MeshMind
    print(f"[MeshMind] loading mesh from {MESH_DB}")
    # Seed MeshMind from the SAME embedding space as the baseline. The mesh's
    # stored vectors were backfilled with MiniLM (see backfill_embeddings.py),
    # so the query embedding must be MiniLM too.
    mm_embedder = SentenceTransformer(EMBEDDER_NAME)

    def mm_embed(text: str) -> np.ndarray:
        return mm_embedder.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

    mesh = Mesh(str(MESH_DB), embed=mm_embed)
    def mm_retrieve(q: str):
        return render_mesh_context(mesh, q)
    print("[MeshMind] running QAs...")
    mm_path = run_system("meshmind", conv, client, mm_retrieve, limit=limit)
    print(f"[MeshMind] wrote {mm_path}")

    # --- Vector-RAG baseline
    print(f"[VectorRAG] loading embedder ({EMBEDDER_NAME})")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    rag = VectorRag(conv, embedder)
    def vr_retrieve(q: str):
        return rag.retrieve(q)
    print("[VectorRAG] running QAs...")
    vr_path = run_system("vector_rag", conv, client, vr_retrieve, limit=limit)
    print(f"[VectorRAG] wrote {vr_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
