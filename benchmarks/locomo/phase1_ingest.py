"""Phase 1: Ingest conv-26 (LoCoMo, 419 turns) → build real MeshMind graph.

Uses Gemini 2.5 Flash to extract each turn via the same JSON schema PR #1's
Bedrock tool enforces, converts to ExtractedMemory, and persists into a real
Mesh instance. Writes the mesh to disk so Phase 2 can just load it.

Output: benchmarks/locomo/runs/phase1/conv-26.sqlite  (Mesh)
        benchmarks/locomo/runs/phase1/conv-26.stats.json  (graph stats)
        benchmarks/locomo/runs/phase1/conv-26.log  (per-turn extraction log)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# --- workspace + env ------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # benchmarks/locomo/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # meshmind package
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `harness` package

env_file = Path.home() / ".config" / "openclaw" / "gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types

from meshmind import Mesh
from meshmind.ingest.extractor import (
    RECORD_MEMORY_TOOL,
    ExtractedMemory,
)

# Closed-vocabulary edge types. Any turn must map to EXACTLY one of these.
# Order matches priority for ambiguous cases: earlier types win over later.
EDGE_TYPES = [
    "Correction",       # someone corrects a prior fact (populates supersedes)
    "StateChange",      # entity's state changed (has/got/lost/became/started/stopped)
    "Decision",         # a choice was made ("I'll go with X", "decided to")
    "Preference",       # like/dislike/prefer/favorite
    "Intention",        # future plan ("going to", "will", "planning")
    "Event",            # something happened at a place/time with participants
    "Achievement",      # completed goal, milestone reached
    "Problem",          # someone has an issue/pain/struggle
    "Advice",           # recommendation given from one party to another
    "Question",         # information-seeking utterance (only when substantive)
    "Observation",      # factual statement about the world/self, no state change
    "Emotion",          # explicit affect: happy/sad/anxious/proud/frustrated
    "SharedActivity",   # multiple people did something together
    "Relationship",     # ties between two persons/entities (family, friend, colleague)
    "SmallTalk",        # pure phatic: greetings, thanks, acknowledgments (LOW value)
]

_SYSTEM_PROMPT = (
    "You are MeshMind's ingestion engine. Decompose one natural-language "
    "conversation turn into a hypergraph memory fragment.\n\n"
    "OUTPUT RULES:\n"
    "1. entities: distinct atomic nouns (people, projects, concepts, events, places). "
    "Use the speaker's actual name (e.g. 'Caroline', 'Melanie'), never 'I' or 'user'.\n"
    "2. hyperedge.type: MUST be EXACTLY ONE of these strings (case-sensitive, no variants):\n"
    f"   {', '.join(EDGE_TYPES)}\n"
    "   Do NOT invent new types. Do NOT combine types (no 'ObservationAndInquiry'). "
    "Pick the single best match. If truly nothing fits, use 'Observation'.\n"
    "3. Prefer semantic types (StateChange, Decision, Preference, Problem, Achievement) "
    "over conversational types (SmallTalk, Question). If a turn contains BOTH "
    "acknowledgment AND new information, extract the information type, not SmallTalk.\n"
    "4. participants: each entity gets a role describing how it appears in the edge "
    "(subject, object, target, cause, location, time, preferred, less_preferred, "
    "authority, etc.). Roles are free-form but must be DISTINCT within one edge "
    "when the entities play different parts.\n"
    "5. timestamp: ONLY when the turn states or clearly implies a specific date/time. "
    "Must be full ISO-8601 (e.g. '2026-07-15T14:30:00'). Never a bare year. "
    "If unsure, omit the field.\n"
    "6. supersedes: populate when the turn explicitly corrects a prior fact "
    "(e.g. 'actually it's a lab, not a border collie'). Include a target_hint "
    "describing what earlier claim is being overridden.\n\n"
    "Always call the record_memory tool exactly once."
)

from harness.load import Conversation, Turn, load

MODEL = "gemini-2.5-flash"
OUT_DIR = ROOT / "runs" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Track edge-type coercions (off-vocabulary → Observation)
_COERCED_TYPES: dict[str, int] = {}


def gemini_schema() -> dict:
    """Adapt PR #1 JSON-Schema (uses ``type: [str, null]``) to Gemini's
    single-type + nullable form."""
    schema = copy.deepcopy(RECORD_MEMORY_TOOL["inputSchema"]["json"])

    def _fix(node):
        if isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, list):
                non_null = [x for x in t if x != "null"]
                node["type"] = non_null[0] if non_null else "string"
                if "null" in t:
                    node["nullable"] = True
            for v in node.values():
                _fix(v)
        elif isinstance(node, list):
            for v in node:
                _fix(v)

    _fix(schema)
    return schema


def extract_turn(client: genai.Client, turn: Turn, session_dt: str) -> ExtractedMemory | None:
    """Call Gemini for one turn. Returns None on unrecoverable error."""
    prompt = f"[{turn.speaker} at {session_dt}] {turn.text}"
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            payload = json.loads(resp.text)
            # Coerce off-vocabulary edge type back to Observation and mark it.
            edge = payload.get("hyperedge") or {}
            raw_type = edge.get("type", "")
            if raw_type not in EDGE_TYPES:
                edge["type"] = "Observation"
                payload["hyperedge"] = edge
                _COERCED_TYPES[raw_type] = _COERCED_TYPES.get(raw_type, 0) + 1
            return ExtractedMemory.from_tool_input(
                payload, source_text=turn.text
            )
        except Exception as e:
            if attempt == 2:
                print(f"  ✗ [{turn.dia_id}] gave up after 3 tries: {e}", flush=True)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def ingest_conversation(client: genai.Client, conv: Conversation, mesh: Mesh) -> dict:
    """Run every turn through Gemini and persist into mesh. Returns per-run stats."""
    n_total = sum(len(s.turns) for s in conv.sessions)
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    log_path = OUT_DIR / f"{conv.sample_id}.log"
    log_f = log_path.open("w")

    for sess in conv.sessions:
        for turn in sess.turns:
            memory = extract_turn(client, turn, sess.date_time)
            if memory is None:
                n_fail += 1
                log_f.write(f"{turn.dia_id}\tFAIL\n")
                continue
            # Persist into the mesh via the same path Mesh.ingest_text uses.
            ex = memory.to_extraction(
                provenance={"dia_id": turn.dia_id, "session": sess.index, "speaker": turn.speaker},
            )
            mesh._persist(ex)  # yes, semi-private, but this is exactly what ingest_text does
            n_ok += 1
            log_f.write(
                f"{turn.dia_id}\tOK\t"
                f"{len(memory.entities)}ents\t"
                f"{memory.hyperedge.type if memory.hyperedge else '-'}\t"
                f"sup={len(memory.supersedes)}\tcon={len(memory.contradictions)}\n"
            )
            if (n_ok + n_fail) % 25 == 0:
                elapsed = time.time() - t0
                rate = (n_ok + n_fail) / elapsed
                eta = (n_total - n_ok - n_fail) / rate if rate else 0
                print(
                    f"  progress: {n_ok+n_fail}/{n_total}  ok={n_ok} fail={n_fail}  "
                    f"{rate:.1f} turns/s  eta={eta:.0f}s",
                    flush=True,
                )

    log_f.close()
    return {
        "sample_id": conv.sample_id,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_s": round(time.time() - t0, 1),
    }


def graph_stats(mesh: Mesh) -> dict:
    """Compact graph shape summary."""
    # Reach in via the store — SqliteStore has direct SQL access.
    store = mesh.store
    n_nodes = store._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    n_edges = store._conn.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0]
    n_members = store._conn.execute("SELECT COUNT(*) FROM hyperedge_nodes").fetchone()[0]
    edge_type_rows = store._conn.execute(
        "SELECT type, COUNT(*) FROM hyperedges GROUP BY type ORDER BY 2 DESC"
    ).fetchall()
    node_kind_rows = store._conn.execute(
        "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY 2 DESC"
    ).fetchall()
    # Nodes not touched by any edge
    orphan_row = store._conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE id NOT IN (SELECT node_id FROM hyperedge_nodes)"
    ).fetchone()
    avg_arity = n_members / n_edges if n_edges else 0
    return {
        "nodes": n_nodes,
        "hyperedges": n_edges,
        "members": n_members,
        "avg_arity": round(avg_arity, 2),
        "orphan_nodes": orphan_row[0] if orphan_row else 0,
        "edge_types": {t: c for t, c in edge_type_rows},
        "node_kinds": {k: c for k, c in node_kind_rows},
    }


SCHEMA: dict = gemini_schema()


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2

    convs = load()
    # Just conv-26 for the pilot.
    conv = next(c for c in convs if c.sample_id == "conv-26")
    n_turns = sum(len(s.turns) for s in conv.sessions)
    print(f"Pilot: {conv.sample_id}  {conv.speaker_a}/{conv.speaker_b}  "
          f"{len(conv.sessions)} sessions  {n_turns} turns  {len(conv.qa)} QAs")

    db_path = OUT_DIR / f"{conv.sample_id}.sqlite"
    if db_path.exists():
        db_path.unlink()

    mesh = Mesh(str(db_path))
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    stats_ingest = ingest_conversation(client, conv, mesh)
    stats_graph = graph_stats(mesh)
    out = {
        "ingest": stats_ingest,
        "graph": stats_graph,
        "coerced_types": dict(sorted(_COERCED_TYPES.items(), key=lambda x: -x[1])),
        "coerced_total": sum(_COERCED_TYPES.values()),
        "model": MODEL,
    }

    (OUT_DIR / f"{conv.sample_id}.stats.json").write_text(json.dumps(out, indent=2))
    print("\n=== INGEST ===")
    print(json.dumps(stats_ingest, indent=2))
    print("\n=== GRAPH ===")
    print(json.dumps(stats_graph, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
