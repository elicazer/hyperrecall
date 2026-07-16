# Extractor v2 — design notes

_Worker A · branch `feat/extractor-v2` · 2026-07-16_

## Why

The v1 extractor asks one LLM prompt per turn and emits **one** hyperedge. On
LoCoMo conv-26 that yields 48.7% strict accuracy. Mem0/Zep hit 65–70% because
their extractors are (a) **dense** — many facts per turn — and (b)
**canonicalized** — one real-world entity maps to one node id across the whole
conversation. The hypergraph substrate here is fine; v1 was starving it.

v2 rebuilds ingestion as a **3-pass pipeline** in
`src/meshmind/ingest/extractor_v2.py`. The old extractor is untouched; v2 is
opt-in behind `mesh.ingest_text(..., extractor="v2")` (or an `ExtractorV2`
instance).

## The three passes

**Pass 1 — Entity extraction.** One Gemini call returns *all* entities in a
turn, each typed as one of `Person, Project, Decision, Event, Place, Time,
Artifact, Belief, Preference`, with a short canonical name and one-line
description. The prompt pushes for density (2–6 entities/turn) and forbids
`I/me/you/user` (uses the speaker's real name).

**Pass 2 — Relation extraction.** A second Gemini call, conditioned on the
Pass-1 entity list + the turn text, emits typed N-ary **hyperedges** from a
fixed vocabulary: `Decision, Preference, Action, Statement, Observation,
Question, Contradiction, Supersession, Ownership, Location, TemporalOrder`.
Each hyperedge carries participants with **roles** (subject/object/topic/
location/time/preferred/…), an optional ISO-8601 timestamp, and a one-line
self-contained summary sentence. Target: 3–10 hyperedges per substantive turn.

**Pass 3 — Canonicalization (coreference).** Each extracted entity is matched
against entities already in the mesh:
1. exact **normalized name** under the same type (lowercase, de-possessive,
   punctuation-stripped);
2. exact normalized name under *any* type (Pass 1 sometimes retypes the same
   proper noun — e.g. "Luna" as `Artifact` then `Person`; logged as
   `name-xtype`);
3. **embedding cosine** ≥ `sim_threshold` (0.86) among same-type entities.

A match reuses the existing node id (coreference across turns); otherwise a new
entity node is created. **Every merge decision is logged** via the injectable
`logger` so the graph is debuggable.

## How it's persisted

Per turn we create: one `fact` node holding the raw turn text; one `statement`
node per hyperedge holding its summary; and (new) entity nodes. Each hyperedge
binds `[summary(role=summary), turn(role=source), *entities(role=…)]`, which
guarantees arity ≥ 2 and links every relation back to both its summary and the
verbatim source. A turn therefore yields many recallable statement nodes plus
canonicalized entity nodes — dense *and* connected.

## Results (smoke, not a full benchmark)

- `python -m meshmind.ingest.extractor_v2 --demo` (real Gemini 2.5 Pro) on 4
  sample turns: 3–4 hyperedges/turn, semantic types (Action/Ownership/Question/
  Decision/Preference/Location), and **"Caroline" and "Luna" each resolve to a
  single node id** across all turns.
- `benchmarks/locomo/phase1_ingest_v2.py --limit N` re-ingests conv-26 turns
  end-to-end through v2 without crashing and writes graph + coreference stats to
  `runs/phase1_v2/conv-26.stats.json`. `--mock` runs offline.
- `tests/test_extractor_v2.py` (8 tests) + module doctests pass fully offline.

## Size

`extractor_v2.py` is ~713 lines, slightly over the ~600 soft target. The
overage is the required `--demo` CLI, the deterministic offline heuristic (so
tests never hit the network), two prompts, two JSON schemas, and the validation
dataclasses — not extra logic. Trimming further would mean cutting one of those
load-bearing pieces, so I left the substance intact and only tightened prose.

## Guardrail / provider note

Task guardrail: **no Bedrock/AWS/Anthropic** — extraction must go through
Gemini. v2 calls **Gemini 2.5 Pro**. The project already standardized on the
current unified **`google.genai`** SDK (see v1 `phase1_ingest.py`), not the
older `google-generativeai` package (which isn't installed here). v2 follows
that in-repo convention — same provider, current SDK. `mock_mode` needs no
network at all.

## Open questions / follow-ups

- **Cross-type name merge risk.** Merging on exact name across types is right
  for proper nouns (Luna) but could wrongly merge homographs (a Place
  "Washington" vs a Person "Washington"). LoCoMo is mostly people/pets/places
  with distinct names, so net-positive here; a real embedder in Pass 3 would let
  us tighten this. Worth an ablation.
- **Embedding fallback is weak by default.** The shipped `hash_embed` is
  lexical, not semantic, so Pass-3 step 3 rarely fires usefully. Coreference
  currently rides on name matching. Wiring a real embedding model would catch
  "my border collie" ↔ "Luna" style aliasing that names miss.
- **No description-aware disambiguation yet.** We match on name/type/embedding
  but ignore the Pass-1 description when deciding merges. Descriptions could
  break ties (two different people both named "Alex").
- **Contradiction/Supersession are in the vocab but not yet reconciled.** v2
  can *emit* `Contradiction`/`Supersession` edges, but doesn't resolve them
  against specific prior nodes (that's retrieval/consolidation territory — out
  of scope for this worker).
- **Cost/latency.** Two Gemini 2.5 Pro calls per turn (~conv-26 = 419 turns =
  838 calls). A windowed variant (several turns per call) or batching would cut
  cost; the pass boundary is clean enough to do that without touching Pass 3.
- **Windowing for context.** Pass 1/2 currently see a single turn. A short
  rolling window would help pronoun resolution across adjacent turns.
